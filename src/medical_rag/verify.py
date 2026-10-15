# -*- coding: utf-8 -*-
"""生成后的逐句蕴含校验：把「不要编造」从提示词软约束升级为可执行的硬闸门。

为什么需要这一层
----------------
提示词里写「无依据必须拒答」是软约束：模型是否遵守不可控，且无法度量。
本模块在生成之后独立复查，判据只看两件事：

1. **数字溯源**：句子里出现的每个数字，必须能在它自己声明引用的片段里逐字找到。
   数值型幻觉（剂量、阈值、时间窗）在医疗场景代价最高，也最容易规则化。
2. **字面覆盖**：句子的字符 3-gram 有多少比例能在被引用片段中找到。
   用于拦叙述性编造（把 A 药的用法安到 B 病上），数字检查覆盖不到这一类。

两条判据都不依赖模型，因此完全可复现、零调用成本，代价是精度有限：
它只能判「字面有没有依据」，判不了语义等价（同义改写会被误判为无依据）。
阈值不是拍脑袋定的，由 scripts/calibrate_verifier.py 在合成污染样本上扫描得出。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

CITE_RE = re.compile(r"\[(\d{1,3})\]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# 答案里的列表序号 / 项目符号，参与覆盖率计算会系统性拉低分数
BULLET_RE = re.compile(r"(?m)^\s*(?:[-*·•]|\d{1,2}[.、)])\s*")

# 句末标点，与 chunking 保持一致（不含英文句点，避免剖开剂量）
_SENTENCE_BREAKS = "。！？；!?;\n"

# 由 scripts/calibrate_verifier.py 在合成污染样本上扫描得出：
# 阈值 0.60 时正样本误报率 6.5%，数字型幻觉检出 91.0%，叙述型幻觉检出 100%。
# 调高到 0.85 能把检出提到 97.5%，但误报率涨到 23.1%，得不偿失。
DEFAULT_COVERAGE_THRESHOLD = 0.60
# 删掉无依据句子后，剩余内容少于原答案的该比例就整条拒答，避免输出残缺答案
DEFAULT_MIN_KEEP_RATIO = 0.4

REFUSAL_TEXT = "根据提供的指南片段无法回答该问题。"

# 模型固定生成的引导语与拒答语。这些句子在语料里必然找不到字面依据，
# 但它们不是「事实陈述」，不参与依据校验——否则会把拒答和开场白全部误判成编造。
# 实测：打开这条过滤前，正样本误报率约 23%，其中绝大多数是这两类句子。
NON_CLAIM_RE = re.compile(
    r"(根据提供的指南片段|根据给定(?:的)?(?:指南)?片段|以下(?:是|内容|为)|综上|说明[：:]|注[：:]|来源[：:])"
)
REFUSAL_RE = re.compile(r"(无法回答|无法确定|未提供|没有提供|不足以外|资料不足|无法给出|没有相关|不知道)")

# 短于该长度的句子不构成独立事实陈述（多为分条小标题）
MIN_CLAIM_CHARS = 12


def is_claim_sentence(sentence: str) -> bool:
    """判断一个句子是不是需要校验依据的「事实陈述」。

    引导语、拒答语、过短的标题行一律不算，它们本来就不该在语料里有对应原文。
    """
    plain = strip_citations(sentence).strip()
    if len(plain) < MIN_CLAIM_CHARS:
        return False
    compact = plain.replace(" ", "")
    if NON_CLAIM_RE.search(compact) or REFUSAL_RE.search(compact):
        return False
    return True


def normalize(text: str) -> str:
    """NFKC + 去空白 + 小写：PDF 抽取会塞空格，网页版常用全角字符。"""
    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"\s+", "", value)
    return value.lower()


def strip_citations(text: str) -> str:
    """剥掉引用编号与项目符号，只留事实文本。"""
    return BULLET_RE.sub("", CITE_RE.sub(" ", text or "")).strip()


def extract_numbers(text: str) -> List[str]:
    """抽事实性数字：先剥引用编号和列表序号，避免 [1]、(1) 被当成剂量。"""
    cleaned = CITE_RE.sub(" ", text or "")
    cleaned = re.sub(r"(?m)^\s*\d{1,2}[.、)]\s*", " ", cleaned)
    cleaned = re.sub(r"[（(]\s*\d{1,2}\s*[)）]", " ", cleaned)
    return NUM_RE.findall(cleaned)


def split_answer_sentences(answer: str) -> List[str]:
    """把答案拆成句子（保留引用编号，便于定位每句声称的来源）。"""
    spans: List[str] = []
    start = 0
    index = 0
    total = len(answer)
    while index < total:
        if answer[index] in _SENTENCE_BREAKS:
            end = index + 1
            while end < total and answer[end] in "”’」』》）】〉]}\"'":
                end += 1
            piece = answer[start:end].strip()
            if piece:
                spans.append(piece)
            index = end
            start = end
        else:
            index += 1
    tail = answer[start:].strip()
    if tail:
        spans.append(tail)
    return spans


def _ngrams(text: str, n: int = 3) -> set:
    compact = normalize(strip_citations(text))
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def coverage_score(sentence: str, context: str, n: int = 3) -> float:
    """句子字符 n-gram 被上下文覆盖的比例。"""
    grams = _ngrams(sentence, n)
    if not grams:
        return 1.0
    pool = _ngrams(context, n)
    return len(grams & pool) / len(grams)


@dataclass
class SentenceVerdict:
    """单句校验结论。

    ``supported`` 判的是「有没有编造」，依据集是送进去的**全部上下文**；
    ``weak_citation`` 单独判「引用标得准不准」，依据集是**该句自己标注的片段**。
    两者刻意分开：引用标错属于溯源质量问题，不该把整句当幻觉删掉。
    """

    text: str
    citations: List[int] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    coverage: float = 0.0
    cited_coverage: float = 0.0
    supported: bool = True
    weak_citation: bool = False
    is_claim: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "citations": self.citations,
            "numbers": self.numbers,
            "unsupported_numbers": self.unsupported_numbers,
            "coverage": round(self.coverage, 4),
            "cited_coverage": round(self.cited_coverage, 4),
            "supported": self.supported,
            "weak_citation": self.weak_citation,
            "is_claim": self.is_claim,
            "reasons": self.reasons,
        }


@dataclass
class VerificationResult:
    """整条答案的校验结论与闸门动作结果。"""

    answer: str
    verdicts: List[SentenceVerdict] = field(default_factory=list)
    unsupported_sentences: List[str] = field(default_factory=list)
    mean_coverage: float = 0.0
    action: str = "pass"  # pass | flagged | pruned | refused
    revised_answer: str = ""
    refused: bool = False
    dropped_chars: int = 0
    weak_citation_sentences: List[str] = field(default_factory=list)

    @property
    def claims(self) -> List[SentenceVerdict]:
        """只含事实陈述句，引导语与拒答语不计入。"""
        return [v for v in self.verdicts if v.is_claim]

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "refused": self.refused,
            "mean_coverage": round(self.mean_coverage, 4),
            "num_sentences": len(self.verdicts),
            "num_claims": len(self.claims),
            "num_unsupported": len(self.unsupported_sentences),
            "unsupported_sentence_rate": (
                round(len(self.unsupported_sentences) / len(self.claims), 4) if self.claims else 0.0
            ),
            "num_weak_citation": len(self.weak_citation_sentences),
            "weak_citation_rate": (
                round(len(self.weak_citation_sentences) / len(self.claims), 4) if self.claims else 0.0
            ),
            "dropped_chars": self.dropped_chars,
            "revised_answer": self.revised_answer,
            "unsupported_sentences": self.unsupported_sentences[:5],
            "weak_citation_sentences": self.weak_citation_sentences[:5],
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def verify_sentence(
    sentence: str,
    contexts: Sequence[str],
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    n: int = 3,
    require_cited_support: bool = False,
) -> SentenceVerdict:
    """校验单句是否有依据，并单独判断引用标得准不准。

    ``require_cited_support=False``（默认）时，判定「编造」只看全部上下文：
    内容在别的片段里找得到就算有依据，引用标错只是溯源质量问题。
    设为 True 则把「引用片段支持不了该句」也当成不支持，用于最严格的口径。
    """
    cites: List[int] = []
    for match in CITE_RE.finditer(sentence):
        idx = int(match.group(1))
        if idx not in cites:
            cites.append(idx)
    valid_cites = [i for i in cites if 1 <= i <= len(contexts)]
    cited_text = "\n".join(contexts[i - 1] for i in valid_cites) if valid_cites else ""
    full_text = "\n".join(contexts)

    if not is_claim_sentence(sentence):
        # 引导语 / 拒答语 / 分条小标题：不构成事实陈述，直接放行
        return SentenceVerdict(
            text=sentence,
            citations=cites,
            coverage=coverage_score(sentence, full_text, n=n),
            cited_coverage=0.0,
            supported=True,
            weak_citation=False,
            is_claim=False,
            reasons=["非事实陈述句，不参与依据校验"],
        )

    numbers = extract_numbers(sentence)
    # 判定编造：以全部上下文为依据集
    basis_text = cited_text if require_cited_support and valid_cites else full_text
    basis_compact = normalize(basis_text)
    unsupported_numbers = [num for num in numbers if normalize(num) not in basis_compact]

    coverage = coverage_score(sentence, full_text, n=n)
    cited_coverage = coverage_score(sentence, cited_text, n=n) if cited_text else 0.0

    # 引用质量：内容在全上下文有依据，却不在自己标注的片段里 → 引用不准
    numbers_in_full = [num for num in numbers if normalize(num) in normalize(full_text)]
    weak_citation = bool(cites) and (
        not valid_cites
        or cited_coverage < coverage_threshold
        or (
            numbers_in_full
            and any(normalize(num) not in normalize(cited_text) for num in numbers_in_full)
        )
    )

    reasons: List[str] = []
    if cites and not valid_cites:
        reasons.append("引用编号越界，指向不存在的片段")
    if not cites:
        reasons.append("该句未标注引用编号")
    if unsupported_numbers:
        reasons.append(f"数字 {unsupported_numbers} 在上下文中找不到")
    if coverage < coverage_threshold:
        reasons.append(f"字面覆盖率 {coverage:.2f} 低于阈值 {coverage_threshold:.2f}")
    if weak_citation:
        reasons.append("引用编号未指向支撑该句的片段（引用不准，非编造）")

    supported = (
        not unsupported_numbers
        and coverage >= coverage_threshold
        and not (cites and not valid_cites)
    )
    if require_cited_support and weak_citation:
        supported = False
    return SentenceVerdict(
        text=sentence,
        citations=cites,
        numbers=numbers,
        unsupported_numbers=unsupported_numbers,
        coverage=coverage,
        cited_coverage=cited_coverage,
        supported=supported,
        weak_citation=weak_citation,
        reasons=reasons,
    )


def verify_answer(
    answer: str,
    contexts: Sequence[str],
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    mode: str = "prune",
    min_keep_ratio: float = DEFAULT_MIN_KEEP_RATIO,
    n: int = 3,
    require_cited_support: bool = False,
) -> VerificationResult:
    """校验整条答案并执行闸门动作。

    require_cited_support=True 时，连「引用不准」也算不支持（更严格的口径）。

    mode:
      - ``flag``  ：不改答案，只打标记（用于先观察、不介入）。
      - ``prune`` ：删掉无依据的句子；剩余内容不足则整条拒答。
      - ``refuse``：只要出现任何无依据句子，整条改判拒答。
    """
    if mode not in ("flag", "prune", "refuse"):
        raise ValueError(f"未知的闸门模式：{mode}")

    # 本来就是拒答的答案：纠正的对象是「编造」，不是「拒答」，直接放行
    plain = strip_citations(answer)
    if REFUSAL_RE.search(plain.replace(" ", "")) and len(plain) < 200:
        return VerificationResult(
            answer=answer,
            verdicts=[],
            action="pass",
            revised_answer=answer,
            refused=False,
        )

    sentences = split_answer_sentences(answer)
    verdicts = [
        verify_sentence(
            sentence,
            contexts,
            coverage_threshold=coverage_threshold,
            n=n,
            require_cited_support=require_cited_support,
        )
        for sentence in sentences
    ]
    claims = [v for v in verdicts if v.is_claim]
    unsupported = [v.text for v in claims if not v.supported]
    weak = [v.text for v in claims if v.weak_citation]
    mean_coverage = sum(v.coverage for v in claims) / len(claims) if claims else 0.0

    result = VerificationResult(
        answer=answer,
        verdicts=verdicts,
        unsupported_sentences=unsupported,
        mean_coverage=mean_coverage,
        action="pass" if not unsupported else "flagged",
        revised_answer=answer,
        weak_citation_sentences=weak,
    )

    if not unsupported:
        return result

    if mode == "flag":
        return result

    if mode == "refuse":
        result.action = "refused"
        result.refused = True
        result.revised_answer = REFUSAL_TEXT
        result.dropped_chars = len(answer)
        return result

    # prune：保留有依据的句子
    kept = [v.text for v in verdicts if v.supported]
    kept_text = "".join(kept)
    original_chars = len(strip_citations(answer))
    kept_chars = len(strip_citations(kept_text))
    if not kept_text or (original_chars and kept_chars / original_chars < min_keep_ratio):
        result.action = "refused"
        result.refused = True
        result.revised_answer = REFUSAL_TEXT
        result.dropped_chars = len(answer)
        return result

    result.action = "pruned"
    result.revised_answer = kept_text
    result.dropped_chars = len(answer) - len(kept_text)
    return result


def verify_record(
    record: Dict,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    mode: str = "prune",
    min_keep_ratio: float = DEFAULT_MIN_KEEP_RATIO,
    require_cited_support: bool = False,
) -> Dict:
    """对一条生成记录（含 answer / contexts）做校验，返回可序列化的结论。"""
    answer = record.get("answer") or ""
    contexts = record.get("contexts") or []
    result = verify_answer(
        answer,
        contexts,
        coverage_threshold=coverage_threshold,
        mode=mode,
        min_keep_ratio=min_keep_ratio,
        require_cited_support=require_cited_support,
    )
    payload = result.to_dict()
    payload["question"] = record.get("question", "")
    payload["qid"] = record.get("qid") or record.get("id") or ""
    payload["type"] = record.get("type", "")
    return payload


def make_polluted_sentence(sentence: str, seed: int = 0) -> Tuple[str, str]:
    """把句子中的一个数字替换成明显不存在的值，产出 (污染后句子, 被替换的数字)。

    用于标定阈值：合成负样本的「数字型幻觉」。
    """
    numbers = extract_numbers(sentence)
    if not numbers:
        return sentence, ""
    target = numbers[seed % len(numbers)]
    replacement = _bogus_number(target, seed)
    polluted = sentence.replace(target, replacement, 1)
    return polluted, target


def _bogus_number(original: str, seed: int) -> str:
    """造一个与原数字同形但几乎不可能出现在语料里的值。"""
    if "." in original:
        base = float(original)
        return f"{base + 7.3 + (seed % 5) * 0.1:.1f}"
    value = int(original)
    return str(value * 13 + 7 + seed % 11)
