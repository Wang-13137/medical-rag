#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标定逐句蕴含校验的覆盖率阈值。

为什么必须标定
--------------
verify.py 的字面覆盖率是一把可调的尺子：阈值低了漏掉幻觉，高了把「换种说法复述」
的正确句子误判成编造（而后者正是 RAG 的价值所在）。拍脑袋定一个数没有说服力。

做法（完全离线、可复现）：
- 正样本：已生成的 strict 答案里的句子。在受限提示词下这些句子绝大多数有依据，
  用来量「误报率」。
- 负样本 A（数字型幻觉）：把句子里的一个数字换成语料里不可能出现的值。
- 负样本 B（叙述型幻觉）：把 A 问题的答案句子嫁接到 B 问题的上下文里，
  模拟「张冠李戴」。会先过滤掉套话型句子（在别人上下文里覆盖率也高的）。

然后扫描阈值，报告误报率与两类检测率的权衡，并给出推荐值。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.verify import (  # noqa: E402
    coverage_score,
    extract_numbers,
    is_claim_sentence,
    make_polluted_sentence,
    split_answer_sentences,
)

FIELDS = ("threshold", "fpr_positive", "detect_numeric", "detect_narrative", "detect_all", "f1_all")


def load_sentences(path: Path, max_records: int = 0) -> List[Dict]:
    """从生成结果里抽 (句子, 上下文) 样本。"""
    samples: List[Dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "unanswerable":
                continue
            answer = record.get("answer") or ""
            if not answer or "无法回答" in answer.replace(" ", ""):
                continue
            contexts = record.get("contexts") or []
            if not contexts:
                continue
            for sentence in split_answer_sentences(answer):
                # 只保留事实陈述句：引导语与拒答语在语料里必然无依据，混进来会污染正样本
                if not is_claim_sentence(sentence):
                    continue
                samples.append(
                    {
                        "qid": record.get("qid") or record.get("id") or "",
                        "sentence": sentence,
                        "contexts": contexts,
                    }
                )
            if max_records and len(samples) >= max_records:
                break
    return samples


def build_polluted(samples: Sequence[Dict], seed: int) -> List[Dict]:
    """负样本 A：数字型幻觉。"""
    rng = random.Random(seed)
    polluted: List[Dict] = []
    for index, sample in enumerate(samples):
        sentence = sample["sentence"]
        if not extract_numbers(sentence):
            continue
        new_sentence, replaced = make_polluted_sentence(sentence, seed=index)
        if not replaced:
            continue
        polluted.append({**sample, "sentence": new_sentence, "pollution": "numeric"})
    rng.shuffle(polluted)
    return polluted


def build_mismatched(
    samples: Sequence[Dict], seed: int, keep_quantile: float = 0.5, min_self_coverage: float = 0.4
) -> Tuple[List[Dict], int]:
    """负样本 B：句子嫁接到无关上下文。返回 (样本, 被当作套话剔除的数量)。

    中文 3-gram 覆盖率天生偏高（常用字组合重合多），写死一个阈值会筛不出样本。
    这里改成两步：
    1. foreign 句在**自己的**上下文里覆盖率要够高（否则它本身就是幻觉句，不配当标准负样本）；
    2. 在候选里按「在别人上下文里的覆盖率」从低到高排序，只保留最低的那部分，
       剩下的才是真正的「张冠李戴」；覆盖率高的属于套话（"建议定期复查"），剔掉。
    """
    rng = random.Random(seed + 1)
    pool = list(samples)
    candidates: List[Tuple[float, Dict]] = []
    dropped = 0
    for sample in samples:
        other = pool[rng.randrange(len(pool))]
        if other["qid"] == sample["qid"]:
            continue
        foreign = other["sentence"]
        self_cov = coverage_score(foreign, "\n".join(other["contexts"]))
        if self_cov < min_self_coverage:
            dropped += 1
            continue
        foreign_cov = coverage_score(foreign, "\n".join(sample["contexts"]))
        candidates.append(
            (
                foreign_cov,
                {
                    "qid": sample["qid"],
                    "sentence": foreign,
                    "contexts": sample["contexts"],
                    "pollution": "narrative",
                },
            )
        )
    candidates.sort(key=lambda item: item[0])
    keep = max(1, int(len(candidates) * keep_quantile))
    dropped += len(candidates) - keep
    return [item[1] for item in candidates[:keep]], dropped


def evaluate_at(samples: Sequence[Dict], threshold: float) -> Dict[str, float]:
    """给定阈值，返回误报率与检测率。"""
    def detect(group: Sequence[Dict]) -> float:
        if not group:
            return float("nan")
        hits = sum(
            1
            for s in group
            if coverage_score(s["sentence"], "\n".join(s["contexts"])) < threshold
            or _has_unsupported_number(s["sentence"], s["contexts"])
        )
        return hits / len(group)

    positives = [s for s in samples if s.get("pollution") is None]
    numeric = [s for s in samples if s.get("pollution") == "numeric"]
    narrative = [s for s in samples if s.get("pollution") == "narrative"]
    all_neg = numeric + narrative

    fpr = detect(positives)
    d_num = detect(numeric)
    d_nar = detect(narrative)
    d_all = detect(all_neg)
    # F1：把「检测出负样本」当召回，「不误报正样本」当精确率
    precision = (1 - fpr) * len(positives) / ((1 - fpr) * len(positives) + (1 - d_all) * len(all_neg)) if all_neg else float("nan")
    recall = d_all
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else 0.0
    )
    return {
        "threshold": threshold,
        "fpr_positive": fpr,
        "detect_numeric": d_num,
        "detect_narrative": d_nar,
        "detect_all": d_all,
        "f1_all": f1,
    }


def _has_unsupported_number(sentence: str, contexts: Sequence[str]) -> bool:
    from medical_rag.verify import normalize

    pool = normalize("\n".join(contexts))
    return any(normalize(num) not in pool for num in extract_numbers(sentence))


def main() -> int:
    ap = argparse.ArgumentParser(description="标定逐句蕴含校验阈值")
    ap.add_argument("--generated", default="data/generated/generation_strict.jsonl")
    ap.add_argument("--out", default="reports/verifier_calibration.json")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--min-threshold", type=float, default=0.30)
    ap.add_argument("--max-threshold", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-fpr", type=float, default=0.10, help="推荐阈值时允许的误报率上限")
    args = ap.parse_args()

    path = ROOT / args.generated
    if not path.exists():
        print(f"[calib] 找不到生成结果：{path}")
        return 1

    positives = load_sentences(path)
    numeric = build_polluted(positives, args.seed)
    narrative, dropped = build_mismatched(positives, args.seed)
    print(
        f"[calib] 正样本 {len(positives)} 句 / 数字型负样本 {len(numeric)} 句 / "
        f"叙述型负样本 {len(narrative)} 句（剔除套话 {dropped} 句）"
    )

    all_samples = list(positives) + numeric + narrative
    rows: List[Dict[str, float]] = []
    threshold = args.min_threshold
    while threshold <= args.max_threshold + 1e-9:
        rows.append(evaluate_at(all_samples, round(threshold, 3)))
        threshold += args.step

    print("\n阈值  正样本误报  数字型检出  叙述型检出  整体检出   F1")
    for row in rows:
        print(
            f"{row['threshold']:.2f}   {row['fpr_positive']:.3f}      "
            f"{row['detect_numeric']:.3f}      {row['detect_narrative']:.3f}      "
            f"{row['detect_all']:.3f}     {row['f1_all']:.3f}"
        )

    eligible = [row for row in rows if row["fpr_positive"] <= args.max_fpr]
    best_f1 = max(rows, key=lambda r: r["f1_all"])
    best = max(eligible, key=lambda r: r["detect_all"]) if eligible else best_f1
    print(
        f"\n[calib] F1 最高阈值 = {best_f1['threshold']:.2f}（误报 {best_f1['fpr_positive']:.3f}，"
        f"检出 {best_f1['detect_all']:.3f}）"
    )
    print(
        f"[calib] 推荐阈值（误报率 ≤ {args.max_fpr:.2f} 下检出最高）= {best['threshold']:.2f}"
        f"（误报 {best['fpr_positive']:.3f}，数字型 {best['detect_numeric']:.3f}，"
        f"叙述型 {best['detect_narrative']:.3f}）"
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated": str(args.generated),
                "seed": args.seed,
                "counts": {
                    "positive": len(positives),
                    "numeric_negative": len(numeric),
                    "narrative_negative": len(narrative),
                    "dropped_generic": dropped,
                },
                "best_f1": best_f1,
                "recommended": best,
                "grid": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[calib] 报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
