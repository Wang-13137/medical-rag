"""基于检索上下文的答案合成（生成端）。

设计上刻意和检索端解耦：本模块只负责「给定若干上下文片段，产出带引用编号的答案」，
不关心片段是怎么召回的。这样检索策略换掉时，生成评测的结果仍然可比。

生成有两个提示词档位：
  - ``strict``：强制只能依据上下文作答、每个事实必须带引用编号、无依据时必须明确拒答。
  - ``naive`` ：给同样的上下文，但不加约束，也不要求引用。
两者构成消融实验，用来量化「约束本身」贡献了多少，而不是模型本来就好。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_STRICT = (
    "你是一个严格的医学指南问答助手。你只能依据用户提供的指南片段作答，"
    "严禁使用片段之外的医学知识、临床经验或推测。"
)

INSTRUCTION_STRICT = """要求：
1. 每个事实陈述后紧跟引用编号，形如 [1]；多个来源写成 [1][3]。
2. 只表述片段中确实写了的内容，不得补充、推断或改写数值、药名、剂量、时间窗。
3. 若片段中没有答案、或内容不足以回答，必须输出这句话：「根据提供的指南片段无法回答该问题。」，然后说明缺少哪方面内容。严禁编造。
4. 答案用中文，控制在 200 字以内，按要点分条输出。"""

# v2：针对 v1 的过度拒答做了修正。观察到 v1 会把「片段用不同措辞说了同一件事」
# 判成无依据（例如 QA073 热性惊厥，片段写的是欧洲共识推荐地西泮肛门给药，
# 与「首选处理」实质相同，v1 仍拒答）。v2 显式允许在片段明确支持下的归纳，
# 只对完全没涉及的情况拒答。
INSTRUCTION_STRICT_V2 = """要求：
1. 每个事实陈述后紧跟引用编号，形如 [1]；多个来源写成 [1][3]。
2. 只表述片段中确实写了的内容，不得补充、推断或改写数值、药名、剂量、时间窗。
3. 判断能否回答时，看的是片段有没有讲到这件事的实质，而不是措辞是否完全一致：
   - 片段用不同说法表达了同一结论、或相关信息散落在不同片段里，应当归纳后作答并给出引用，不要拒答。
   - 只有当片段完全没涉及该问题，才输出这句话：「根据提供的指南片段无法回答该问题。」，并说明缺少哪方面内容。
4. 严禁编造。片段里找不到的数值、药名、推荐等级一律不写；宁可少说也不要说错。
5. 答案用中文，控制在 200 字以内，按要点分条输出。"""

INSTRUCTION_NAIVE = """请回答下面的医学问题，尽量给出具体的数值、药物名称或时间窗。答案用中文，控制在 200 字以内。"""

REFUSAL_MARKERS = (
    "无法回答",
    "没有提供",
    "未提供",
    "不足以",
    "不足以外",
    "资料不足",
    "不知道",
    "无法确定",
    "没有相关",
    "未包含",
    "无法给出",
)


@dataclass
class GenerationResult:
    """一次生成调用的产物。"""

    answer: str
    citations: List[int] = field(default_factory=list)
    refused: bool = False
    prompt_mode: str = "strict"
    model: str = ""
    latency_sec: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "refused": self.refused,
            "prompt_mode": self.prompt_mode,
            "model": self.model,
            "latency_sec": round(self.latency_sec, 3),
            "usage": self.usage,
            "error": self.error,
        }


def parse_citations(text: str, max_index: Optional[int] = None) -> List[int]:
    """从答案文本里抽出引用编号 [1]、[3]，按出现顺序去重。"""
    import re

    found: List[int] = []
    for m in re.finditer(r"\[(\d{1,3})\]", text):
        idx = int(m.group(1))
        if max_index is not None and not (1 <= idx <= max_index):
            continue
        if idx not in found:
            found.append(idx)
    return found


def detect_refusal(text: str) -> bool:
    """判断答案是否属于拒答。取宽松 marker 命中，避免只认我写死的那句话。"""
    compact = text.replace(" ", "")
    return any(marker in compact for marker in REFUSAL_MARKERS)


def build_context_block(contexts: Sequence[str], titles: Optional[Sequence[str]] = None) -> str:
    """把若干片段拼成带编号的上下文块。编号从 1 开始，与引用编号一致。"""
    lines: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        head = f"[{i}]"
        if titles and i - 1 < len(titles) and titles[i - 1]:
            head += f" {titles[i - 1]}"
        lines.append(f"{head}\n{ctx.strip()}")
    return "\n\n".join(lines)


class DeepSeekGenerator:
    """DeepSeek chat 接口的最小封装（HTTP 直连，不引入 SDK）。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout: float = 90.0,
        max_retries: int = 3,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.last_usage: Dict[str, int] = {}
        if not self.api_key:
            raise RuntimeError(
                "缺少 API Key：请设置环境变量 DEEPSEEK_API_KEY（见 README 生成端一节）"
            )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """发一次对话请求，带指数退避重试。"""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    self.last_usage = dict(data.get("usage") or {})
                    choices = data.get("choices") or []
                    return (choices[0].get("message") or {}).get("content", "") or ""
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except httpx.HTTPError as exc:  # 网络类异常才重试
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek 调用失败（重试 {self.max_retries} 次）：{last_err}")

    def answer(
        self,
        query: str,
        contexts: Sequence[str],
        titles: Optional[Sequence[str]] = None,
        prompt_mode: str = "strict",
    ) -> GenerationResult:
        """按指定提示词档位生成答案。"""
        if prompt_mode not in ("strict", "strict_v2", "naive", "closed-book"):
            raise ValueError(f"未知的 prompt_mode：{prompt_mode}")
        block = build_context_block(contexts, titles)
        if prompt_mode == "closed-book":
            # 对照组：完全不给上下文，用来量化「检索这一步本身」贡献了多少
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{INSTRUCTION_NAIVE}\n\n问题：{query}\n\n答案："
                    ),
                }
            ]
        elif prompt_mode in ("strict", "strict_v2"):
            instruction = INSTRUCTION_STRICT_V2 if prompt_mode == "strict_v2" else INSTRUCTION_STRICT
            messages = [
                {"role": "system", "content": SYSTEM_STRICT},
                {
                    "role": "user",
                    "content": (
                        f"{instruction}\n\n指南片段：\n{block}\n\n问题：{query}\n\n答案："
                    ),
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{INSTRUCTION_NAIVE}\n\n指南片段：\n{block}\n\n问题：{query}\n\n答案："
                    ),
                }
            ]
        t0 = time.time()
        text = self.chat(messages)
        latency = time.time() - t0
        max_idx = len(contexts) if prompt_mode != "closed-book" else 0
        result = GenerationResult(
            answer=text.strip(),
            citations=parse_citations(text, max_index=max_idx or None),
            refused=detect_refusal(text),
            prompt_mode=prompt_mode,
            model=self.model,
            latency_sec=latency,
            usage=dict(self.last_usage),
        )
        if self.verbose:
            print(
                f"[gen] mode={prompt_mode} model={self.model} "
                f"latency={latency:.1f}s chars={len(result.answer)} citations={result.citations}",
                flush=True,
            )
        return result


class EchoGenerator:
    """离线替身：把首条上下文原样返回，用于单元测试和演示，不联网。"""

    def __init__(self, model: str = "echo-offline") -> None:
        self.model = model
        self.last_usage: Dict[str, int] = {}

    def answer(
        self,
        query: str,
        contexts: Sequence[str],
        titles: Optional[Sequence[str]] = None,
        prompt_mode: str = "strict",
    ) -> GenerationResult:
        text = contexts[0].strip() if contexts else "根据提供的指南片段无法回答该问题。"
        cited = f"{text} [1]" if contexts else text
        return GenerationResult(
            answer=cited,
            citations=parse_citations(cited, max_index=len(contexts)),
            refused=not contexts,
            prompt_mode=prompt_mode,
            model=self.model,
        )


def load_generator(config_path: Optional[str] = None, cfg: Any = None, verbose: bool = True):
    """按配置构造生成端。``generation.enabled: false`` 时返回离线替身。"""
    if cfg is None:
        from medical_rag.config import load_config

        if config_path is None:
            raise ValueError("必须提供 config_path 或 cfg")
        cfg = load_config(config_path)
    gen_cfg = cfg.get("generation", {})
    enabled = bool(gen_cfg.get("enabled", False))
    if not enabled:
        return EchoGenerator()
    return DeepSeekGenerator(
        model=gen_cfg.get("model", DEFAULT_MODEL),
        base_url=gen_cfg.get("base_url", DEFAULT_BASE_URL),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        max_tokens=int(gen_cfg.get("max_tokens", 800)),
        verbose=verbose,
    )
