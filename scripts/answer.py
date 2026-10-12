#!/usr/bin/env python
"""命令行问答：检索 + 精排 + 生成，输出带引用编号的答案与溯源来源。

用法：
    python scripts/answer.py --q "儿童退热药用多大剂量"
    python scripts/answer.py --q "..." --prompt-mode naive   # 对照：不加约束
    python scripts/answer.py --q "..." --offline              # 不调用 API
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.generate import EchoGenerator, load_generator  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="带引用的指南问答")
    ap.add_argument("--q", required=True, help="要问的问题")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--top-n", type=int, default=0, help="进入上下文的片段数，默认取配置")
    ap.add_argument("--budget", type=int, default=0, help="上下文字符预算，默认取配置")
    ap.add_argument("--prompt-mode", default="strict", choices=["strict", "naive"])
    ap.add_argument("--offline", action="store_true", help="不调用 API，用于冒烟测试")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    top_n = args.top_n or int(cfg.get("generation.top_n", 8))
    budget = args.budget or int(cfg.get("generation.budget_chars", 2400))
    candidate_k = int(cfg.get("rerank.candidate_k", 20))

    retriever_mode, retriever = make_retriever(
        cfg, not args.no_rerank, candidate_k, verbose=False
    )
    generator: Any = EchoGenerator() if args.offline else load_generator(cfg=cfg, verbose=False)

    hits = retriever.retrieve(args.q, top_k=max(candidate_k, top_n))
    texts: List[str] = []
    titles: List[str] = []
    sources: List[str] = []
    used = 0
    for h in hits:
        t = h.chunk.text
        if used + len(t) > budget and texts:
            break
        texts.append(t)
        titles.append(f"来源：{h.chunk.doc_name}")
        sources.append(h.chunk.citation())
        used += len(t)
        if len(texts) >= top_n:
            break

    res = generator.answer(args.q, texts, titles, args.prompt_mode)

    print(f"[answer] 问题：{args.q}")
    print(f"[answer] 检索={retriever_mode}  生成={type(generator).__name__}  "
          f"提示词={args.prompt_mode}  上下文 {used} 字符 / {len(texts)} 片")
    print("-" * 78)
    print(res.answer)
    print("-" * 78)
    cited = [sources[i - 1] for i in res.citations if 1 <= i <= len(sources)]
    if cited:
        print("[answer] 引用来源：")
        for i, s in enumerate(cited, start=1):
            print(f"    {i}. {s}")
    else:
        print("[answer] 该答案未附引用编号")
    print(f"[answer] 判为拒答：{'是' if res.refused else '否'}  延迟 {res.latency_sec:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
