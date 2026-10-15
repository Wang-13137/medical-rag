#!/usr/bin/env python
"""在 benchmark v1 上跑 LangGraph 固定编排图，产出与 run_agent.py 同构的 JSONL。

产出字段与 run_generation.py / run_agent.py 对齐，因此可以直接喂给
eval_generation.py 评测，再用 compare_generation.py 与 ReAct 版做配对比较。

    python scripts/run_langgraph.py --limit 30
    python scripts/run_langgraph.py --ids QA001,QA002 --out data/generated/graph.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.generate import (  # noqa: E402
    DeepSeekGenerator,
    EchoGenerator,
    load_generator,
    parse_citations,
)
from medical_rag.index import FaissIndex  # noqa: E402
from medical_rag.langgraph_flow import build_graph  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402
from run_agent import make_keyword_fn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark v1 上的 LangGraph 固定编排图")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--benchmark", default="data/eval/benchmark_v1.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--candidate-k", type=int, default=40)
    ap.add_argument("--max-revisions", type=int, default=1)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--keyword", action="store_true", help="启用 BM25 关键词检索")
    ap.add_argument("--mode", default="strict", choices=["strict", "naive"])
    ap.add_argument("--out", default="data/generated/langgraph.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = json.load(open(args.benchmark, encoding="utf-8"))
    items: List[Dict[str, Any]] = bench["items"]

    sel = set(x.strip() for x in args.ids.split(",") if x.strip())
    if sel:
        items = [q for q in items if q["id"] in sel]
    elif args.limit:
        items = items[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"[resume] 已有 {len(done)} 条，跳过")

    print("[load] 载入索引与检索器 ...", flush=True)
    mode, retriever = make_retriever(cfg, not args.no_rerank, args.candidate_k, True)

    keyword_fn = None
    if args.keyword:
        index = FaissIndex.load(cfg.get("paths.index_path"), cfg.get("paths.meta_path"))
        keyword_fn = make_keyword_fn(index, cfg.get("retrieval.hybrid.cache_path"))

    try:
        generator: Any = load_generator(args.config, cfg, verbose=True)
    except Exception:
        generator = EchoGenerator()
    if not isinstance(generator, DeepSeekGenerator):
        print("[warn] 未配置 API Key，生成器退回 EchoGenerator（只验证链路）")

    graph = build_graph(
        retriever,
        generator,
        keyword_search=keyword_fn,
        reranker=None if args.no_rerank else getattr(retriever, "reranker", None),
        candidate_k=args.candidate_k,
        top_n=args.top_n,
        budget_chars=args.budget,
        max_revisions=args.max_revisions,
        mode=args.mode,
    )

    print(f"[run] 检索={mode}  题数={len(items)}  预算={args.budget}  修订上限={args.max_revisions}", flush=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for item in items:
            qid = item["id"]
            if qid in done:
                continue
            result = graph.run(item["question"])
            row = {
                "id": qid,
                "question": item["question"],
                "type": item.get("type", ""),
                "answer": result["answer"],
                "contexts": result["contexts"],
                "titles": result["titles"],
                "used_chars": result["used_chars"],
                # 与 run_generation.py 对齐的别名，便于共用 eval_generation.py
                "context_titles": result["titles"],
                "context_chars": result["used_chars"],
                "citations": parse_citations(result["answer"], max_index=len(result["contexts"])),
                "answer_keys": item.get("answer_keys") or [],
                "refused": result["refused"],
                "steps": result["trace"],
                "tool_calls": {"llm": result["llm_calls"]},
                "stop_reason": result["stop_reason"],
                "latency_sec": result["latency_sec"],
                "prompt_mode": "langgraph",
                "retrieval": mode,
                "model": cfg.get("generation.model"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            print(
                f"  {qid} 节点{len(result['nodes'])} 修订{result['revisions']} "
                f"拒答{int(result['refused'])} ctx{len(result['contexts'])} "
                f"{result['used_chars']}字 {result['latency_sec']:.1f}s",
                flush=True,
            )
    print(f"[save] {out_path}  新增 {written} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
