#!/usr/bin/env python
"""在 benchmark v1 上跑 ReAct 智能体，产出与 run_generation.py 同构的 JSONL。

两种模式：
- 在线：chat_fn 走 DeepSeek，Agent 自主决定检索 / 精排 / 校验 / 拒答。
- 离线：用脚本化策略代替模型（检索 -> read -> finish），只验证工程链路，
  不产生任何可引用的质量数字。

产出字段与 run_generation.py 对齐，因此 eval_generation.py 可直接评测。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.agent import make_agent  # noqa: E402
from medical_rag.generate import DeepSeekGenerator, load_generator  # noqa: E402
from medical_rag.index import FaissIndex  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402

IDS_RE = re.compile(r"编号 \[([^\]]*)\]")


def make_keyword_fn(index: FaissIndex, cache_path: str):
    from medical_rag.hybrid import BM25Store

    store = BM25Store.load(list(index.chunks), cache_path)

    def _search(query: str, top_k: int) -> List[Dict[str, Any]]:
        pairs = store.retrieve(query, int(top_k))
        items = []
        for idx, score in pairs:
            chunk = index.chunks[int(idx)]
            head = " > ".join(chunk.heading_path or [])
            items.append(
                {
                    "text": chunk.text,
                    "title": f"来源：{chunk.doc_name}" + (f" · {head}" if head else ""),
                    "score": float(score),
                }
            )
        return items

    return _search


def offline_policy(messages: List[Dict[str, str]]) -> str:
    """确定性替身：检索 -> 全读 -> 收尾。只为跑通链路，不代表模型能力。"""
    question = ""
    for m in messages:
        if m["role"] == "user" and m["content"].startswith("问题："):
            question = m["content"][3:].strip()
    last_obs = messages[-1]["content"] if messages else ""
    if "观察" not in last_obs:
        return f"思考：先做语义检索\n行动：vector_search\n行动输入：query={question}, top_k=8"
    if "已读入" not in last_obs:
        m = IDS_RE.search(last_obs)
        ids = m.group(1) if m else "1"
        return f"思考：把相关片段读进 context\n行动：read\n行动输入：ids={ids}"
    body = last_obs.split("当前 context")[-1][:200]
    return f"思考：离线替身不做生成\n行动：finish\n行动输入：answer=离线链路冒烟：{body}"


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark v1 上的 ReAct 智能体")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--benchmark", default="data/eval/benchmark_v1.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument(
        "--min-read",
        type=int,
        default=0,
        help="finish 前至少读入多少条片段（欠召回护栏，0 表示不设限）",
    )
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--candidate-k", type=int, default=40)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--keyword", action="store_true", help="启用 BM25 关键词检索工具")
    ap.add_argument("--offline", action="store_true", help="脚本化替身，不调用 API")
    ap.add_argument("--out", default="")
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

    out_path = Path(
        args.out
        or f"data/generated/agent_{'offline' if args.offline else 'react'}.jsonl"
    )
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
    reranker = None
    if args.keyword:
        index = FaissIndex.load(cfg.get("paths.index_path"), cfg.get("paths.meta_path"))
        keyword_fn = make_keyword_fn(index, cfg.get("retrieval.hybrid.cache_path"))
    if not args.no_rerank:
        reranker = getattr(retriever, "reranker", None)

    chat_fn = offline_policy
    if not args.offline:
        gen = load_generator(args.config, cfg, verbose=True)
        if isinstance(gen, DeepSeekGenerator):
            chat_fn = gen.chat
        else:
            print("[warn] 未配置 API Key，退回离线替身")
            chat_fn = offline_policy

    print(f"[run] 检索={mode}  题数={len(items)}  预算={args.budget}  步数上限={args.max_steps}", flush=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for item in items:
            qid = item["id"]
            if qid in done:
                continue
            agent, _ = make_agent(
                chat_fn,
                retriever,
                keyword_search=keyword_fn,
                reranker=reranker,
                max_steps=args.max_steps,
                budget_chars=args.budget,
                min_read=args.min_read,
            )
            run = agent.run(item["question"])
            row = run.to_dict()
            row.update(
                {
                    "id": qid,
                    "question": item["question"],
                    "type": item.get("type", ""),
                    "answer_keys": item.get("answer_keys") or [],
                    "prompt_mode": "agent_offline" if args.offline else "agent_react",
                    "retrieval": mode,
                    "model": "offline-policy" if args.offline else cfg.get("generation.model"),
                }
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            print(
                f"  {qid} 步{len(run.steps)} "
                f"工具{sum(run.tool_calls.values())} 拒答{int(run.refused)} "
                f"ctx{len(run.contexts)} {run.used_chars}字 {run.latency_sec:.1f}s",
                flush=True,
            )
    print(f"[save] {out_path}  新增 {written} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
