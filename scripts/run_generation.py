#!/usr/bin/env python
"""在 benchmark v1 上批量生成答案。

产出一份 JSONL，每行一条：召回上下文快照 + 生成的答案 + 引用编号 + 拒答标记。
上下文一并落盘，是为了让后续评测完全离线可复核——任何人拿到这个文件
都能重新算幻觉率，不必重复调用 API。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.generate import EchoGenerator, load_generator  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402


def build_contexts(
    hits: Sequence, budget: int, top_n: int
) -> tuple[List[str], List[str], int]:
    """按顺序把切片塞进预算。返回 (正文列表, 溯源标题列表, 实际字符数)。"""
    texts: List[str] = []
    titles: List[str] = []
    used = 0
    for h in hits:
        t = h.chunk.text
        if used + len(t) > budget and texts:
            break
        texts.append(t)
        chunk = h.chunk
        head = " > ".join(chunk.heading_path or [])
        titles.append(f"来源：{chunk.doc_name}" + (f" · {head}" if head else ""))
        used += len(t)
        if len(texts) >= top_n:
            break
    return texts, titles, used


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark v1 答案生成")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--benchmark", default="data/eval/benchmark_v1.json")
    ap.add_argument("--prompt-mode", default="strict",
                    choices=["strict", "strict_v2", "naive", "closed-book"])
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--candidate-k", type=int, default=40)
    ap.add_argument("--no-rerank", action="store_true", help="不做精排，用纯向量召回")
    ap.add_argument("--offline", action="store_true", help="用离线替身生成，不调用 API")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--ids", default="", help="只跑指定题号，逗号分隔")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true",
                    help="输出文件已存在时跳过其中已有的题号，继续补齐（API 中断后可续跑）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = json.load(open(args.benchmark, encoding="utf-8"))
    items: List[Dict[str, Any]] = bench["items"]

    sel = set(x.strip() for x in args.ids.split(",") if x.strip())
    if sel:
        items = [q for q in items if q["id"] in sel]
    elif args.limit:
        items = items[: args.limit]

    closed_book = args.prompt_mode == "closed-book"
    if closed_book:
        # 闭卷对照不需要检索，跳过加载索引与 2.2GB 的精排模型
        mode, retriever = "无（闭卷）", None
    else:
        mode, retriever = make_retriever(cfg, not args.no_rerank, args.candidate_k, verbose=False)
    if args.offline:
        generator: Any = EchoGenerator()
        gen_mode = "offline-echo"
    else:
        generator = load_generator(cfg=cfg, verbose=False)
        gen_mode = f"{type(generator).__name__}:{getattr(generator, 'model', '')}"

    print(f"[gen] 检索={mode}  生成={gen_mode}  prompt={args.prompt_mode}", flush=True)
    print(f"[gen] 待生成 {len(items)} 条，预算 {args.budget} 字符 / 最多 {args.top_n} 片", flush=True)

    out_path = Path(
        args.out or f"data/generated/generation_{args.prompt_mode}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        pending = [q for q in items if q["id"] not in done_ids]
        print(f"[gen] 续跑模式：已有 {len(done_ids)} 条，待补 {len(pending)} 条", flush=True)
        items = pending
        if not items:
            print("[gen] 没有需要补的题，退出")
            return 0
    elif out_path.exists():
        out_path.unlink()

    closed_book = args.prompt_mode == "closed-book"
    mode_desc = "闭卷（不送上下文）" if closed_book else "检索+精排提供上下文"
    print(f"[gen] 检索={mode}  生成={gen_mode}  prompt={args.prompt_mode}  上下文={mode_desc}", flush=True)
    print(f"[gen] 待生成 {len(items)} 条，预算 {args.budget} 字符 / 最多 {args.top_n} 片", flush=True)

    started = time.time()
    n_refused = 0
    total_tokens = 0
    with out_path.open("a" if done_ids else "w", encoding="utf-8") as f:
        for n, item in enumerate(items, start=1):
            if closed_book:
                texts, titles, used = [], [], 0
                hits = []
            else:
                hits = retriever.retrieve(item["question"], top_k=max(args.candidate_k, args.top_n))
                texts, titles, used = build_contexts(hits, args.budget, args.top_n)
            res = generator.answer(item["question"], texts, titles, args.prompt_mode)
            n_refused += int(res.refused)
            total_tokens += int(res.usage.get("total_tokens", 0))
            record = {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "gold_doc": item.get("gold_doc"),
                "gold_answer": item.get("gold_answer"),
                "answer_keys": item.get("answer_keys") or [],
                "probe_terms": item.get("probe_terms") or [],
                "prompt_mode": args.prompt_mode,
                "generator": gen_mode,
                "contexts": texts,
                "context_titles": titles,
                "context_chars": used,
                "retrieved_docs": [h.chunk.doc_name for h in hits[: args.top_n]],
                **res.to_dict(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(items):
                print(
                    f"[gen] {n}/{len(items)}  拒答累计 {n_refused}  "
                    f"tokens {total_tokens}  累计 {time.time() - started:.0f}s",
                    flush=True,
                )

    print(f"[gen] 完成，输出 {out_path}；拒答 {n_refused}/{len(items)}，耗时 {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
