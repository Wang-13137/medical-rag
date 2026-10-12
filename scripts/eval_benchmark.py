"""在 benchmark v1 上评测当前检索管线，产出 baseline 指标。

评测口径（与 W2 保持一致，便于跨阶段比较）：
  - 文档级：gold_doc 是否出现在 Top-k
  - 答案级：在等上下文预算下，标准答案的关键串召回比例
  - 拒答：应拒答题与可答题的检索置信度分布，并给出阈值权衡曲线

用法:
    python scripts/eval_benchmark.py                       # 默认策略（配置决定：向量 + 精排）
    python scripts/eval_benchmark.py --no-rerank           # 纯向量对照
    python scripts/eval_benchmark.py --budget 1200 --top-n 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_rag.config import load_config
from medical_rag.embedding import BGEEmbedder
from medical_rag.index import FaissIndex
from medical_rag.rerank import CrossEncoderReranker, RerankRetriever
from medical_rag.retrieve import Retriever


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("＜", "<").replace("＞", ">")
    folded = folded.replace("－", "-").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", "", folded)


def text_of(hit) -> str:
    return getattr(hit, "text", None) or hit.chunk.text


def doc_of(hit) -> str:
    return getattr(hit, "doc_name", None) or hit.chunk.doc_name


def make_retriever(cfg, use_rerank: bool, candidate_k: int, verbose: bool):
    """构造检索器：第一阶段固定纯向量，第二阶段可选 cross-encoder 精排。"""
    index = FaissIndex.load(cfg.get("paths.index_path"), cfg.get("paths.meta_path"))
    embedder = BGEEmbedder.from_config(cfg, verbose=verbose)
    first = Retriever(index=index, embedder=embedder, top_k=int(cfg.get("retrieval.top_k", 5)))
    if not use_rerank:
        return "纯向量", first
    reranker = CrossEncoderReranker(
        model_name=cfg.get("rerank.model_name", "BAAI/bge-reranker-base"),
        local_model_dir=cfg.get("rerank.local_model_dir", None) or None,
        device=cfg.get("rerank.device", "cpu"),
        batch_size=int(cfg.get("rerank.batch_size", 8)),
        max_length=int(cfg.get("rerank.max_length", 512)),
        verbose=verbose,
    )
    two = RerankRetriever(first, reranker, candidate_k=candidate_k, top_k=candidate_k)
    return "向量召回 + cross-encoder 精排", two


def run_benchmark(retriever, items, top_n: int, budget: int, wide_k: int) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = []
    for item in items:
        question = item["question"]
        started = time.time()
        hits: Sequence = retriever.retrieve(question, top_k=wide_k)
        elapsed = time.time() - started

        docs_at_rank = [doc_of(h) for h in hits]
        gold_doc = item.get("gold_doc")

        # 文档级指标
        first_gold = next((i + 1 for i, d in enumerate(docs_at_rank) if d == gold_doc), None)
        mrr = 1.0 / first_gold if first_gold else 0.0

        # 等上下文预算：按顺序拼接切片文本，超出预算即停
        keys = item.get("answer_keys") or []
        context, used = [], 0
        for h in hits:
            t = text_of(h)
            if used + len(t) > budget and context:
                break
            context.append(t)
            used += len(t)
        ctx_norm = normalize("\n".join(context))

        hit_keys = [k for k in keys if normalize(k) in ctx_norm]
        recall = len(hit_keys) / len(keys) if keys else 0.0

        top_score = float(hits[0].score) if hits else 0.0
        per_query.append(
            {
                "id": item["id"],
                "type": item["type"],
                "gold_doc": gold_doc,
                "first_gold_rank": first_gold,
                "mrr": round(mrr, 4),
                "answer_recall": round(recall, 4),
                "full_coverage": 1.0 if recall == 1.0 else 0.0,
                "hit_keys": len(hit_keys),
                "total_keys": len(keys),
                "missing_keys": [k for k in keys if k not in hit_keys],
                "top_score": round(top_score, 4),
                "top1_doc": docs_at_rank[0] if docs_at_rank else "",
                "context_chars": used,
                "elapsed_sec": round(elapsed, 3),
            }
        )
    return {"per_query": per_query}


def summarize(per_query: List[Dict[str, Any]], top_k_list, prefix="") -> Dict[str, Any]:
    n = len(per_query)
    if n == 0:
        return {}
    out: Dict[str, Any] = {"n": n}
    for k in top_k_list:
        out[f"doc_hit@{k}"] = round(
            sum(1 for q in per_query if q["first_gold_rank"] and q["first_gold_rank"] <= k) / n, 4
        )
    out["mrr"] = round(sum(q["mrr"] for q in per_query) / n, 4)
    out["answer_recall"] = round(sum(q["answer_recall"] for q in per_query) / n, 4)
    out["full_coverage"] = round(sum(q["full_coverage"] for q in per_query) / n, 4)

    by_type: Dict[str, List] = defaultdict(list)
    by_doc: Dict[str, List] = defaultdict(list)
    for q in per_query:
        by_type[q["type"]].append(q)
        if q["gold_doc"]:
            by_doc[q["gold_doc"]].append(q)

    def _agg(rows, label):
        return {
            "n": len(rows),
            "doc_hit@1": round(sum(1 for q in rows if q["first_gold_rank"] == 1) / len(rows), 4),
            "doc_hit@3": round(
                sum(1 for q in rows if q["first_gold_rank"] and q["first_gold_rank"] <= 3) / len(rows), 4
            ),
            "answer_recall": round(sum(q["answer_recall"] for q in rows) / len(rows), 4),
            "full_coverage": round(sum(q["full_coverage"] for q in rows) / len(rows), 4),
        }

    out["by_type"] = {t: _agg(rows, t) for t, rows in sorted(by_type.items())}
    out["by_doc"] = {d: _agg(rows, d) for d, rows in sorted(by_doc.items())}
    return out


def refusal_tradeoff(answerable, unanswerable) -> Dict[str, Any]:
    """用 Top-1 检索置信度做拒答判断，扫描阈值看能否把两类分开。

    生成模块还没做，这里先用检索侧置信度作为「系统是否会硬答」的代理指标：
    置信度低于阈值就应当拒答。
    """
    rows = []
    cand = sorted({q["top_score"] for q in answerable + unanswerable})
    grid = [round(x, 4) for x in cand] + [round(x + 0.01, 4) for x in cand]
    best = None
    for t in sorted(set(grid)):
        correct_refuse = sum(1 for q in unanswerable if q["top_score"] < t) / len(unanswerable)
        false_refuse = sum(1 for q in answerable if q["top_score"] < t) / len(answerable)
        youden = correct_refuse - false_refuse  # 拒答正确的收益减去误拒可答题的代价
        rows.append(
            {
                "threshold": t,
                "refuse_correct_rate": round(correct_refuse, 4),
                "answer_false_refuse_rate": round(false_refuse, 4),
                "youden_j": round(youden, 4),
            }
        )
        if best is None or youden > best["youden_j"]:
            best = rows[-1]
    return {"best": best, "curve_sample": rows[:: max(1, len(rows) // 25)]}


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark v1 评测")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--benchmark", default="data/eval/benchmark_v1.json")
    ap.add_argument("--no-rerank", action="store_true", help="只用向量召回作为对照")
    ap.add_argument("--budget", type=int, default=2400, help="上下文字符预算")
    ap.add_argument("--top-n", type=int, default=8, help="进入上下文的最大切片数")
    ap.add_argument("--candidate-k", type=int, default=40, help="精排候选数")
    ap.add_argument("--out", default="reports/benchmark_eval_v1.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = json.load(open(args.benchmark, encoding="utf-8"))
    items = bench["items"]
    answerable = [q for q in items if q.get("type") != "unanswerable"]
    unanswerable = [q for q in items if q.get("type") == "unanswerable"]

    mode, retriever = make_retriever(cfg, not args.no_rerank, args.candidate_k, verbose=False)
    print(f"[eval] 检索模式：{mode}")
    print(f"[eval] 可回答题 {len(answerable)} 条，应拒答题 {len(unanswerable)} 条")
    print(f"[eval] 上下文预算 {args.budget} 字符 / 最多 {args.top_n} 片")

    started = time.time()
    res_a = run_benchmark(retriever, answerable, args.top_n, args.budget, args.candidate_k)
    res_u = run_benchmark(retriever, unanswerable, args.top_n, args.budget, args.candidate_k)
    total_sec = time.time() - started

    # 拒答题没有关键串，借用 answer_recall 字段会失真，这里只保留分布信息
    overall = summarize(res_a["per_query"], top_k_list=[1, 3, 5])
    tradeoff = refusal_tradeoff(res_a["per_query"], res_u["per_query"])

    report = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "budget_chars": args.budget,
        "top_n": args.top_n,
        "candidate_k": args.candidate_k,
        "index": str(cfg.get("paths.index_path")),
        "total_seconds": round(total_sec, 1),
        "overall": overall,
        "refusal_tradeoff": tradeoff,
        "unanswerable_top1_stats": {
            "scores": [q["top_score"] for q in res_u["per_query"]],
            "mean": round(sum(q["top_score"] for q in res_u["per_query"]) / len(res_u["per_query"]), 4),
        },
        "answerable_per_query": res_a["per_query"],
        "unanswerable_per_query": res_u["per_query"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval] doc_hit@1 = {overall['doc_hit@1']:.3f}  doc_hit@3 = {overall['doc_hit@3']:.3f}  "
          f"doc_hit@5 = {overall['doc_hit@5']:.3f}  MRR = {overall['mrr']:.3f}")
    print(f"[eval] AnswerRecall = {overall['answer_recall']:.3f}  全中率 = {overall['full_coverage']:.3f}")
    print("[eval] 分题型：")
    for t, m in overall["by_type"].items():
        print(f"    {t:<15} n={m['n']:<3} hit@1={m['doc_hit@1']:.3f} recall={m['answer_recall']:.3f} 全中={m['full_coverage']:.3f}")
    if tradeoff["best"]:
        b = tradeoff["best"]
        print(f"[eval] 拒答阈值建议 > {b['threshold']}：拒答正确率 {b['refuse_correct_rate']:.3f}，"
              f"可答题误拒率 {b['answer_false_refuse_rate']:.3f}")
    print(f"[eval] 耗时 {total_sec:.1f}s，结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
