# -*- coding: utf-8 -*-
"""检索策略横向对比：同一套探针 query、同一批索引、同一个上下文预算，只换召回/排序方式。

背景（为什么要比这些）：
    纯向量检索的失败模式是「文档找对了、段落找错了」。围绕这一点试了三种改法，
    结果必须逐条量出来，不能凭直觉挑：
      1. vector          —— 基线
      2. hybrid_rrf      —— BM25 与向量各自召回后 RRF 融合
      3. bm25_rerank     —— 向量宽召回 Top-N，再用 BM25 在候选集内重排
      4. cross_encoder   —— 向量宽召回 Top-N，再用 bge-reranker-base 精排

用法：
    python scripts/compare_retrievers.py
    python scripts/compare_retrievers.py --index-tag clean_base --budget 1200 --candidate-k 20
    python scripts/compare_retrievers.py --no-cross-encoder     # 没下载重排模型时跳过第四项
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np

from medical_rag.config import load_config
from medical_rag.embedding import BGEEmbedder
from medical_rag.hybrid import BM25Store, HybridRetriever, tokenize
from medical_rag.index import FaissIndex

sys.path.insert(0, str(ROOT_DIR))
from scripts.eval_answer_recall import QUERIES, load_corpus, verify_tokens  # noqa: E402


class VectorRetriever:
    """纯向量召回。"""

    def __init__(self, index: FaissIndex, embedder: BGEEmbedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int):
        vec = np.asarray(self.embedder.encode_queries(query), dtype=np.float32).reshape(1, -1)
        return list(self.index.search(vec, top_k=top_k)[0])


class BM25RerankRetriever:
    """向量宽召回 + BM25 在候选集内重排。

    与 RRF 融合的区别：BM25 不再独立贡献候选，只在向量已经筛过的集合里调整顺序，
    因此它的全局词频偏好（被「基层医疗卫生机构」这类高频短语带偏）影响范围被限制住了。
    """

    def __init__(self, index: FaissIndex, embedder: BGEEmbedder, bm25: BM25Store, candidate_k: int = 20):
        self.index = index
        self.base = VectorRetriever(index, embedder)
        self.bm25 = bm25
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int):
        from medical_rag.index import SearchHit

        candidates = self.base.retrieve(query, top_k=max(self.candidate_k, top_k))
        if not candidates:
            return []
        scores = {idx: score for idx, score in self.bm25.retrieve(query, len(self.index))}
        scored = sorted(
            ((i, scores.get(self._pos_of(h), 0.0)) for i, h in enumerate(candidates)),
            key=lambda kv: -kv[1],
        )
        return [SearchHit(chunk=candidates[i].chunk, score=float(s), rank=r) for r, (i, s) in enumerate(scored[:top_k], 1)]

    def _pos_of(self, hit) -> int:
        return hit.chunk.chunk_index if hit.chunk.chunk_index is not None else 0


def build_strategies(index, embedder, cfg, index_dir: Path, tag: str, candidate_k: int, with_cross: bool):
    strategies = [("vector", VectorRetriever(index, embedder))]

    hybrid = HybridRetriever.from_index(
        index,
        embedder,
        cache_path=index_dir / f"bm25_{tag}.pkl",
        top_k=int(cfg.get("retrieval.top_k", 5)),
        rrf_k=int(cfg.get("retrieval.hybrid.rrf_k", 60)),
        candidate_multiplier=int(cfg.get("retrieval.hybrid.candidate_multiplier", 4)),
        verbose=False,
    )
    strategies.append(("hybrid_rrf", hybrid))

    bm25 = BM25Store.load(list(index.chunks), index_dir / f"bm25_{tag}.pkl")
    strategies.append(("bm25_rerank", BM25RerankRetriever(index, embedder, bm25, candidate_k=candidate_k)))

    if with_cross:
        from medical_rag.rerank import CrossEncoderReranker, RerankRetriever

        reranker = CrossEncoderReranker(
            model_name=cfg.get("rerank.model_name", "BAAI/bge-reranker-base"),
            local_model_dir=cfg.get("rerank.local_model_dir", None) or None,
            device=cfg.get("rerank.device", "cpu"),
            batch_size=int(cfg.get("rerank.batch_size", 8)),
            max_length=int(cfg.get("rerank.max_length", 512)),
            verbose=False,
        )
        strategies.append(
            (
                "cross_encoder",
                RerankRetriever(VectorRetriever(index, embedder), reranker, candidate_k=candidate_k, top_k=int(cfg.get("retrieval.top_k", 5))),
            )
        )
    return strategies


def evaluate(retriever, queries, top_n: int) -> dict:
    per_query = []
    for query, tokens in queries:
        if not tokens:
            continue
        hits = list(retriever.retrieve(query, top_k=top_n))
        ctx = "\n".join(h.chunk.text for h in hits)
        covered = [t for t in tokens if t in ctx]
        per_query.append(
            {
                "query": query,
                "coverage": len(covered) / len(tokens),
                "top1_doc": hits[0].chunk.doc_name if hits else "",
                "missing": [t for t in tokens if t not in covered],
                "retrieved_chars": len(ctx),
            }
        )
    return {
        "answer_recall": sum(1 for q in per_query if q["coverage"] == 1.0) / len(per_query),
        "token_coverage": float(np.mean([q["coverage"] for q in per_query])),
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检索策略横向对比")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"))
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--index-dir", default=str(ROOT_DIR / "data" / "index"))
    parser.add_argument("--index-tag", default="clean_base")
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--candidate-k", type=int, default=20, help="二阶段方案的第一阶段召回条数")
    parser.add_argument("--no-cross-encoder", action="store_true")
    parser.add_argument("--out", default=str(ROOT_DIR / "reports" / "retriever_compare.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else ROOT_DIR / cfg.require("paths.data_dir")
    index_dir = Path(args.index_dir)

    index = FaissIndex.load(index_dir / f"faiss_{args.index_tag}.index", index_dir / f"chunks_{args.index_tag}.json")
    embedder = BGEEmbedder.from_config(cfg, verbose=False)

    corpus = load_corpus(corpus_dir)
    queries, dropped = verify_tokens(corpus)
    mean = float(np.mean([len(c.text) for c in index.chunks]))
    top_n = max(1, round(args.budget / mean))
    print(f"[cmp] 索引 {args.index_tag}：{len(index)} 片，均长 {mean:.0f} 字符")
    print(f"[cmp] 预算 {args.budget} 字符 → Top-{top_n}；剔除的关键串：{dropped if dropped else '无'}")

    results = []
    for name, retriever in build_strategies(
        index, embedder, cfg, index_dir, args.index_tag, args.candidate_k, not args.no_cross_encoder
    ):
        m = evaluate(retriever, queries, top_n)
        print(f"[cmp] 【{name:<14}】AnswerRecall={m['answer_recall']:.3f}  Coverage={m['token_coverage']:.3f}")
        results.append(
            {
                "index_tag": args.index_tag,
                "strategy": name,
                "top_n": top_n,
                "budget_chars": args.budget,
                "candidate_k": args.candidate_k,
                "answer_recall": m["answer_recall"],
                "token_coverage": m["token_coverage"],
                "per_query": m["per_query"],
            }
        )

    base = results[0]
    print("\n[cmp] 逐条覆盖率（0.00 表示这条 query 在该策略下一个关键串都没召回）")
    header = f"{'query':<26}" + "".join(f"{r['strategy'][:12]:>14}" for r in results)
    print("      " + header)
    for i, q in enumerate(results[0]["per_query"]):
        row = f"      {q['query'][:24]:<26}" + "".join(f"{r['per_query'][i]['coverage']:>14.2f}" for r in results)
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[cmp] 结果已写入 {out}")
    print(f"[cmp] 基线 {base['strategy']} AnswerRecall={base['answer_recall']:.3f}；"
          f"最佳 {max(results, key=lambda r: (r['answer_recall'], r['token_coverage']))['strategy']} "
          f"AnswerRecall={max(r['answer_recall'] for r in results):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
