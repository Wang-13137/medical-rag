# -*- coding: utf-8 -*-
"""答案关键串召回实验（文档级指标已经饱和，需要更细的一层）。

背景：scripts/eval_chunk_params.py 里三档切片参数的「文档级 Recall@1」全是 1.000 —— 因为每份
指南主题单一，query 里带病名就能命中正确文档，这个指标失去了区分能力。这里换成更贴近生成效果的
指标：把答案里的关键实体（数值、药名、检查项）列出来，看定长上下文预算内能不能把这些实体全召回。

公平性处理：
    1. 关键串必须在语料正文中真实出现，脚本会先核验，未出现的自动剔除并打印；
    2. 检索条数按「等上下文预算」换算（round(budget / 平均片长)），避免长切片因为塞进更多
       文本而天然占便宜；
    3. 纯向量与 BM25+向量混合共用同一个索引、同一批关键串、同一预算，唯一变量是召回方式。

用法：
    python scripts/eval_answer_recall.py                          # 纯向量 vs 混合
    python scripts/eval_answer_recall.py --budget 900             # 换上下文预算
    python scripts/eval_answer_recall.py --index-tags clean_base  # 指定索引
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
from medical_rag.hybrid import HybridRetriever
from medical_rag.index import FaissIndex

# (query, 候选关键串)。关键串先核验是否出现在语料里，未出现的不计入。
QUERIES: list[tuple[str, list[str]]] = [
    ("2型糖尿病患者糖化血红蛋白的控制目标是多少", ["HbA1c", "7%", "糖化血红蛋白"]),
    ("高血压的诊断标准和血压水平分级", ["140", "90", "诊室血压"]),
    ("急性缺血性脑卒中静脉溶栓的时间窗是多久", ["4.5", "静脉溶栓", "阿替普酶"]),
    ("儿童发热如何处理，退热药物怎么用", ["布洛芬", "对乙酰氨基酚"]),
    ("幽门螺杆菌的根除治疗方案有哪些", ["四联", "铋剂", "克拉霉素"]),
    ("痛风急性发作期的一线治疗药物是什么", ["秋水仙碱", "NSAID"]),
    ("社区获得性肺炎的经验性抗感染治疗方案", ["喹诺酮", "阿莫西林", "克拉维酸"]),
    ("慢性心力衰竭的药物治疗方案包括哪些药物", ["沙库巴曲", "达格列净", "ARNI"]),
    ("支气管哮喘的长期控制药物如何选择", ["吸入性糖皮质激素", "ICS", "布地奈德"]),
    ("基层医疗卫生机构2型糖尿病的筛查和转诊建议", ["空腹血糖", "OGTT", "糖化血红蛋白"]),
]


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8", errors="ignore") for p in sorted(corpus_dir.glob("*.txt"))}


def verify_tokens(corpus: dict[str, str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """剔除非语料中出现过的关键串，保证指标不会因标注错误而虚低。"""
    all_text = "\n".join(corpus.values())
    kept, dropped = [], []
    for query, tokens in QUERIES:
        good = [t for t in tokens if t in all_text]
        dropped.extend(t for t in tokens if t not in all_text)
        kept.append((query, good))
    return kept, dropped


class VectorRetriever:
    """最小的向量召回适配器：把 FaissIndex + embedder 包成 retrieve(query, top_k)。"""

    def __init__(self, index: FaissIndex, embedder: BGEEmbedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int):
        vec = np.asarray(self.embedder.encode_queries(query), dtype=np.float32).reshape(1, -1)
        return list(self.index.search(vec, top_k=top_k)[0])


def run_one(retriever, queries, top_n: int) -> dict:
    """跑一遍全部 query，返回关键串全中率、平均覆盖率与逐条明细。"""
    per_query = []
    for query, tokens in queries:
        if not tokens:
            continue
        hits = list(retriever.retrieve(query, top_k=top_n))
        ctx = "\n".join(h.chunk.text for h in hits)
        covered = [t for t in tokens if t in ctx]
        rate = len(covered) / len(tokens)
        per_query.append(
            {
                "query": query,
                "tokens": tokens,
                "covered": covered,
                "coverage": rate,
                "retrieved_chars": len(ctx),
                "top1_doc": hits[0].chunk.doc_name if hits else "",
            }
        )
    n = len(per_query)
    return {
        "answer_recall": sum(1 for q in per_query if q["coverage"] == 1.0) / n,
        "token_coverage": float(np.mean([q["coverage"] for q in per_query])) if per_query else 0.0,
        "per_query": per_query,
    }


def build_variants(index: FaissIndex, embedder: BGEEmbedder, cfg, index_dir: Path, tag: str, with_hybrid: bool):
    variants = [("纯向量", VectorRetriever(index, embedder))]
    if with_hybrid:
        hybrid = HybridRetriever.from_index(
            index,
            embedder,
            cache_path=index_dir / f"bm25_{tag}.pkl",
            top_k=int(cfg.get("retrieval.top_k", 5)),
            rrf_k=int(cfg.get("retrieval.hybrid.rrf_k", 60)),
            candidate_multiplier=int(cfg.get("retrieval.hybrid.candidate_multiplier", 4)),
            verbose=False,
        )
        variants.append(("BM25+向量RRF", hybrid))
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="答案关键串召回实验")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"))
    parser.add_argument("--corpus-dir", default=None, help="核验关键串用的语料目录，默认取 paths.data_dir")
    parser.add_argument("--index-dir", default=str(ROOT_DIR / "data" / "index"), help="索引所在目录")
    parser.add_argument("--index-tags", nargs="*", default=["clean_base"], help="要评估的索引名（faiss_ 之后的部分）")
    parser.add_argument("--budget", type=int, default=1200, help="上下文预算（字符）")
    parser.add_argument("--top-n", type=int, default=None, help="强制指定检索条数，覆盖预算换算")
    parser.add_argument("--no-hybrid", action="store_true", help="只跑纯向量")
    parser.add_argument("--out", default=str(ROOT_DIR / "reports" / "answer_recall_eval.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else ROOT_DIR / cfg.require("paths.data_dir")
    index_dir = Path(args.index_dir)

    embedder = BGEEmbedder.from_config(cfg, verbose=False)
    corpus = load_corpus(corpus_dir)
    queries, dropped = verify_tokens(corpus)
    print(f"[eval] 语料目录 {corpus_dir.name}（{len(corpus)} 份）；"
          f"因正文中查不到而被剔除的关键串：{dropped if dropped else '无'}")

    results = []
    for tag in args.index_tags:
        idx_path = index_dir / f"faiss_{tag}.index"
        meta_path = index_dir / f"chunks_{tag}.json"
        if not idx_path.exists():
            print(f"[eval] 缺少索引 {idx_path.name}，跳过（先跑 build_index.py --index-path {idx_path}）")
            continue
        index = FaissIndex.load(idx_path, meta_path)
        mean = float(np.mean([len(c.text) for c in index.chunks]))
        top_n = args.top_n or max(1, round(args.budget / mean))

        print(f"\n[eval] ===== 索引 {tag}：{len(index)} 片，均长 {mean:.0f} 字符，"
              f"预算 {args.budget} → Top-{top_n}（约 {top_n * mean:.0f} 字符）=====")
        variants = build_variants(index, embedder, cfg, index_dir, tag, not args.no_hybrid)
        for name, retriever in variants:
            m = run_one(retriever, queries, top_n)
            print(f"[eval] 【{name}】关键串全中率 AnswerRecall={m['answer_recall']:.3f}  "
                  f"平均覆盖率={m['token_coverage']:.3f}")
            results.append(
                {
                    "index_tag": tag,
                    "retriever": name,
                    "top_n": top_n,
                    "budget_chars": args.budget,
                    "num_chunks": len(index),
                    "mean_chars": mean,
                    "answer_recall": m["answer_recall"],
                    "token_coverage": m["token_coverage"],
                    "per_query": m["per_query"],
                }
            )

        # 最后一版的逐条明细，便于看是哪几条拉低了分数
        for q in results[-1]["per_query"]:
            miss = [t for t in q["tokens"] if t not in q["covered"]]
            flag = "OK  " if q["coverage"] == 1.0 else "MISS"
            print(f"       [{flag} {q['coverage']:.2f}] {q['query'][:20]:<22} "
                  f"{q['top1_doc']:<34} 缺：{miss if miss else '无'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果已写入 {out}")
    if results:
        best = max(results, key=lambda r: (r["answer_recall"], r["token_coverage"]))
        print(f"[eval] 最佳：{best['retriever']} @ {best['index_tag']} "
              f"（AnswerRecall={best['answer_recall']:.3f}, Coverage={best['token_coverage']:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
