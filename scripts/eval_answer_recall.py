# -*- coding: utf-8 -*-
"""答案关键串召回实验（文档级指标已经饱和，需要更细的一层）。

背景：scripts/eval_chunk_params.py 里三档切片参数的「文档级 Recall@1」全是 1.000 —— 因为每份
指南主题单一，query 里带病名就能命中正确文档，这个指标失去了区分能力。这里换成更贴近生成效果的
指标：把答案里的关键实体（数值、药名、检查项）列出来，看定长上下文预算内能不能把这些实体全召回。

公平性处理：
    1. 关键串必须在对应指南正文中真实出现，脚本会先核验，未出现的自动剔除并打印；
    2. 三档切片的检索条数按「等上下文预算」换算（round(budget / 该配置平均片长)），避免长切片
       因为塞进更多文本而天然占便宜。

用法：
    python scripts/eval_answer_recall.py                 # 默认预算 1200 字符，读 data/index 下三份索引
    python scripts/eval_answer_recall.py --budget 900
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
from medical_rag.index import FaissIndex

RAW_DIR = ROOT_DIR / "data" / "raw"

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

CONFIGS = [(300, 60), (400, 80), (600, 120)]


def build_corpus_index() -> dict[str, str]:
    """把语料按文件名读进来，用于核验关键串是否真实存在。"""
    return {p.name: p.read_text(encoding="utf-8", errors="ignore") for p in sorted(RAW_DIR.glob("*.txt"))}


def verify_tokens(corpus: dict[str, str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """剔除非语料中出现过的关键串，保证指标不会因标注错误而虚低。"""
    all_text = "\n".join(corpus.values())
    kept, dropped = [], []
    for query, tokens in QUERIES:
        good = [t for t in tokens if t in all_text]
        dropped.extend(t for t in tokens if t not in all_text)
        kept.append((query, good))
    return kept, dropped


def run_one(index: FaissIndex, embedder: BGEEmbedder, queries, top_n: int) -> dict:
    full_hits, coverage = 0, []
    per_query = []
    for query, tokens in queries:
        if not tokens:
            continue
        vec = np.asarray(embedder.encode_queries(query), dtype=np.float32).reshape(1, -1)
        hits = index.search(vec, top_k=top_n)[0]
        ctx = "\n".join(h.chunk.text for h in hits)
        covered = [t for t in tokens if t in ctx]
        rate = len(covered) / len(tokens)
        coverage.append(rate)
        full_hits += 1 if rate == 1.0 else 0
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
        "answer_recall": full_hits / n,
        "token_coverage": float(np.mean(coverage)),
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="答案关键串召回实验")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"))
    parser.add_argument("--budget", type=int, default=1200, help="上下文预算（字符）")
    parser.add_argument("--top-n", type=int, default=None, help="强制指定检索条数，覆盖预算换算")
    parser.add_argument("--out", default=str(ROOT_DIR / "reports" / "answer_recall_eval.json"))
    args = parser.parse_args()

    embedder = BGEEmbedder.from_config(load_config(args.config), verbose=False)
    corpus = build_corpus_index()
    queries, dropped = verify_tokens(corpus)
    print(f"[eval] 语料 {len(corpus)} 份；因正文中查不到而被剔除的关键串：{dropped if dropped else '无'}")

    summary_path = ROOT_DIR / "reports" / "chunk_param_eval.json"
    means = {}
    if summary_path.exists():
        for r in json.loads(summary_path.read_text(encoding="utf-8")):
            means[(r["chunk_size"], r["chunk_overlap"])] = r["mean_chars"]

    results = []
    for size, overlap in CONFIGS:
        tag = f"cs{size}_ov{overlap}"
        idx_path = ROOT_DIR / "data" / "index" / f"faiss_{tag}.index"
        meta_path = ROOT_DIR / "data" / "index" / f"chunks_{tag}.json"
        if not idx_path.exists():
            print(f"[eval] 缺少索引 {idx_path.name}，跳过（先跑 eval_chunk_params.py --save）")
            continue
        index = FaissIndex.load(idx_path, meta_path)
        mean = means.get((size, overlap), 0.0)
        top_n = args.top_n if args.top_n else max(1, round(args.budget / mean)) if mean else 3
        budget_used = top_n * mean
        print(f"\n[eval] ===== {tag}：{len(index)} 片，均长 {mean:.0f} 字符，"
              f"预算 {args.budget} 字符 → Top-{top_n}（约 {budget_used:.0f} 字符）=====")
        m = run_one(index, embedder, queries, top_n)
        print(f"[eval] 关键串全中率 AnswerRecall={m['answer_recall']:.3f}  平均覆盖率={m['token_coverage']:.3f}")
        for q in m["per_query"]:
            flag = "OK  " if q["coverage"] == 1.0 else "MISS"
            miss = [t for t in q["tokens"] if t not in q["covered"]]
            print(f"       [{flag} {q['coverage']:.2f}] {q['query'][:20]:<22} 命中来源 {q['top1_doc'][:34]:<36}"
                  f" 缺：{miss if miss else '无'}")
        results.append(
            {
                "chunk_size": size,
                "chunk_overlap": overlap,
                "budget_chars": args.budget,
                "top_n": top_n,
                "num_chunks": len(index),
                "mean_chars": mean,
                "answer_recall": m["answer_recall"],
                "token_coverage": m["token_coverage"],
                "per_query": m["per_query"],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果已写入 {out}")
    if results:
        best = max(results, key=lambda r: (r["answer_recall"], r["token_coverage"]))
        print(f"[eval] 最佳：chunk_size={best['chunk_size']} overlap={best['chunk_overlap']} "
              f"（AnswerRecall={best['answer_recall']:.3f}, Coverage={best['token_coverage']:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
