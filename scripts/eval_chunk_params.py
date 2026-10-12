# -*- coding: utf-8 -*-
"""切片参数对比实验：同一批真实中文临床指南语料，不同 chunk_size / overlap 下的检索质量。

评测方式（可复现、无主观打分）：
    每个 query 预先标注一组「应当命中的来源文档」（gold docs，可多份，凡多份指南都能回答的
    允许多答案）。指标为文档级 Recall@1 / Recall@3 / MRR@5 —— 即 Top-K 结果里是否出现了
    来源正确的文档，而不是看文本像不像。这避免了「读起来不错但没有引用依据」的主观评估。

用法：
    python scripts/eval_chunk_params.py                      # 跑全部配置
    python scripts/eval_chunk_params.py --configs 400:80     # 只跑指定配置
    python scripts/eval_chunk_params.py --save               # 把各配置的索引也落盘（默认只放内存）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np

from medical_rag.chunking import chunk_document
from medical_rag.config import load_config
from medical_rag.embedding import BGEEmbedder
from medical_rag.index import FaissIndex
from medical_rag.ingest import load_documents

# （query, 允许命中的 gold 文档文件名集合）
# gold 的选择规则：该题的标准答案主要出自本语料中的哪一份指南；若多份都能回答则都列入。
QUERIES: list[tuple[str, set[str]]] = [
    ("2型糖尿病患者糖化血红蛋白的控制目标是多少", {"diabetes_2024.txt", "primary_diabetes_2025.txt"}),
    ("高血压的诊断标准和血压水平分级", {"hypertension_clinical_practice_2022.txt", "primary_hypertension_2025.txt"}),
    ("急性缺血性脑卒中静脉溶栓的时间窗是多久", {"stroke_ais_2023.txt"}),
    ("儿童发热如何处理，退热药物怎么用", {"pediatric_fever_2026.txt"}),
    ("幽门螺杆菌的根除治疗方案有哪些", {"hp_2022.txt", "hp_consensus6_2022.txt"}),
    ("痛风急性发作期的一线治疗药物是什么", {"hyperuricemia_gout_2019.txt", "hyperuricemia_gout_2024.txt"}),
    ("妊娠期急性阑尾炎如何诊断和处理", {"appendicitis_pregnancy_2026.txt"}),
    ("社区获得性肺炎的经验性抗感染治疗方案", {"cap_2016.txt"}),
    ("慢性心力衰竭的药物治疗方案包括哪些药物", {"heart_failure_2024_official.txt"}),
    ("支气管哮喘的长期控制药物如何选择", {"asthma_2024.txt"}),
    ("基层医疗卫生机构2型糖尿病的筛查和转诊建议", {"primary_diabetes_2025.txt", "diabetes_2024.txt"}),
]


def evaluate(index: FaissIndex, embedder: BGEEmbedder, top_k: int = 5) -> dict:
    """对全部 query 跑一遍检索，返回文档级指标与逐条明细。"""
    recalls: list[dict] = []
    for query, gold in QUERIES:
        vec = embedder.encode_queries(query)
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        hits = index.search(vec, top_k=top_k)[0]
        docs = [h.chunk.doc_name for h in hits]
        rank = 0
        for i, doc in enumerate(docs, start=1):
            if doc in gold:
                rank = i
                break
        recalls.append(
            {
                "query": query,
                "gold": sorted(gold),
                "top1_doc": docs[0] if docs else "",
                "docs@3": docs[:3],
                "rank": rank,
                "rr": 1.0 / rank if rank else 0.0,
            }
        )
    n = len(recalls)
    r1 = sum(1 for r in recalls if r["rank"] == 1) / n
    r3 = sum(1 for r in recalls if 0 < r["rank"] <= 3) / n
    mrr = sum(r["rr"] for r in recalls) / n
    return {"recall@1": r1, "recall@3": r3, "mrr@5": mrr, "details": recalls}


def main() -> int:
    parser = argparse.ArgumentParser(description="切片参数对比实验")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"))
    parser.add_argument("--data-dir", default=None, help="默认 data/raw 真实指南语料")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["300:60", "400:80", "600:120"],
        help="切片配置，格式 chunk_size:overlap，例如 300:60",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="同时把索引落盘到 data/index/")
    parser.add_argument("--out", default=str(ROOT_DIR / "reports" / "chunk_param_eval.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else ROOT_DIR / cfg.require("paths.data_dir")
    documents = load_documents(data_dir)
    print(f"[eval] 语料：{data_dir.name}，{len(documents)} 份文档")
    for doc in documents:
        full = "\n".join(line.text for line in doc.lines)
        print(f"       - {doc.doc_name}（{len(full)} 字符）")

    embedder = BGEEmbedder.from_config(cfg, verbose=False)
    max_len = int(cfg.get("embedding.max_seq_length", 512))

    results = []
    for spec in args.configs:
        size_str, _, overlap_str = spec.partition(":")
        size = int(size_str)
        overlap = int(overlap_str) if overlap_str else max(1, size // 5)
        print(f"\n[eval] ===== chunk_size={size}, overlap={overlap} =====")
        t0 = time.time()
        chunks = []
        for doc in documents:
            chunks.extend(
                chunk_document(
                    doc,
                    chunk_size=size,
                    chunk_overlap=overlap,
                    min_tail_chars=int(cfg.get("chunking.min_tail_chars", 30)),
                    heading_max_chars=int(cfg.get("chunking.heading_max_chars", 60)),
                    prepend_heading_path=bool(cfg.get("chunking.prepend_heading_path", True)),
                )
            )
        lengths = np.array([len(c.text) for c in chunks])
        # bge 会被截断到 max_seq_length；中文一个字约 0.6~1 个 token，这里给出粗略提示
        over_ratio = float((lengths > max_len).mean())
        print(f"[eval] 切片 {len(chunks)} 片，均值 {lengths.mean():.0f} 字符，最长 {lengths.max()}"
              f"，超过 {max_len} 字符的占比 {over_ratio:.1%}")

        t1 = time.time()
        embeddings = np.asarray(embedder.encode_documents([c.text for c in chunks]), dtype=np.float32)
        encode_sec = time.time() - t1

        index = FaissIndex.build(embeddings, chunks, meta={"chunking": {"chunk_size": size, "chunk_overlap": overlap}})
        metrics = evaluate(index, embedder, top_k=args.top_k)
        total_sec = time.time() - t0

        print(f"[eval] 文档级 Recall@1={metrics['recall@1']:.3f} "
              f"Recall@3={metrics['recall@3']:.3f} MRR@5={metrics['mrr@5']:.3f}")
        for d in metrics["details"]:
            flag = "OK " if d["rank"] else "MISS"
            print(f"       [{flag} rank={d['rank']}] {d['query'][:22]:<24} -> {d['top1_doc']}")
        print(f"[eval] 耗时：编码 {encode_sec:.1f}s，合计 {total_sec:.1f}s")

        results.append(
            {
                "chunk_size": size,
                "chunk_overlap": overlap,
                "num_chunks": len(chunks),
                "mean_chars": float(lengths.mean()),
                "max_chars": int(lengths.max()),
                "over_max_seq_ratio": over_ratio,
                "encode_sec": round(encode_sec, 1),
                "recall@1": metrics["recall@1"],
                "recall@3": metrics["recall@3"],
                "mrr@5": metrics["mrr@5"],
                "details": metrics["details"],
            }
        )

        if args.save:
            tag = f"cs{size}_ov{overlap}"
            idx, meta = index.save(ROOT_DIR / "data" / "index" / f"faiss_{tag}.index",
                                   ROOT_DIR / "data" / "index" / f"chunks_{tag}.json")
            print(f"[eval] 索引已保存：{idx.name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果已写入 {out}")

    best = max(results, key=lambda r: (r["recall@3"], r["mrr@5"], r["recall@1"]))
    print(f"[eval] 最佳配置：chunk_size={best['chunk_size']}, overlap={best['chunk_overlap']} "
          f"（Recall@3={best['recall@3']:.3f}, MRR@5={best['mrr@5']:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
