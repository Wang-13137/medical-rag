# -*- coding: utf-8 -*-
"""命令行检索测试：加载已构建的索引，输出 Top-K 片段。

用法（在仓库根目录执行）：
    python scripts/query.py --q "2型糖尿病血糖控制目标"
    python scripts/query.py --q "他汀类药物的禁忌" --smoke --top-k 3
    python scripts/query.py --index-path data/index/main.index --meta-path data/index/chunks_main.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from typing import Any, Dict, Optional, Tuple

from medical_rag.config import load_config
from medical_rag.embedding import BGEEmbedder
from medical_rag.hybrid import HybridRetriever
from medical_rag.index import FaissIndex
from medical_rag.rerank import CrossEncoderReranker, RerankRetriever
from medical_rag.retrieve import Retriever


def use_hybrid(args: argparse.Namespace) -> bool:
    """命令行 --no-hybrid 优先级最高，其次看配置里的 retrieval.use_hybrid。"""
    if args.no_hybrid:
        return False
    cfg = load_config(args.config)
    return bool(cfg.get("retrieval.use_hybrid", False))


def build_retriever(args: argparse.Namespace, overrides: Optional[Dict[str, Any]] = None) -> Tuple[str, Any]:
    """按配置把「第一阶段召回 + 第二阶段精排」串起来，返回 (模式名, 检索器)。

    优先级：命令行 --no-hybrid / --no-rerank > 配置 retrieval.use_hybrid / rerank.enabled。
    """
    cfg = load_config(args.config, overrides)
    index_path = Retriever._resolve(cfg, None, "paths.index_path")
    meta_path = Retriever._resolve(cfg, None, "paths.meta_path")
    verbose = not args.quiet

    index = FaissIndex.load(index_path, meta_path)
    embedder = BGEEmbedder.from_config(cfg, verbose=verbose)

    if not args.no_hybrid and bool(cfg.get("retrieval.use_hybrid", False)):
        cache = cfg.get("retrieval.hybrid.cache_path", None)
        cache_path = Retriever._resolve(cfg, cache, "retrieval.hybrid.cache_path") if cache else None
        first_stage: Any = HybridRetriever.from_index(
            index,
            embedder,
            cache_path=cache_path,
            top_k=int(cfg.get("retrieval.top_k", 5)),
            rrf_k=int(cfg.get("retrieval.hybrid.rrf_k", 60)),
            candidate_multiplier=int(cfg.get("retrieval.hybrid.candidate_multiplier", 4)),
            verbose=verbose,
        )
        mode = "BM25+向量 RRF 混合"
    else:
        first_stage = Retriever(index=index, embedder=embedder, top_k=int(cfg.get("retrieval.top_k", 5)))
        mode = "纯向量"

    if args.no_rerank or not bool(cfg.get("rerank.enabled", False)):
        return mode, first_stage

    reranker = CrossEncoderReranker(
        model_name=cfg.get("rerank.model_name", "BAAI/bge-reranker-base"),
        local_model_dir=cfg.get("rerank.local_model_dir", None) or None,
        device=cfg.get("rerank.device", "cpu"),
        batch_size=int(cfg.get("rerank.batch_size", 8)),
        max_length=int(cfg.get("rerank.max_length", 512)),
        verbose=verbose,
    )
    # 精排交给 RerankRetriever 自己按 rerank.candidate_k 放大第一阶段召回，这里不再额外处理
    reranked = RerankRetriever(
        first_stage,
        reranker,
        candidate_k=int(cfg.get("rerank.candidate_k", 20)),
        top_k=int(cfg.get("retrieval.top_k", 5)),
    )
    return f"{mode} + cross-encoder 精排", reranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="向量检索基线查询")
    parser.add_argument("--q", "--query", dest="query", required=True, help="查询文本")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"), help="配置文件路径")
    parser.add_argument("--index-path", default=None, help="覆盖配置里的 paths.index_path")
    parser.add_argument("--meta-path", default=None, help="覆盖配置里的 paths.meta_path")
    parser.add_argument("--smoke", action="store_true", help="使用配置里的 smoke 索引路径（默认即此路径）")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数，默认取配置 retrieval.top_k")
    parser.add_argument("--full", action="store_true", help="打印完整文本而不是截断预览")
    parser.add_argument("--quiet", action="store_true", help="不打印模型加载信息")
    parser.add_argument("--no-hybrid", action="store_true", help="只用向量召回，不启用 BM25 混合（便于对比）")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 cross-encoder 精排")
    parser.add_argument("--explain", action="store_true", help="额外打印每条结果在两路中的原始排名")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    overrides = {}
    if args.index_path:
        overrides["paths.index_path"] = args.index_path
    if args.meta_path:
        overrides["paths.meta_path"] = args.meta_path

    try:
        mode, retriever = build_retriever(args, overrides)
    except (FileNotFoundError, ImportError) as exc:
        print(f"[query] {exc}")
        print("[query] 请先运行 scripts/build_index.py 构建索引")
        return 1

    hits = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"[query] query：{args.query}")
    print(f"[query] 模式：{mode}")
    print(f"[query] 命中 {len(hits)} 条（索引共 {len(retriever.index)} 条）")
    print("-" * 78)
    width = 10_000 if args.full else 200
    print(retriever.format_hits(hits, text_width=width))

    if not args.explain:
        return 0
    print("-" * 78)
    if hasattr(retriever, "bm25"):  # 混合模式：打印两路原始排名
        print("[explain] 两路原始排名（None = 该路未取到这一条）")
        for row in retriever.explain(args.query, top_k=args.top_k):
            print(f"    [#{row['rank']}] rrf={row['rrf_score']:.5f} "
                  f"vector_rank={row['vector_rank']} bm25_rank={row['bm25_rank']}  {row['doc_name']}")
    elif hasattr(retriever, "reranker"):  # 精排模式：打印精排前后的名次变化
        print("[explain] cross-encoder 精排前后的名次变化")
        for row in retriever.explain(args.query, top_k=args.top_k):
            print(f"    [#{row['rank']}] rerank={row['rerank_score']:+.3f} "
                  f"召回名次={row['first_stage_rank']}  {row['doc_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
