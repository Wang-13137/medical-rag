# -*- coding: utf-8 -*-
"""检索层：把 "文本 query" 到 "Top-K 切片" 的流程串起来。

本层只做向量召回基线，不含 BM25、不含重排、不含生成，
这些属于后续阶段；检索结果统一输出 SearchHit，带分数与溯源信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import Config, ROOT_DIR, load_config
from .embedding import BGEEmbedder
from .index import FaissIndex, SearchHit


class Retriever:
    """封装「文本 → query 向量 → FAISS 近邻」的检索入口。"""

    def __init__(self, index: FaissIndex, embedder: BGEEmbedder, top_k: int = 5, score_floor: float = 0.0):
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.score_floor = score_floor

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        index_path: Optional[Path | str] = None,
        meta_path: Optional[Path | str] = None,
        verbose: bool = True,
    ) -> "Retriever":
        """从配置加载索引并按需加载模型。

        先加载模型，是为了拿已有的 resolved_model_dir 与索引元数据里的模型名做一致性提示，
        避免用错 embedding 模型去检索别人建的索引。
        """
        index_file = cls._resolve(cfg, index_path, "paths.index_path")
        meta_file = cls._resolve(cfg, meta_path, "paths.meta_path")

        embedder = BGEEmbedder.from_config(cfg, verbose=verbose)
        index = FaissIndex.load(index_file, meta_file)

        built_model = (index.meta.get("model") or {}).get("model_name")
        if built_model and built_model != embedder.model_name:
            print(f"[retrieve] 提示：索引由 {built_model} 构建，当前加载的是 {embedder.model_name}")

        return cls(
            index=index,
            embedder=embedder,
            top_k=int(cfg.get("retrieval.top_k", 5)),
            score_floor=float(cfg.get("retrieval.score_floor", 0.0)),
        )

    @classmethod
    def from_config_paths(
        cls,
        config_path: Path | str | None = None,
        overrides: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> "Retriever":
        """命令行入口：从配置文件路径构造，覆盖项形如 {"paths.index_path": "..."}。"""
        cfg = load_config(config_path or (ROOT_DIR / "configs" / "config.yaml"), overrides)
        return cls.from_config(cfg, verbose=verbose)

    @staticmethod
    def _resolve(cfg: Config, override: Optional[Path | str], dotted_key: str) -> Path:
        raw = Path(override) if override else Path(cfg.require(dotted_key))
        return raw if raw.is_absolute() else (ROOT_DIR / raw)

    # ------------------------------------------------------------ 检索

    def embed_query(self, query: str) -> np.ndarray:
        """单条 query 向量，内部自动加 bge 指令前缀。"""
        return self.embedder.encode_queries([query])[0]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchHit]:
        """返回 Top-K 命中，已按分数从高到低排列。"""
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        k = int(top_k or self.top_k)
        vector = self.embed_query(query)
        return self.index.search_one(vector, top_k=k)

    def retrieve_batch(self, queries: Sequence[str], top_k: Optional[int] = None) -> List[List[SearchHit]]:
        """批量检索，复用一次前向推理。"""
        k = int(top_k or self.top_k)
        vectors = self.embedder.encode_queries(list(queries))
        return self.index.search(vectors, top_k=k)

    def format_hits(self, hits: Sequence[SearchHit], text_width: int = 200) -> str:
        """把命中结果格式化成人类可读的多行文本。"""
        lines: List[str] = []
        for hit in hits:
            flag = "" if hit.score >= self.score_floor else "  （分数偏低，可能不相关）"
            preview = hit.chunk.text.replace("\n", " ")[:text_width]
            lines.append(f"[{hit.rank}] score={hit.score:.4f}  {hit.chunk.citation()}{flag}")
            lines.append(f"    {preview}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "num_chunks": len(self.index),
            "dimension": self.index.dimension,
            "top_k": self.top_k,
            "model_name": self.embedder.model_name,
        }


def build_retriever(
    config_path: Path | str | None = None,
    overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Retriever:
    """命令行脚本的一次性入口。"""
    cfg = load_config(config_path or (ROOT_DIR / "configs" / "config.yaml"), overrides)
    return Retriever.from_config(cfg, verbose=verbose)
