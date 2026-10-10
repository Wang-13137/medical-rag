# -*- coding: utf-8 -*-
"""FAISS 索引：构建、持久化、加载与近邻搜索。

使用 IndexFlatIP，配合已做 L2 归一化的向量，其内积等价于余弦相似度。
索引文件与切片元数据分开存储但同回合写入，元数据里的 chunk_id 顺序
必须与写入索引的向量行号严格一致，否则召回结果会对不上来源。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from .chunking import Chunk


@dataclass
class SearchHit:
    """一次检索命中：切片 + 分数 + 排名。"""

    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "score": self.score, "citation": self.chunk.citation(), "text": self.chunk.text}


class FaissIndex:
    """IndexFlatIP 的薄封装，负责向量存储与元数据对齐。"""

    def __init__(self, dimension: int, index: Optional[faiss.Index] = None):
        self.dimension = dimension
        self.index = index if index is not None else faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []
        self.meta: Dict[str, Any] = {}

    # ------------------------------------------------------------ 构建

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: Sequence[Chunk], meta: Optional[Dict[str, Any]] = None) -> "FaissIndex":
        """用文档向量直接构建索引，要求 embeddings 行数与 chunks 数量相等。"""
        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"切片数量({len(chunks)}) 与向量行数({vectors.shape[0]}) 不一致")
        if vectors.ndim != 2:
            raise ValueError(f"embeddings 必须是二维矩阵，当前形状 {vectors.shape}")

        instance = cls(dimension=int(vectors.shape[1]))
        instance.index.add(vectors)
        instance.chunks = list(chunks)
        instance.meta = dict(meta or {})
        instance.meta.setdefault("num_chunks", len(chunks))
        instance.meta.setdefault("index_type", "IndexFlatIP")
        return instance

    def add(self, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> None:
        """增量写入；向量维度必须与索引一致。"""
        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度({vectors.shape[1]}) 与索引维度({self.dimension}) 不一致")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.meta["num_chunks"] = len(self.chunks)

    # ------------------------------------------------------------ 搜索

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> List[List[SearchHit]]:
        """批量检索；返回每个 query 的 SearchHit 列表。"""
        vectors = np.ascontiguousarray(np.asarray(query_vectors, dtype=np.float32))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"query 向量维度({vectors.shape[1]}) 与索引维度({self.dimension}) 不一致")

        top_k = max(1, min(top_k, len(self.chunks)))
        scores, indices = self.index.search(vectors, top_k)

        results: List[List[SearchHit]] = []
        for row in range(vectors.shape[0]):
            hits: List[SearchHit] = []
            for rank, (idx, score) in enumerate(zip(indices[row], scores[row]), start=1):
                if idx < 0:  # FAISS 用 -1 表示不足 top_k
                    continue
                hits.append(SearchHit(chunk=self.chunks[int(idx)], score=float(score), rank=rank))
            results.append(hits)
        return results

    def search_one(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchHit]:
        """单 query 检索的便捷入口。"""
        return self.search(query_vector, top_k=top_k)[0]

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------ 持久化

    def save(self, index_path: Path | str, meta_path: Optional[Path | str] = None) -> Tuple[Path, Path]:
        """写索引二进制与元数据 JSON（含全部切片的溯源字段）。"""
        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))

        meta_file = Path(meta_path) if meta_path else index_file.with_name(f"{index_file.stem}_meta.json")
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimension": self.dimension,
            "num_chunks": len(self.chunks),
            "index_file": index_file.name,
            "meta": self.meta,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return index_file, meta_file

    @classmethod
    def load(cls, index_path: Path | str, meta_path: Optional[Path | str] = None) -> "FaissIndex":
        """从磁盘恢复索引与元数据。"""
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在：{index_file}，请先运行 scripts/build_index.py")
        meta_file = Path(meta_path) if meta_path else index_file.with_name(f"{index_file.stem}_meta.json")
        if not meta_file.exists():
            raise FileNotFoundError(f"元数据文件不存在：{meta_file}")

        index = faiss.read_index(str(index_file))
        payload = json.loads(meta_file.read_text(encoding="utf-8"))

        instance = cls(dimension=int(payload.get("dimension", index.d)), index=index)
        instance.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        instance.meta = payload.get("meta", {})
        if len(instance.chunks) != index.ntotal:
            raise ValueError(
                f"元数据切片数({len(instance.chunks)}) 与索引向量数({index.ntotal}) 不一致，索引已损坏"
            )
        return instance
