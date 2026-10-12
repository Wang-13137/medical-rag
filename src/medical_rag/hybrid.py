# -*- coding: utf-8 -*-
"""混合检索：BM25（稀疏、字面精确匹配）+ 向量（稠密、语义召回）两路融合。

为什么需要这一路：
    纯向量检索在本项目语料上的失败模式是「文档找对了，段落找错了」——同一份指南里，概述段与
    具体推荐条目的语义距离很近，向量分不出差别；而具体条目往往带有鲜明的字面特征（药名、
    数值、检查项缩写），BM25 对这些词的精确匹配很敏感，能把含准确实体词的片段顶上来。

融合策略用 RRF（Reciprocal Rank Fusion）：
    score(d) = Σ 1 / (k + rank_i(d))
    只用排名、不用原始分。BM25 分数量纲和余弦分数量纲不可比，硬相加要先归一化且对参数敏感，
    RRF 避免了「用一个加权系数把两路分数调到同一刻度」这种调参陷阱。

边缘干净利落：分词时刻意保留数字与英文缩写（ HbA1c / OGTT / 4.5 这类实体词），
因为医疗 query 的判别力很大一部分来自这些串，用普通中文停用词过滤会把它们洗掉。
"""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .index import FaissIndex, SearchHit

try:  # pragma: no cover - 依赖缺失时的友好提示
    import jieba
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 jieba，请先 pip install jieba") from exc

try:  # pragma: no cover
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 rank_bm25，请先 pip install rank_bm25") from exc


# 保留中英文字符、数字以及数字里的小数点；其余标点一律丢弃
_TOKEN_RE = re.compile(r"[一-鿿]+|[A-Za-z]+[0-9A-Za-z]*|[0-9]+(?:\.[0-9]+)?")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> List[str]:
    """中英混排分词：中文走 jieba，英文/数字整串保留并转小写。"""
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        if _CJK_RE.search(raw):
            tokens.extend(t for t in jieba.lcut(raw) if t.strip())
        else:
            tokens.append(raw.lower())
    return tokens


class BM25Store:
    """对切片集合建 BM25 索引，支持落盘缓存避免每次重复分词。"""

    def __init__(self, chunks: Sequence, tokenizer=tokenize, corpus_tokens: Optional[List[List[str]]] = None):
        self.tokenizer = tokenizer
        self.chunks = list(chunks)
        self.doc_ids: List[str] = [getattr(c, "chunk_id", str(i)) for i, c in enumerate(self.chunks)]
        # 允许外部直接喂入已算好的分词结果（缓存命中时避免重复分词）
        self.corpus_tokens: List[List[str]] = (
            list(corpus_tokens) if corpus_tokens is not None else [tokenizer(c.text) for c in self.chunks]
        )
        self.model: Optional[BM25Okapi] = None
        if any(self.corpus_tokens):
            self.model = BM25Okapi(self.corpus_tokens)

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------ 落盘与加载

    @classmethod
    def build(cls, chunks: Sequence, tokenizer=tokenize) -> "BM25Store":
        return cls(chunks, tokenizer=tokenizer)

    def fingerprint(self) -> str:
        """由切片 id 与文本算指纹，换索引就自动失效。"""
        payload = "\x00".join(
            f"{cid}:{len(chunk.text)}:{hashlib.md5(chunk.text.encode('utf-8')).hexdigest()[:8]}"
            for cid, chunk in zip(self.doc_ids, self.chunks)
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"fingerprint": self.fingerprint(), "tokens": self.corpus_tokens}, f)
        return path

    @classmethod
    def load(cls, chunks: Sequence, path: Path | str, tokenizer=tokenize) -> "BM25Store":
        """带缓存地构造：指纹一致就复用分词结果，否则重建并写回缓存。"""
        path = Path(path)
        # 先按无 tokenizer 的方式拿到空壳，只为算指纹：避免缓存必然不命中时白白分词一遍
        probe = cls(chunks, tokenizer=lambda t: [])
        if path.exists():
            try:
                with path.open("rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == probe.fingerprint():
                    return cls(chunks, corpus_tokens=cached["tokens"])
            except (pickle.UnpicklingError, EOFError, KeyError, AttributeError):
                pass  # 缓存损坏则重建
        store = cls(chunks, tokenizer=tokenizer)
        store.save(path)
        return store

    # ------------------------------------------------------------ 检索

    def retrieve(self, query: str, top_n: int) -> List[Tuple[int, float]]:
        """返回 [(chunk 下标, bm25 分数)]，按分数降序，最多 top_n 条。"""
        if self.model is None or top_n <= 0:
            return []
        scores = self.model.get_scores(self.tokenizer(query))
        n = min(top_n, len(scores))
        if n <= 0:
            return []
        top_idx = np.argpartition(-scores, n - 1)[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def rrf_fuse(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion：只用排名，不用原始分数。

    ranked_lists: 每个元素是一路召回的 chunk 下标序列（按该路分数降序）。
    返回按融合分降序的 [(下标, rrf 分数)]。
    """
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """稀疏 + 稠密两路召回后再 RRF 融合。"""

    def __init__(
        self,
        index: FaissIndex,
        embedder,
        bm25: BM25Store,
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ):
        self.index = index
        self.embedder = embedder
        self.bm25 = bm25
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self._positions: Dict[int, int] = {id(chunk): i for i, chunk in enumerate(index.chunks)}

    # ------------------------------------------------------------ 构造入口

    @classmethod
    def from_index(
        cls,
        index: FaissIndex,
        embedder,
        cache_path: Optional[Path | str] = None,
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        verbose: bool = True,
    ) -> "HybridRetriever":
        chunks = list(index.chunks)
        if cache_path:
            bm25 = BM25Store.load(chunks, cache_path)
        else:
            bm25 = BM25Store.build(chunks)
        if verbose:
            print(f"[hybrid] BM25 索引就绪：{len(bm25)} 片（rrf_k={rrf_k}）")
        return cls(index, embedder, bm25, top_k=top_k, rrf_k=rrf_k, candidate_multiplier=candidate_multiplier)

    # ------------------------------------------------------------ 检索

    def candidate_pool_size(self, top_k: int) -> int:
        """每路各取多少候选参与融合；不足则至少保证覆盖全库。"""
        return min(len(self.index), max(top_k, 20) * self.candidate_multiplier)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchHit]:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        k = int(top_k or self.top_k)
        pool = self.candidate_pool_size(k)

        vector_hits = self.index.search_one(np.asarray(self.embedder.encode_queries(query), dtype=np.float32), top_k=pool)
        vector_ranked = [self._index_of(hit) for hit in vector_hits]
        bm25_ranked = [idx for idx, _ in self.bm25.retrieve(query, pool)]

        fused = rrf_fuse([vector_ranked, bm25_ranked], k=self.rrf_k)[:k]
        return [
            SearchHit(chunk=self.index.chunks[idx], score=float(score), rank=rank)
            for rank, (idx, score) in enumerate(fused, start=1)
        ]

    def _index_of(self, hit: SearchHit) -> int:
        """把命中映射回 chunks 下标。命中对象是 load 时的同一批 chunk，用对象身份定位最快。"""
        pos = self._positions.get(id(hit.chunk))
        if pos is not None:
            return pos
        for i, chunk in enumerate(self.index.chunks):
            if chunk.chunk_id == hit.chunk.chunk_id:
                self._positions[id(chunk)] = i
                return i
        raise ValueError("命中的切片不在当前索引的 chunks 列表里")

    def format_hits(self, hits: Sequence[SearchHit], text_width: int = 200) -> str:
        """把命中结果格式化成人类可读的多行文本（与 Retriever 的输出风格保持一致）。"""
        lines: List[str] = []
        for hit in hits:
            preview = hit.chunk.text.replace("\n", " ")[:text_width]
            lines.append(f"[{hit.rank}] rrf={hit.score:.5f}  {hit.chunk.citation()}")
            lines.append(f"    {preview}")
        return "\n".join(lines)

    def explain(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回带两路原始排名的明细，用于排查融合效果。"""
        k = int(top_k or self.top_k)
        pool = self.candidate_pool_size(k)

        vector_hits = self.index.search_one(np.asarray(self.embedder.encode_queries(query), dtype=np.float32), top_k=pool)
        vector_rank = {self._index_of(h): r for r, h in enumerate(vector_hits, start=1)}
        bm25_rank = {idx: r for r, (idx, _) in enumerate(self.bm25.retrieve(query, pool), start=1)}

        rows: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(rrf_fuse([list(vector_rank), list(bm25_rank)], k=self.rrf_k)[:k], start=1):
            rows.append(
                {
                    "rank": rank,
                    "rrf_score": score,
                    "vector_rank": vector_rank.get(idx),
                    "bm25_rank": bm25_rank.get(idx),
                    "doc_name": self.index.chunks[idx].doc_name,
                    "text": self.index.chunks[idx].text,
                }
            )
        return rows
