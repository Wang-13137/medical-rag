# -*- coding: utf-8 -*-
"""混合检索（BM25 + 向量 RRF 融合）的测试。

分片-level 的行为用最小合成语料验证，不依赖真实大模型也跑得动：
向量侧用一个确定性的假 embedder 代替 bge，BM25 侧是真的 rank_bm25。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from medical_rag.chunking import Chunk  # noqa: E402
from medical_rag.hybrid import BM25Store, HybridRetriever, rrf_fuse, tokenize  # noqa: E402
from medical_rag.index import FaissIndex  # noqa: E402


def make_chunk(text: str, doc_name: str, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_name}-{idx}",
        text=text,
        doc_id=doc_name,
        doc_name=doc_name,
        source_path=f"/tmp/{doc_name}.txt",
        file_type="txt",
        page=None,
        heading_path="",
        char_start=idx * 100,
        char_end=idx * 100 + len(text),
        chunk_index=idx,
    )


CORPUS = [
    "2型糖尿病患者糖化血红蛋白 HbA1c 控制目标一般为 7%",
    "高血压诊断标准：诊室血压 ≥140/90 mmHg",
    "急性缺血性脑卒中静脉溶栓时间窗为发病 4.5 小时内，常用阿替普酶",
    "儿童发热可用布洛芬或对乙酰氨基酚退热",
]


class HashEmbedder:
    """确定性假 embedder：把文本投影成固定维度的稀疏哈希向量并归一化。"""

    dimension = 32
    model_name = "hash-stub"

    def encode_queries(self, texts):
        return np.asarray([self._encode(t) for t in ([texts] if isinstance(texts, str) else texts)], dtype=np.float32)

    encode_documents = encode_queries

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for ch in text:
            vec[hash(ch) % self.dimension] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


def build_fake_index() -> FaissIndex:
    embedder = HashEmbedder()
    chunks = [make_chunk(t, f"doc{i}.txt", i) for i, t in enumerate(CORPUS)]
    vectors = embedder.encode_documents(CORPUS)
    return FaissIndex.build(vectors, chunks, meta={"model": {"model_name": "hash-stub"}})


# ---------------------------------------------------------------- 分词

def test_tokenize_keeps_numbers_and_abbrev():
    tokens = tokenize("糖化血红蛋白 HbA1c 控制目标 7%，OGTT 检查")
    lowered = [t.lower() for t in tokens]
    assert "hba1c" in lowered
    assert "7" in lowered
    assert "ogtt" in lowered
    assert any("糖化" in t or "血红蛋白" in t for t in tokens)


def test_tokenize_drops_punctuation():
    tokens = tokenize("诊断标准：≥140/90 mmHg（2024年版）。")
    assert all(t.strip() for t in tokens)
    assert not any(t in {"：", "（", "）", "。", "/"} for t in tokens)


# ---------------------------------------------------------------- RRF

def test_rrf_fuse_single_list_preserves_order():
    fused = rrf_fuse([[10, 20, 30]], k=60)
    assert [idx for idx, _ in fused] == [10, 20, 30]


def test_rrf_fuse_rewards_agreement():
    # 5 号在两路都进前排，7 号只在第一路且排名更后，融合后 5 号必须靠前
    fused = rrf_fuse([[7, 5], [5, 9, 7]], k=60)
    order = [idx for idx, _ in fused]
    assert order.index(5) < order.index(7)


def test_rrf_fuse_score_is_reciprocal():
    fused = dict(rrf_fuse([[3]], k=60))
    assert fused[3] == pytest.approx(1 / 61)


# ---------------------------------------------------------------- BM25

def test_bm25_finds_distinctive_term():
    store = BM25Store.build([make_chunk(t, f"d{i}.txt", i) for i, t in enumerate(CORPUS)])
    top = store.retrieve("儿童退热用布洛芬", top_n=1)
    assert top and top[0][0] == 3  # 第 4 条讲布洛芬


def test_bm25_empty_corpus_does_not_crash():
    """语料全为空时 BM25Okapi 会因 idf 除零崩溃，这里必须优雅退化。"""
    store = BM25Store.build([make_chunk("", "empty.txt", 0)])
    assert store.model is None
    assert store.retrieve("任意查询", top_n=5) == []


def test_bm25_cache_roundtrip(tmp_path):
    chunks = [make_chunk(t, f"d{i}.txt", i) for i, t in enumerate(CORPUS)]
    cache = tmp_path / "bm25.pkl"
    first = BM25Store.load(chunks, cache)
    assert cache.exists()

    # 指纹一致时应直接复用缓存里的分词结果
    second = BM25Store.load(chunks, cache)
    assert second.corpus_tokens == first.corpus_tokens

    # 文本变了（指纹变了）应重建，而不是拿旧结果糊弄
    changed = list(chunks)
    changed[0] = make_chunk("完全不同的内容：肺炎抗菌治疗", "d0.txt", 0)
    third = BM25Store.load(changed, cache)
    assert third.corpus_tokens[0] != first.corpus_tokens[0]


def test_bm25_corrupt_cache_rebuilds(tmp_path):
    chunks = [make_chunk(t, f"d{i}.txt", i) for i, t in enumerate(CORPUS)]
    cache = tmp_path / "broken.pkl"
    cache.write_bytes(b"\x00not-a-pickle")
    store = BM25Store.load(chunks, cache)
    assert len(store.corpus_tokens) == len(CORPUS)


# ---------------------------------------------------------------- 混合检索端到端

def test_hybrid_retrieve_returns_ranked_hits():
    index = build_fake_index()
    embedder = HashEmbedder()
    retriever = HybridRetriever.from_index(index, embedder, verbose=False)
    hits = retriever.retrieve("儿童发热布洛芬", top_k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score >= hits[1].score
    assert hits[0].chunk.doc_name == "doc3.txt"


def test_hybrid_top_k_is_respected():
    index = build_fake_index()
    retriever = HybridRetriever.from_index(index, HashEmbedder(), verbose=False)
    assert len(retriever.retrieve("血压", top_k=3)) == 3
    assert len(retriever.retrieve("血压", top_k=99)) <= len(index)


def test_hybrid_rejects_empty_query():
    index = build_fake_index()
    retriever = HybridRetriever.from_index(index, HashEmbedder(), verbose=False)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_hybrid_explain_reports_both_rankings():
    index = build_fake_index()
    retriever = HybridRetriever.from_index(index, HashEmbedder(), verbose=False)
    rows = retriever.explain("阿替普酶溶栓 4.5 小时", top_k=2)
    assert len(rows) == 2
    assert rows[0]["rank"] == 1
    # 至少第一条在某一路里有排名，否则说明融合逻辑没真的取到候选
    assert rows[0]["vector_rank"] is not None or rows[0]["bm25_rank"] is not None
    assert rows[0]["doc_name"] == "doc2.txt"
