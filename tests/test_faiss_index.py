# -*- coding: utf-8 -*-
"""FAISS 索引测试：构建、持久化、重载一致性、维度校验。"""

from __future__ import annotations

import numpy as np
import pytest

from medical_rag.chunking import Chunk
from medical_rag.index import FaissIndex


def _random_vectors(count: int, dim: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    # L2 归一化，使 IndexFlatIP 的内积等价于余弦相似度
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _fake_chunks(count: int, doc_name: str = "样本.md") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_name}-{index:04d}",
            text=f"第 {index} 段示例文本，用于验证索引与元数据的对齐关系。",
            doc_id=doc_name.rsplit(".", 1)[0],
            doc_name=doc_name,
            source_path=f"D:/语料/{doc_name}",
            file_type="md",
            page=index // 5 + 1,
            heading_path=[f"第 {index} 节"],
            char_start=index * 100,
            char_end=index * 100 + 20,
            chunk_index=index,
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------- 1. 保存后重载结果一致

def test_save_and_load_keeps_search_results_identical(tmp_path):
    dim, count = 16, 12
    vectors = _random_vectors(count, dim)
    chunks = _fake_chunks(count)

    index = FaissIndex.build(vectors, chunks, meta={"model": {"model_name": "unit-test"}})
    index_file, meta_file = index.save(tmp_path / "unit.index", tmp_path / "unit_meta.json")
    assert index_file.exists() and meta_file.exists()

    reloaded = FaissIndex.load(index_file, meta_file)

    assert reloaded.dimension == dim
    assert len(reloaded.chunks) == count
    assert reloaded.chunks[3].chunk_id == chunks[3].chunk_id

    query = _random_vectors(1, dim, seed=99)
    before = index.search_one(query, top_k=5)
    after = reloaded.search_one(query, top_k=5)

    assert [hit.chunk.chunk_id for hit in before] == [hit.chunk.chunk_id for hit in after], "重载后命中切片顺序应一致"
    assert np.allclose(
        [hit.score for hit in before], [hit.score for hit in after], atol=1e-6
    ), "重载后相似度应逐位相等"


# ---------------------------------------------------------------- 2. 元数据溯源字段不丢失

def test_metadata_fields_survive_roundtrip(tmp_path):
    vectors = _random_vectors(4, 8)
    chunks = _fake_chunks(4, doc_name="指南.pdf")

    index = FaissIndex.build(vectors, chunks)
    reloaded_after = FaissIndex.load(*index.save(tmp_path / "rt.index", tmp_path / "rt_meta.json"))
    chunk = reloaded_after.chunks[2]

    assert chunk.doc_name == "指南.pdf"
    assert chunk.source_path == "D:/语料/指南.pdf"
    assert chunk.heading_path == ["第 2 节"]
    assert chunk.char_start == 200 and chunk.char_end == 220
    assert reloaded_after.meta["num_chunks"] == 4


# ---------------------------------------------------------------- 3. 缺失文件与数量不一致要报错

def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaissIndex.load(tmp_path / "不存在.index", tmp_path / "不存在_meta.json")


def test_mismatched_vector_and_chunk_count_raises():
    chunks = _fake_chunks(3)
    with pytest.raises(ValueError, match="不一致"):
        FaissIndex.build(_random_vectors(5, 8), chunks)


def test_wrong_dimension_raises():
    index = FaissIndex.build(_random_vectors(3, 8), _fake_chunks(3))
    with pytest.raises(ValueError, match="不一致"):
        index.search(_random_vectors(1, 16), top_k=1)


# ---------------------------------------------------------------- 4. 行为正确性

def test_search_returns_nearest_first():
    dim = 8
    vectors = _random_vectors(20, dim, seed=3)
    chunks = _fake_chunks(20)
    index = FaissIndex.build(vectors, chunks)

    # 用第 5 条自身的向量去检索，Top-1 必须是它自己
    hits = index.search_one(vectors[5], top_k=3)
    assert hits[0].chunk.chunk_id == chunks[5].chunk_id
    assert hits[0].score > hits[1].score >= hits[2].score, "结果应按相似度降序"
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_batch_search_shape():
    dim = 8
    index = FaissIndex.build(_random_vectors(10, dim, seed=11), _fake_chunks(10))
    results = index.search(_random_vectors(4, dim, seed=12), top_k=2)
    assert len(results) == 4
    assert all(len(row) == 2 for row in results)


def test_topk_is_capped_by_corpus_size():
    index = FaissIndex.build(_random_vectors(3, 8), _fake_chunks(3))
    hits = index.search_one(_random_vectors(1, 8, seed=5), top_k=10)
    assert len(hits) == 3, "top_k 超过语料规模时应被裁剪"
