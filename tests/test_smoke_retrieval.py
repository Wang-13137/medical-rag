# -*- coding: utf-8 -*-
"""端到端冒烟测试：用 data/smoke 的样例注释走完「解析 → 切片 → 向量化 → 建索引 → 检索」。

注意：data/smoke 下的数据仅为样例，标注为冒烟测试数据，不是真实医疗知识库。
"""

from __future__ import annotations

import pytest

from pathlib import Path

from medical_rag.chunking import chunk_documents_from_config
from medical_rag.config import ROOT_DIR
from medical_rag.embedding import BGEEmbedder
from medical_rag.index import FaissIndex
from medical_rag.ingest import load_documents
from medical_rag.retrieve import Retriever


@pytest.fixture(scope="session")
def smoke_index_path(tmp_path_factory, config, embedder: BGEEmbedder):
    """在临时目录构建一次 smoke 索引，供本文件全部用例复用。"""
    smoke_dir = Path(config.get("paths.smoke_data_dir"))
    resolved = smoke_dir if smoke_dir.is_absolute() else (ROOT_DIR / smoke_dir)

    documents = load_documents(resolved)
    assert documents, f"样例数据为空：{resolved}"
    chunks = chunk_documents_from_config(documents, config)
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])

    target = tmp_path_factory.mktemp("index") / "smoke.index"
    index = FaissIndex.build(vectors, chunks, meta={"model": embedder.metadata()})
    index.save(target, target.with_name("smoke_meta.json"))
    return target


@pytest.fixture(scope="session")
def smoke_retriever(smoke_index_path, config, embedder: BGEEmbedder) -> Retriever:
    return Retriever(
        index=FaissIndex.load(smoke_index_path, smoke_index_path.with_name("smoke_meta.json")),
        embedder=embedder,
        top_k=int(config.get("retrieval.top_k", 5)),
        score_floor=float(config.get("retrieval.score_floor", 0.0)),
    )


@pytest.mark.parametrize(
    "query, expected_heading",
    [
        ("2型糖尿病血糖控制目标", "2型糖尿病的血糖控制目标"),
        ("服用华法林期间需要监测什么指标", "华法林的抗凝监测"),
        ("他汀类药物使用要注意哪些不良反应", "他汀类药物的使用注意事项"),
        ("怎么诊断社区获得性肺炎", "社区获得性肺炎的诊断要点"),
        ("评估肾功能应该看哪个指标", "肾功能与估算肾小球滤过率"),
    ],
)
def test_smoke_top1_hits_expected_chunk(smoke_retriever: Retriever, query: str, expected_heading: str):
    hits = smoke_retriever.retrieve(query, top_k=3)

    assert hits, "检索结果不应为空"
    assert expected_heading in hits[0].chunk.heading, (
        f"Top-1 应为「{expected_heading}」，实际是「{hits[0].chunk.heading}」"
        f"｜score={hits[0].score:.4f}｜{hits[0].chunk.citation()}"
    )
    assert hits[0].score >= 0.6, f"Top-1 相似度偏低：{hits[0].score:.4f}"


def test_smoke_retrieval_returns_citation_and_text(smoke_retriever: Retriever):
    hits = smoke_retriever.retrieve("老年人血压应该降到多少", top_k=2)

    assert len(hits) == 2
    top = hits[0]
    assert top.chunk.doc_name.endswith(".md"), "命中切片应能追溯到源文件名"
    assert top.chunk.citation(), "应生成可读的引用串"
    assert "降压" in top.chunk.text or "血压" in top.chunk.text
    # 分数按降序，且排序名次连续
    assert hits[0].score >= hits[1].score
    assert [hit.rank for hit in hits] == [1, 2]


def test_smoke_batch_retrieval_matches_single(smoke_retriever: Retriever):
    queries = ["长期血糖管理", "抗菌药物的使用原则"]
    batch = smoke_retriever.retrieve_batch(queries, top_k=1)

    assert len(batch) == 2
    for query, hits in zip(queries, batch):
        single = smoke_retriever.retrieve(query, top_k=1)
        assert hits[0].chunk.chunk_id == single[0].chunk.chunk_id, "批量与单条检索结果应一致"


def test_smoke_index_stats(smoke_retriever: Retriever):
    stats = smoke_retriever.stats()
    assert stats["dimension"] == 512
    assert stats["num_chunks"] >= 20, "样例数据约 20 条医学条目，切片数不应少于此"
    assert stats["model_name"] == "BAAI/bge-small-zh-v1.5"


def test_empty_query_is_rejected(smoke_retriever: Retriever):
    with pytest.raises(ValueError):
        smoke_retriever.retrieve("   ")
