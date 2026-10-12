# -*- coding: utf-8 -*-
"""二阶段检索（召回 + cross-encoder 精排）的测试。

真模型有 2.2GB，单元测试不能依赖它，因此这里用一个可控的替身打分器：
它按「候选文本里有没有预定的关键词」给分，足够验证「精排改变了顺序」这件事本身。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from medical_rag.index import SearchHit  # noqa: E402
from medical_rag.rerank import RerankRetriever  # noqa: E402

sys.path.insert(0, str(ROOT_DIR / "tests"))
from test_hybrid import CORPUS, HashEmbedder, build_fake_index, make_chunk  # noqa: E402


class KeywordReranker:
    """确定性替身：命中预设关键词给高分，否则给负分。"""

    def __init__(self, keyword: str):
        self.keyword = keyword

    def score(self, query: str, texts) -> np.ndarray:
        return np.asarray([10.0 if self.keyword in t else -1.0 for t in texts], dtype=np.float32)


def build_retriever(keyword: str, candidate_k: int = 4, top_k: int = 2):
    index = build_fake_index()
    base = _BaseStub(index)
    return RerankRetriever(base, KeywordReranker(keyword), candidate_k=candidate_k, top_k=top_k)


class _BaseStub:
    """第一阶段替身：固定按 CORPUS 顺序返回，模拟「召回对了文档但段落顺序不理想」。"""

    def __init__(self, index, embedder=None):
        self.index = index
        self.embedder = embedder
        self.calls: list[int] = []

    def retrieve(self, query: str, top_k: int):
        self.calls.append(top_k)
        chunks = list(self.index.chunks)[:top_k]
        return [SearchHit(chunk=c, score=1.0 - 0.01 * i, rank=i + 1) for i, c in enumerate(chunks)]


def test_reranker_reorders_by_relevance():
    # 第 4 条（讲布洛芬）在召回顺序里排最后，精排后应该升到第 1
    retriever = build_retriever("布洛芬", candidate_k=4, top_k=2)
    hits = retriever.retrieve("任意查询")
    assert hits[0].chunk.text == CORPUS[3]
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score


def test_first_stage_is_widened_to_candidate_k():
    # 哪怕最终只要 2 条，第一阶段也必须按 candidate_k 宽召回，否则精排没有挑选空间
    retriever = build_retriever("布洛芬", candidate_k=4, top_k=2)
    retriever.retrieve("任意查询")
    assert retriever.first_stage.calls[0] >= 4


def test_top_k_is_respected():
    retriever = build_retriever("布洛芬", candidate_k=4, top_k=1)
    assert len(retriever.retrieve("任意查询")) == 1


def test_empty_candidates_returns_empty():
    index = build_fake_index()
    base = _BaseStub(index)
    base.retrieve = lambda query, top_k: []  # type: ignore[assignment]
    retriever = RerankRetriever(base, KeywordReranker("任意"), candidate_k=4, top_k=2)
    assert retriever.retrieve("任意查询") == []


def test_rejects_empty_query():
    retriever = build_retriever("布洛芬")
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_explain_reports_rank_change():
    retriever = build_retriever("布洛芬", candidate_k=4, top_k=4)
    rows = retriever.explain("任意查询", top_k=4)
    top, = [r for r in rows if r["doc_name"] == "doc3.txt"]
    assert top["first_stage_rank"] == 4  # 召回时排第 4
    assert rows[0]["doc_name"] == "doc3.txt"  # 精排后被提到第 1
    assert rows[0]["rerank_score"] > rows[-1]["rerank_score"]


@pytest.mark.skipif(
    not Path(r"%USERPROFILE%/.cache/modelscope/models/BAAI--bge-reranker-base/snapshots/master").exists(),
    reason="本机没下载 bge-reranker-base 时跳过真实模型冒烟",
)
def test_real_reranker_discriminates():
    """真模型冒烟：相关句得分必须明显高于无关句。"""
    from medical_rag.rerank import CrossEncoderReranker

    local = Path(r"%USERPROFILE%/.cache/modelscope/models/BAAI--bge-reranker-base/snapshots/master")
    reranker = CrossEncoderReranker(local_model_dir=local, verbose=False)
    scores = reranker.score(
        "哮喘长期控制药物",
        ["吸入性糖皮质激素是哮喘长期控制的首选药物", "本指南适用于基层医疗卫生机构的医务工作者"],
    )
    assert scores[0] > scores[1] + 5
