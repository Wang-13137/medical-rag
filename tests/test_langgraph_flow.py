# -*- coding: utf-8 -*-
"""LangGraph 固定编排图的测试：用假检索器与假生成器，不加载模型。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Sequence

import pytest

from medical_rag.langgraph_flow import build_graph

DOC_OK = "妊娠期急性阑尾炎首选腹腔镜手术，推荐疗程为 14 天。"
DOC_OTHER = "术后需要常规随访，注意监测感染指标。"


class FakeChunk:
    def __init__(self, text: str, doc: str = "指南A"):
        self.text = text
        self.doc_name = doc
        self.heading_path = ["第一章"]


class FakeHit:
    def __init__(self, text: str, score: float, rank: int):
        self.chunk = FakeChunk(text)
        self.score = score
        self.rank = rank


class FakeRetriever:
    def __init__(self, texts: Sequence[str] | None = None):
        self.texts = list(texts or [DOC_OK, DOC_OTHER])

    def retrieve(self, query: str, top_k: int = 5):
        return [FakeHit(t, 0.9 - 0.1 * i, i + 1) for i, t in enumerate(self.texts[:top_k])]


class FakeReranker:
    """把顺序倒过来，方便验证精排真的改了顺序。"""

    def score(self, query: str, texts: List[str]):
        return [float(len(t)) for t in texts]


class FakeGenerator:
    def __init__(self, answer: str = "首选腹腔镜手术，疗程 14 天。", citations=(1,)):
        self.answer_text = answer
        self.citations = list(citations)

    def answer(self, question, contexts, titles, mode):
        return SimpleNamespace(answer=self.answer_text, citations=self.citations)


def make_graph(texts=None, generator=None, reranker=None, **kwargs):
    return build_graph(
        FakeRetriever(texts),
        generator or FakeGenerator(),
        reranker=reranker,
        **kwargs,
    )


def test_run_returns_full_payload():
    result = make_graph().run("妊娠阑尾炎怎么办")
    for key in ("answer", "contexts", "titles", "used_chars", "nodes", "latency_sec"):
        assert key in result
    assert result["stop_reason"] == "finish"


def test_node_order_is_fixed():
    result = make_graph().run("妊娠阑尾炎怎么办")
    assert result["nodes"] == ["retrieve", "rerank", "read", "generate", "verify", "finish"]


def test_read_respects_top_n_and_budget():
    result = make_graph(top_n=1, budget_chars=2400).run("问题")
    assert len(result["contexts"]) == 1
    assert result["used_chars"] <= 2400


def test_generation_costs_one_llm_call():
    result = make_graph().run("问题")
    assert result["llm_calls"] == 1


def test_rerank_node_reorders_evidence():
    graph = make_graph(reranker=FakeReranker())
    result = graph.run("问题")
    # 精排按文本长度排序，两条文本长度不同，顺序应当被改动
    assert [t["node"] for t in result["trace"]][1] == "rerank"
    assert "skipped" not in result["trace"][1]


def test_rerank_node_skipped_without_reranker():
    result = make_graph().run("问题")
    assert result["trace"][1]["skipped"] is True


def test_unsupported_sentence_triggers_revision():
    gen = FakeGenerator("妊娠期急性阑尾炎推荐疗程为 30 天，术后需要随访。")
    result = make_graph(texts=[DOC_OK], generator=gen, max_revisions=1).run("疗程多久")
    assert result["revisions"] == 1
    assert "30" not in result["answer"]
    assert "revise" in result["nodes"]


def test_max_revisions_zero_skips_revision():
    gen = FakeGenerator("妊娠期急性阑尾炎推荐疗程为 30 天，术后需要随访。")
    result = make_graph(texts=[DOC_OK], generator=gen, max_revisions=0).run("疗程多久")
    assert result["revisions"] == 0
    assert "revise" not in result["nodes"]
    assert result["unsupported"]


def test_keyword_results_are_merged_and_deduped():
    def keyword_fn(query: str, top_k: int):
        return [
            {"text": DOC_OK, "title": "来源：指南A", "score": 0.5},  # 重复，应被去重
            {"text": "甲硝唑 0.5 g 每 8 小时一次。", "title": "来源：指南B", "score": 0.4},
        ]

    graph = build_graph(FakeRetriever([DOC_OK]), FakeGenerator(), keyword_search=keyword_fn)
    result = graph.run("用药")
    assert len(result["contexts"]) == 2
    assert any("甲硝唑" in c for c in result["contexts"])


def test_trace_records_budget_usage():
    result = make_graph().run("问题")
    read_steps = [t for t in result["trace"] if t["node"] == "read"]
    assert read_steps and read_steps[0]["used_chars"] == result["used_chars"]


def test_empty_retrieval_answers_without_crash():
    class EmptyRetriever:
        def retrieve(self, query, top_k=5):
            return []

    graph = build_graph(EmptyRetriever(), FakeGenerator())
    result = graph.run("问题")
    assert result["contexts"] == []
    assert result["stop_reason"] == "finish"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
