# -*- coding: utf-8 -*-
"""多轮会话记忆与结构化输出的测试。"""

from __future__ import annotations

import json

import pytest

from medical_rag.schema import (
    StructuredAnswer,
    extract_json,
    parse_structured_answer,
    render_json_instruction,
)
from medical_rag.session import SessionMemory, SessionStore


# ------------------------------------------------------------------ 会话记忆


def test_add_keeps_only_recent_turns():
    mem = SessionMemory("s1", max_turns=2)
    mem.add("第一个问题")
    mem.add("第二个问题")
    mem.add("第三个问题")
    assert [t.question for t in mem.turns] == ["第二个问题", "第三个问题"]


def test_followup_needs_history():
    assert SessionMemory().is_followup("这个药怎么用") is False
    mem = SessionMemory()
    mem.add("妊娠期急性阑尾炎怎么处理")
    assert mem.is_followup("这个病要手术吗") is True


def test_short_weak_query_counts_as_followup():
    mem = SessionMemory()
    mem.add("妊娠期急性阑尾炎怎么处理")
    assert mem.is_followup("疗程多久") is True


def test_new_topic_is_not_followup():
    mem = SessionMemory()
    mem.add("妊娠期急性阑尾炎怎么处理")
    assert mem.is_followup("妊娠期糖尿病的筛查标准是什么") is False


def test_rewrite_prepends_topic_terms():
    mem = SessionMemory()
    mem.add("妊娠期急性阑尾炎怎么处理")
    rewritten = mem.rewrite("那疗程多久？")
    assert "疗程" in rewritten
    assert rewritten != "那疗程多久？"
    assert "阑尾炎" in rewritten


def test_rewrite_leaves_standalone_query_untouched():
    mem = SessionMemory()
    mem.add("妊娠期急性阑尾炎怎么处理")
    assert mem.rewrite("妊娠期糖尿病的筛查标准是什么") == "妊娠期糖尿病的筛查标准是什么"


def test_history_block_prefers_recent_and_is_bounded():
    mem = SessionMemory(max_turns=3, history_chars=120)
    mem.add("问题一", "答案一" * 200)
    mem.add("问题二", "答案二" * 200)
    block = mem.history_block()
    assert "问题二" in block  # 预算不够时保住最近的
    # 每轮截断到 120 字，超预算后不再往上加，长度不会失控
    assert len(block) <= 120 + 140


def test_history_block_empty_without_turns():
    assert SessionMemory().history_block() == ""


def test_history_block_includes_finished_turns():
    mem = SessionMemory()
    mem.add("只有一个问题", "只有一个答案")
    assert "只有一个问题" in mem.history_block()


def test_preferred_titles_come_from_last_turn():
    mem = SessionMemory()
    mem.add("问一", "答一", titles=["来源：指南A"])
    assert mem.preferred_titles() == ["来源：指南A"]


def test_rank_contexts_prioritizes_previous_source():
    mem = SessionMemory()
    mem.add("问一", "答一", titles=["来源：指南B"])
    order = mem.rank_contexts(["a", "b", "c"], ["来源：指南A", "来源：指南B", "来源：指南C"])
    assert order.index(1) == 0


def test_rank_contexts_untouched_without_history():
    assert SessionMemory().rank_contexts(["a", "b"], ["t1", "t2"]) == [0, 1]


def test_roundtrip_serialization():
    mem = SessionMemory("s1")
    mem.add("问一", "答一", titles=["t"], citations=[1], contexts=3)
    restored = SessionMemory.from_dict(mem.to_dict())
    assert restored.session_id == "s1"
    assert restored.turns[0].citations == [1]
    assert restored.turns[0].contexts == 3


def test_session_store_evicts_oldest():
    store = SessionStore(capacity=2)
    store.get("a")
    store.get("b")
    store.get("c")
    assert len(store) == 2
    assert store._sessions.get("a") is None
    assert store._sessions.get("c") is not None


# ------------------------------------------------------------------ 结构化输出


def test_instruction_contains_citation_bound():
    assert "1 到 5" in render_json_instruction(5)


def test_extract_json_strips_fence():
    raw = '```json\n{"answer": "手术[1]", "citations": [1]}\n```'
    assert extract_json(raw)["answer"] == "手术[1]"


def test_extract_json_from_chatty_output():
    raw = '好的，这是结果：\n{"answer": "疗程 14 天[2]", "citations": [2]}\n希望有帮助'
    assert extract_json(raw)["citations"] == [2]


def test_extract_json_returns_none_on_garbage():
    assert extract_json("我不是 JSON") is None


def test_parse_structured_answer_basic():
    raw = json.dumps({"answer": "首选腹腔镜[1]", "citations": [1], "refused": False}, ensure_ascii=False)
    model = parse_structured_answer(raw, n_contexts=3)
    assert model is not None and model.citations == [1]
    assert model.refused is False


def test_citations_out_of_range_are_dropped():
    raw = json.dumps({"answer": "x[9]", "citations": [1, 9, 12]}, ensure_ascii=False)
    model = parse_structured_answer(raw, n_contexts=3)
    assert model.citations == [1]


def test_missing_citations_fall_back_to_text():
    raw = json.dumps({"answer": "首选腹腔镜[2]"}, ensure_ascii=False)
    model = parse_structured_answer(raw, n_contexts=3)
    assert model.citations == [2]


def test_empty_answer_is_rejected():
    assert parse_structured_answer('{"answer": ""}', n_contexts=1) is None


def test_model_json_is_serializable():
    model = StructuredAnswer(answer="a", citations=[1], grounding=0.5)
    payload = json.loads(model.model_dump_json())
    assert payload["grounding"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
