"""ReAct 智能体的离线测试：全部用脚本化 chat_fn，不联网、不加载模型。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from medical_rag.agent import (
    AgentRun,
    ReActAgent,
    build_system_prompt,
    make_agent,
    parse_action,
    parse_kwargs,
    parse_thought,
)


def _hit(text: str, doc: str = "指南A", score: float = 0.8, head=None):
    return SimpleNamespace(
        chunk=SimpleNamespace(text=text, doc_name=doc, heading_path=head or ["概述"]),
        score=score,
    )


class FakeRetriever:
    def __init__(self, pool):
        self.pool = pool
        self.calls = []

    def retrieve(self, query: str, top_k: int = 8):
        self.calls.append((query, top_k))
        return self.pool[: int(top_k)]


def _scripted(responses):
    """按调用次数依次返回预设回复；用完则抛错，暴露循环失控。"""
    it = iter(responses)

    def chat(messages):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("模型被调用次数超过预设脚本")

    return chat


# ------------------------------------------------------------------ 解析


def test_parse_action_standard():
    text = "思考：先检索\n行动：vector_search\n行动输入：query=高血压, top_k=8"
    assert parse_action(text) == ("vector_search", "query=高血压, top_k=8")
    assert parse_thought(text) == "先检索"


def test_parse_action_inline_args():
    assert parse_action("行动：finish(answer=甲)") == ("finish", "answer=甲")


def test_parse_action_missing():
    assert parse_action("我觉得应该直接回答") == ("", "")


def test_parse_kwargs_variants():
    assert parse_kwargs("query=高血压, top_k=8") == {"query": "高血压", "top_k": 8}
    assert parse_kwargs('{"query": "痛风", "top_k": 3}') == {"query": "痛风", "top_k": 3}
    assert parse_kwargs("高血压") == {"query": "高血压"}
    assert parse_kwargs("") == {}


# ------------------------------------------------------------------ 提示词


def test_system_prompt_lists_all_tools():
    agent, _ = make_agent(lambda m: "", FakeRetriever([]))
    prompt = build_system_prompt(list(agent.tools.values()))
    for name in ("vector_search", "keyword_search", "rerank", "read", "remember", "verify", "finish"):
        assert name in prompt


# ------------------------------------------------------------------ 主循环


def test_finish_stops_and_returns_answer():
    retriever = FakeRetriever([_hit("高血压诊断标准为收缩压≥140 mmHg。")])
    chat = _scripted(
        [
            "思考：先检索\n行动：vector_search\n行动输入：query=高血压",
            "思考：读进来\n行动：read\n行动输入：ids=1",
            "思考：可以作答\n行动：finish\n行动输入：answer=标准为收缩压≥140 mmHg[1]。",
        ]
    )
    agent, _ = make_agent(chat, retriever, max_steps=6, budget_chars=2400)
    run = agent.run("高血压诊断标准？")

    assert run.stop_reason == "finish"
    assert "140" in run.answer
    assert run.refused is False
    assert run.contexts and "140" in run.contexts[0]
    assert run.tool_calls == {"vector_search": 1, "read": 1, "finish": 1}
    assert run.llm_calls == 3


def test_refusal_is_detected():
    chat = _scripted(["思考：没有依据\n行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。"])
    agent, _ = make_agent(chat, FakeRetriever([]))
    run = agent.run("2028 年会发新版卒中指南吗")
    assert run.refused is True
    assert run.stop_reason == "finish"


def test_unknown_tool_is_reported_not_crash():
    chat = _scripted(
        [
            "行动：google_search\n行动输入：query=痛风",
            "行动：finish\n行动输入：answer=无依据，根据提供的指南片段无法回答该问题。",
        ]
    )
    agent, _ = make_agent(chat, FakeRetriever([]))
    run = agent.run("痛风？")
    assert "没有这个工具" in run.steps[0].observation
    assert run.tool_calls.get("google_search") == 1
    assert run.stop_reason == "finish"


def test_parse_failure_is_retried_then_gives_up():
    chat = _scripted(["我直接猜一个答案", "再想想", "还是不会"])
    agent, _ = make_agent(chat, FakeRetriever([]), max_steps=8, max_parse_retries=1)
    run = agent.run("随便问")
    assert run.stop_reason == "parse_failed"
    assert run.answer == ""


def test_max_steps_falls_back_to_refusal():
    chat = _scripted([f"行动：vector_search\n行动输入：query=第{i}次" for i in range(6)])
    agent, _ = make_agent(chat, FakeRetriever([_hit("甲")]), max_steps=3)
    run = agent.run("会不会停不下来")
    assert run.stop_reason == "max_steps"
    assert run.refused is True
    assert run.answer  # 退化为拒答句，而不是空串


def test_budget_is_enforced():
    long_hits = [_hit("甲" * 500), _hit("乙" * 500), _hit("丙" * 500)]
    chat = _scripted(
        [
            "行动：vector_search\n行动输入：query=甲, top_k=3",
            "行动：read\n行动输入：ids=1,2,3",
            "行动：finish\n行动输入：answer=甲[1]。",
        ]
    )
    agent, _ = make_agent(chat, FakeRetriever(long_hits), budget_chars=1200)
    run = agent.run("甲？")
    assert run.used_chars <= 1200
    assert len(run.contexts) == 2  # 第三条放不进预算


def test_remember_stores_notes():
    chat = _scripted(
        [
            "行动：remember\n行动输入：key=诊断阈值, value=140/90",
            "行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。",
        ]
    )
    agent, _ = make_agent(chat, FakeRetriever([]))
    run = agent.run("阈值？")
    assert run.notes == {"诊断阈值": "140/90"}


def test_verify_flags_unsupported_number():
    retriever = FakeRetriever([_hit("推荐起始剂量为 5 mg，每日一次。")])
    chat = _scripted(
        [
            "行动：vector_search\n行动输入：query=剂量",
            "行动：read\n行动输入：ids=1",
            "行动：verify\n行动输入：answer=推荐剂量为 50 mg。",
            "行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。",
        ]
    )
    agent, _ = make_agent(chat, retriever)
    run = agent.run("剂量？")
    assert "找不到依据" in run.steps[2].observation
    assert run.tool_calls.get("verify") == 1


def test_keyword_tool_disabled_by_default():
    chat = _scripted(
        [
            "行动：keyword_search\n行动输入：query=痛风",
            "行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。",
        ]
    )
    agent, _ = make_agent(chat, FakeRetriever([]))
    run = agent.run("痛风")
    assert "BM25 未启用" in run.steps[0].observation


def test_rerank_requires_model():
    chat = _scripted(
        [
            "行动：vector_search\n行动输入：query=甲",
            "行动：rerank\n行动输入：query=甲",
            "行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。",
        ]
    )
    agent, _ = make_agent(chat, FakeRetriever([_hit("甲")]))
    run = agent.run("甲")
    assert "精排模型未加载" in run.steps[1].observation


class FakeReranker:
    def score(self, query, texts):
        # 让包含关键词的排在前面
        return [1.0 if "尿酸" in t else 0.1 for t in texts]


def test_rerank_reorders_working_set():
    retriever = FakeRetriever([_hit("无关内容。"), _hit("尿酸目标值 360 μmol/L。")])
    chat = _scripted(
        [
            "行动：vector_search\n行动输入：query=尿酸, top_k=2",
            "行动：rerank\n行动输入：query=尿酸",
            "行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。",
        ]
    )
    agent, _ = make_agent(chat, retriever, reranker=FakeReranker())
    run = agent.run("尿酸目标值")
    assert agent.evidence[0].text.startswith("尿酸")
    assert run.tool_calls.get("rerank") == 1


def test_bare_text_input_is_aligned_to_tool_parameter():
    """模型把 finish 的输入写成裸答案（不带 answer=）时不能报错。"""
    chat = _scripted(["思考：够用了\n行动：finish\n行动输入：推荐剂量为 5 mg[1]。"])
    agent, _ = make_agent(chat, FakeRetriever([]))
    run = agent.run("剂量？")
    assert run.answer == "推荐剂量为 5 mg[1]。"
    assert run.steps[0].error == ""


def test_bare_ids_are_aligned_to_read():
    retriever = FakeRetriever([_hit("甲。"), _hit("乙。")])
    chat = _scripted(
        [
            "行动：vector_search\n行动输入：query=甲, top_k=2",
            "行动：read\n行动输入：1,2",
            "行动：finish\n行动输入：answer=甲[1]乙[2]。",
        ]
    )
    agent, _ = make_agent(chat, retriever)
    run = agent.run("甲和乙")
    assert len(run.contexts) == 2


def test_min_read_gate_blocks_premature_finish():
    retriever = FakeRetriever([_hit("甲。"), _hit("乙。"), _hit("丙。")])
    chat = _scripted(
        [
            "行动：vector_search\n行动输入：query=甲乙丙, top_k=3",
            "行动：finish\n行动输入：answer=甲[1]。",
            "行动：read\n行动输入：ids=1,2,3",
            "行动：finish\n行动输入：answer=甲[1]乙[2]丙[3]。",
        ]
    )
    agent, _ = make_agent(chat, retriever, min_read=3)
    run = agent.run("甲乙丙")
    assert "未达下限" in run.steps[1].observation
    assert run.answer.endswith("丙[3]。")
    assert len(run.contexts) == 3


def test_min_read_gate_allows_refusal():
    chat = _scripted(["行动：finish\n行动输入：answer=根据提供的指南片段无法回答该问题。"])
    agent, _ = make_agent(chat, FakeRetriever([]), min_read=5)
    run = agent.run("语料里没有的问题")
    assert run.refused is True
    assert run.stop_reason == "finish"


def test_run_serialisable():
    chat = _scripted(["行动：finish\n行动输入：answer=甲[1]。"])
    agent, _ = make_agent(chat, FakeRetriever([]))
    run = agent.run("甲")
    data = run.to_dict()
    assert isinstance(data, dict)
    assert data["num_steps"] == 1
    assert data["tool_calls_total"] == 1
    assert isinstance(AgentRun(query="x"), AgentRun)
