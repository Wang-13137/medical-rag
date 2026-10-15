# -*- coding: utf-8 -*-
"""MCP 工具服务的测试：只验协议与工具语义，不加载真实模型。"""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from typing import List

import pytest

from medical_rag.mcp_server import MCPToolServer, PROTOCOL_VERSION


# ------------------------------------------------------------------ 假资源


class FakeChunk:
    def __init__(self, text: str, doc: str = "指南A", head: List[str] | None = None):
        self.text = text
        self.doc_name = doc
        self.heading_path = head or ["第一章"]


class FakeHit:
    def __init__(self, text: str, score: float, rank: int):
        self.chunk = FakeChunk(text)
        self.score = score
        self.rank = rank


class FakeRetriever:
    """只实现 retrieve，故意不实现 explain，用来测降级分支。"""

    def __init__(self):
        self.index = SimpleNamespace(chunks=[1, 2, 3])

    def retrieve(self, query: str, top_k: int = 5):
        return [FakeHit("妊娠合并急性阑尾炎首选腹腔镜手术。", 0.91, 1)][:top_k]


class FakeRerankRetriever(FakeRetriever):
    def explain(self, query: str, top_k: int = 5):
        return [
            {
                "rank": 1,
                "rerank_score": 0.87,
                "first_stage_rank": 3,
                "doc_name": "指南A",
                "text": "推荐剂量为 14 d。",
            }
        ]


class FakeGenerator:
    def answer(self, question, contexts, titles, mode):
        return SimpleNamespace(answer="首选腹腔镜手术[1]。", citations=[1])


def make_provider(rerank: bool = False, keyword: bool = False):
    def provider() -> dict:
        return {
            "cfg": SimpleNamespace(get=lambda k, d=None: "W6"),
            "mode": "vector+rerank" if rerank else "vector",
            "retriever": FakeRerankRetriever() if rerank else FakeRetriever(),
            "generator": FakeGenerator(),
            "candidate_k": 20,
            "keyword_fn": (lambda q, k: [{"title": "t", "text": "文本", "score": 0.5}]) if keyword else None,
        }

    return provider


def call(server: MCPToolServer, name: str, arguments: dict) -> dict:
    raw = json.dumps(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        ensure_ascii=False,
    )
    reply = json.loads(server.handle(raw))
    assert "result" in reply, reply
    return reply["result"]


# ------------------------------------------------------------------ 协议


def test_initialize_returns_protocol_and_server_info():
    server = MCPToolServer(make_provider())
    raw = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    result = json.loads(server.handle(raw))["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "medical-rag"
    assert "tools" in result["capabilities"]


def test_initialize_echoes_supported_client_version():
    server = MCPToolServer(make_provider())
    raw = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
    )
    assert json.loads(server.handle(raw))["result"]["protocolVersion"] == "2024-11-05"


def test_tools_list_has_six_tools_with_schemas():
    server = MCPToolServer(make_provider())
    raw = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = json.loads(server.handle(raw))["result"]["tools"]
    names = [t["name"] for t in tools]
    assert set(names) == {
        "guideline_search",
        "keyword_search",
        "rerank_explain",
        "verify_answer",
        "answer_question",
        "index_stats",
    }
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]


def test_unknown_method_returns_method_not_found():
    server = MCPToolServer(make_provider())
    raw = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "does_not_exist"})
    assert json.loads(server.handle(raw))["error"]["code"] == -32601


def test_broken_json_returns_parse_error():
    server = MCPToolServer(make_provider())
    assert json.loads(server.handle("{oops"))["error"]["code"] == -32700


def test_notification_returns_none():
    server = MCPToolServer(make_provider())
    raw = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert server.handle(raw) is None


def test_ping_returns_empty_result():
    server = MCPToolServer(make_provider())
    raw = json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"})
    assert json.loads(server.handle(raw))["result"] == {}


# ------------------------------------------------------------------ 工具语义


def test_guideline_search_returns_structured_content():
    server = MCPToolServer(make_provider())
    result = call(server, "guideline_search", {"query": "妊娠阑尾炎", "top_k": 3})
    payload = result["structuredContent"]
    assert payload["count"] == 1
    assert "指南A" in payload["results"][0]["source"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"


def test_unknown_tool_call_is_flagged_as_error():
    server = MCPToolServer(make_provider())
    result = call(server, "nope", {})
    assert result["isError"] is True
    assert "未知工具" in result["structuredContent"]["error"]


def test_verify_answer_flags_unsupported_sentence():
    server = MCPToolServer(make_provider())
    result = call(
        server,
        "verify_answer",
        {
            "answer": "妊娠期急性阑尾炎推荐疗程为 30 天，术后需要随访。",
            "contexts": ["妊娠期急性阑尾炎推荐疗程为 14 天，术后需要随访。"],
        },
    )
    assert result["structuredContent"]["unsupported_sentences"]


def test_verify_answer_accepts_supported_sentence():
    server = MCPToolServer(make_provider())
    result = call(
        server,
        "verify_answer",
        {
            "answer": "妊娠期急性阑尾炎推荐疗程为 14 天，术后需要随访。",
            "contexts": ["妊娠期急性阑尾炎推荐疗程为 14 天，术后需要随访。"],
        },
    )
    assert result["structuredContent"]["unsupported_sentences"] == []


def test_answer_question_returns_citations_and_budget():
    server = MCPToolServer(make_provider())
    result = call(server, "answer_question", {"question": "妊娠阑尾炎怎么处理", "top_n": 2, "budget": 500})
    payload = result["structuredContent"]
    assert payload["citations"] == [1]
    assert payload["used_chars"] > 0
    assert payload["generator"] == "FakeGenerator"


def test_index_stats_reports_chunk_count():
    server = MCPToolServer(make_provider())
    assert call(server, "index_stats", {})["structuredContent"]["index_chunks"] == 3


def test_keyword_tool_degrades_when_disabled():
    server = MCPToolServer(make_provider())
    payload = call(server, "keyword_search", {"query": "甲硝唑"})["structuredContent"]
    assert payload["available"] is False


def test_keyword_tool_works_when_enabled():
    server = MCPToolServer(make_provider(keyword=True))
    payload = call(server, "keyword_search", {"query": "甲硝唑"})["structuredContent"]
    assert payload["available"] is True and payload["count"] == 1


def test_rerank_tool_degrades_without_cross_encoder():
    server = MCPToolServer(make_provider())
    assert call(server, "rerank_explain", {"query": "x"})["structuredContent"]["available"] is False


def test_rerank_tool_explains_when_enabled():
    server = MCPToolServer(make_provider(rerank=True))
    payload = call(server, "rerank_explain", {"query": "x"})["structuredContent"]
    assert payload["available"] is True
    assert payload["results"][0]["first_stage_rank"] == 3


def test_tool_exception_does_not_kill_server():
    server = MCPToolServer(make_provider())

    def boom(**kwargs):
        raise RuntimeError("检索挂了")

    server._tool_search = boom  # tools() 每次新建列表，只能换实例方法
    result = call(server, "guideline_search", {"query": "x"})
    assert result["isError"] is True
    assert "RuntimeError" in result["structuredContent"]["error"]


def test_serve_stdio_roundtrip():
    server = MCPToolServer(make_provider())
    stdin = StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
    )
    stdout = StringIO()
    server.serve_stdio(stdin, stdout)
    lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1  # 通知不回包
    assert "guideline_search" in lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
