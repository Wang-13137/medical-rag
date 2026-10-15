"""FastAPI 服务层的离线测试：用替身资源，不加载任何模型。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import serve  # noqa: E402


def _chunk(text: str, doc: str = "指南A.txt"):
    return SimpleNamespace(text=text, doc_name=doc, heading_path=["概述"])


class FakeGenerator:
    def __init__(self, answer="推荐起始剂量为 5 mg[1]。"):
        self._answer = answer

    def answer(self, query, contexts, titles=None, prompt_mode="strict"):
        return SimpleNamespace(
            answer=self._answer, citations=[1], refused=False, latency_sec=0.01
        )


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.index = SimpleNamespace(chunks=list(range(9404)))

    def retrieve(self, query, top_k=8):
        return self.hits


def _fake_resources():
    hits = [
        SimpleNamespace(chunk=_chunk("推荐起始剂量为 5 mg，每日一次。"), score=0.9),
        SimpleNamespace(chunk=_chunk("最大剂量为 20 mg。"), score=0.5),
    ]
    return {
        "cfg": SimpleNamespace(get=lambda k, d=None: d),
        "mode": "向量召回 + cross-encoder 精排",
        "retriever": FakeRetriever(hits),
        "generator": FakeGenerator(),
        "candidate_k": 20,
    }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_answer_returns_structured_payload(client):
    r = client.post("/v1/answer", json={"question": "起始剂量是多少？"})
    assert r.status_code == 200
    data = r.json()
    assert "5 mg" in data["answer"]
    assert data["citations"] == [1]
    assert data["used_chars"] > 0
    assert data["request_id"]
    assert isinstance(data["verdicts"], list)


def test_verdicts_flag_unsupported_number(client):
    r = client.post("/v1/answer", json={"question": "起始剂量？"})
    data = r.json()
    assert any(v["supported"] for v in data["verdicts"])


def test_cache_hit_on_second_call(client):
    body = {"question": "剂量？", "use_cache": True}
    first = client.post("/v1/answer", json=body).json()
    second = client.post("/v1/answer", json=body).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["request_id"] != first["request_id"]


def test_cache_can_be_disabled(client):
    body = {"question": "剂量？", "use_cache": False}
    assert client.post("/v1/answer", json=body).json()["cached"] is False
    assert client.post("/v1/answer", json=body).json()["cached"] is False


def test_api_key_required_when_env_set(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        assert c.post("/v1/answer", json={"question": "剂量？"}).status_code == 401
        ok = c.post(
            "/v1/answer", json={"question": "剂量？"}, headers={"X-API-Key": "secret"}
        )
        assert ok.status_code == 200


def test_empty_question_rejected(client):
    assert client.post("/v1/answer", json={"question": ""}).status_code == 422


def test_stats_reports_cache_and_index(client):
    client.post("/v1/answer", json={"question": "剂量？"})
    stats = client.get("/v1/stats").json()
    assert stats["cache_size"] >= 1
    assert stats["index_chunks"] == 9404


def test_stats_reports_trace_summary(client):
    client.post("/v1/answer", json={"question": "剂量？"})
    trace = client.get("/v1/stats").json()["trace"]
    assert trace["count"] >= 1
    assert "retrieve" in trace["span_ms_mean"]


# ------------------------------------------------------------------ 多轮


class FakeJSONGenerator(FakeGenerator):
    """带 chat 的替身，用于验证「模型按 schema 输出」这条路径。"""

    def chat(self, messages):
        return '```json\n{"answer": "起始剂量 5 mg[1]", "citations": [1], "refused": false}\n```'


def _json_resources():
    res = _fake_resources()
    res["generator"] = FakeJSONGenerator()
    return res


@pytest.fixture()
def clean_sessions():
    serve.SESSIONS._sessions.clear()
    serve.SESSIONS._order.clear()
    yield
    serve.SESSIONS._sessions.clear()
    serve.SESSIONS._order.clear()


def test_chat_rewrites_followup(clean_sessions):
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        c.post("/v1/chat", json={"session_id": "s1", "question": "妊娠期急性阑尾炎怎么处理"})
        second = c.post("/v1/chat", json={"session_id": "s1", "question": "那疗程多久？"}).json()
    assert second["rewritten_query"] != "那疗程多久？"
    assert "阑尾炎" in second["rewritten_query"]
    assert second["history_used"] is True


def test_chat_first_turn_has_no_history(clean_sessions):
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        first = c.post("/v1/chat", json={"session_id": "s2", "question": "妊娠期急性阑尾炎怎么处理"}).json()
    assert first["history_used"] is False
    assert first["rewritten_query"] == "妊娠期急性阑尾炎怎么处理"


def test_chat_text_format_has_no_structured_field(clean_sessions):
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        data = c.post("/v1/chat", json={"session_id": "s3", "question": "剂量？"}).json()
    assert data["structured"] is None


def test_chat_json_format_falls_back_to_server_side(clean_sessions):
    """生成器不支持 chat 时也要给出完整结构，不能返回 null。"""
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        data = c.post(
            "/v1/chat", json={"session_id": "s4", "question": "剂量？", "response_format": "json"}
        ).json()
    assert data["structured"] is not None
    assert data["structured"]["citations"] == [1]
    assert data["structured"]["answer"]


def test_chat_json_format_uses_model_output(clean_sessions):
    app = serve.create_app(provider=_json_resources)
    with TestClient(app) as c:
        data = c.post(
            "/v1/chat", json={"session_id": "s5", "question": "剂量？", "response_format": "json"}
        ).json()
    assert data["structured"]["answer"] == "起始剂量 5 mg[1]"
    assert data["structured"]["citations"] == [1]


def test_chat_requires_api_key_when_env_set(monkeypatch, clean_sessions):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        denied = c.post("/v1/chat", json={"question": "剂量？"})
        ok = c.post("/v1/chat", json={"question": "剂量？"}, headers={"X-API-Key": "secret"})
    assert denied.status_code == 401
    assert ok.status_code == 200


def test_stats_counts_sessions(clean_sessions):
    app = serve.create_app(provider=_fake_resources)
    with TestClient(app) as c:
        c.post("/v1/chat", json={"session_id": "s6", "question": "剂量？"})
        stats = c.get("/v1/stats").json()
    assert stats["sessions"] == 1
