#!/usr/bin/env python
"""FastAPI 服务层：把检索问答链封装成生产可用的 HTTP 接口。

AI 应用岗 JD 里反复出现的几条，这里逐条落实：
- 服务化：/v1/answer 同步接口，/healthz 探活，/v1/stats 观测接口
- 鉴权：设置环境变量 SERVICE_API_KEY 后，请求必须带 X-API-Key，否则 401
- 结构化输出：请求与响应都用 pydantic 模型约束，答案带 citations / verdicts 字段
- 可观测性：每次请求打一条 JSON 日志（request_id、延迟、上下文字数、缓存命中）
- 缓存：同参数问题走 LRU 缓存，避免重复付检索与模型调用的钱
- 异常处理：模型侧出错返回 503 且不泄漏堆栈；缺 Key 自动退回离线替身

用法：
    python scripts/serve.py --port 8000          # 或 uvicorn scripts.serve:app
    curl -X POST localhost:8000/v1/answer -H "Content-Type: application/json" \
         -d '{"question":"妊娠合并急性阑尾炎首选手术还是保守治疗？"}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from medical_rag.config import load_config  # noqa: E402
from medical_rag.generate import (  # noqa: E402
    DeepSeekGenerator,
    EchoGenerator,
    build_context_block,
    load_generator,
    parse_citations,
)
from medical_rag.schema import parse_structured_answer, render_json_instruction  # noqa: E402
from medical_rag.session import SessionStore  # noqa: E402
from medical_rag.telemetry import Tracer, tracer_from_env  # noqa: E402
from medical_rag.verify import verify_answer  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402
from run_generation import build_contexts  # noqa: E402

CACHE_MAX = int(os.environ.get("SERVICE_CACHE_MAX", "128"))
_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
TRACER: Tracer = tracer_from_env()
SESSIONS = SessionStore(capacity=int(os.environ.get("SERVICE_SESSIONS", "200")))


# ------------------------------------------------------------------ 请求与响应模型


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    mode: str = Field("strict", description="strict | naive | agent")
    top_n: int = Field(8, ge=1, le=12)
    budget: int = Field(2400, ge=400, le=4000)
    min_read: int = Field(0, ge=0, le=8, description="Agent 欠召回护栏，0 表示不设限")
    use_cache: bool = True


class Verdict(BaseModel):
    sentence: str
    supported: bool
    coverage: float
    unsupported_numbers: List[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool
    citations: List[int] = Field(default_factory=list)
    titles: List[str] = Field(default_factory=list)
    contexts: List[str] = Field(default_factory=list)
    verdicts: List[Verdict] = Field(default_factory=list)
    retrieval_mode: str = ""
    generator: str = ""
    used_chars: int = 0
    latency_ms: int = 0
    cached: bool = False


class ChatRequest(BaseModel):
    session_id: str = Field("default", max_length=64)
    question: str = Field(..., min_length=1, max_length=500)
    mode: str = Field("strict", description="strict | naive")
    top_n: int = Field(8, ge=1, le=12)
    budget: int = Field(2400, ge=400, le=4000)
    response_format: str = Field("text", description="text | json")
    use_cache: bool = True


class ChatResponse(AnswerResponse):
    """多轮响应：额外给出改写后的检索 query、用到的历史与结构化产物。"""

    rewritten_query: str = ""
    history_used: bool = False
    structured: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    cache_size: int
    cache_max: int
    stage: str
    index_chunks: int = 0
    sessions: int = 0
    trace: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ 资源


def load_resources() -> Dict[str, Any]:
    cfg = load_config()
    candidate_k = int(cfg.get("rerank.candidate_k", 20))
    mode, retriever = make_retriever(cfg, True, candidate_k, verbose=True)
    try:
        generator: Any = load_generator(cfg=cfg, verbose=True)
    except Exception:
        generator = EchoGenerator()
    return {
        "cfg": cfg,
        "mode": mode,
        "retriever": retriever,
        "generator": generator,
        "candidate_k": candidate_k,
    }


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if key not in _CACHE:
        return None
    _CACHE.move_to_end(key)
    return _CACHE[key]


def _cache_put(key: str, value: Dict[str, Any]) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX:
        _CACHE.popitem(last=False)


# ------------------------------------------------------------------ 应用


def create_app(provider: Callable[[], Dict[str, Any]] = load_resources) -> FastAPI:
    app = FastAPI(
        title="中文临床指南问答服务",
        version="0.5.0",
        description="检索增强问答 + 逐句依据校验，供上层 Agent 或业务系统调用。",
    )
    state: Dict[str, Any] = {"resources": None}

    def resources() -> Dict[str, Any]:
        if state["resources"] is None:
            state["resources"] = provider()
        return state["resources"]

    def _auth(x_api_key: Optional[str]) -> None:
        expected = os.environ.get("SERVICE_API_KEY", "")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key")

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok", "stage": "W5"}

    @app.get("/v1/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        res = resources()
        chunks = len(getattr(getattr(res["retriever"], "index", None), "chunks", []) or [])
        return StatsResponse(
            cache_size=len(_CACHE),
            cache_max=CACHE_MAX,
            stage=str(res["cfg"].get("project.stage", "")),
            index_chunks=chunks,
            sessions=len(SESSIONS),
            trace=TRACER.summary(),
        )

    def _pipeline(
        trace: Any,
        res: Dict[str, Any],
        question: str,
        mode: str,
        top_n: int,
        budget: int,
        min_read: int = 0,
    ) -> Dict[str, Any]:
        """跑一遍问答链，返回上下文、答案与引用。各阶段都在 trace 里计时。"""
        retriever, generator = res["retriever"], res["generator"]
        contexts: List[str] = []
        titles: List[str] = []
        answer_text = ""
        used = 0
        citations: List[int] = []

        if mode == "agent":
            from medical_rag.agent import make_agent
            from run_agent import offline_policy

            use_api = isinstance(generator, DeepSeekGenerator)
            with TRACER.span(trace, "agent", steps=0):
                agent, _ = make_agent(
                    generator.chat if use_api else offline_policy,
                    retriever,
                    reranker=getattr(retriever, "reranker", None),
                    max_steps=8,
                    budget_chars=budget,
                    min_read=min_read,
                )
                run = agent.run(question)
                contexts, titles, answer_text = run.contexts, run.titles, run.answer
                used = run.used_chars
                trace.spans[-1].fields.update(
                    steps=len(run.steps), tool_calls=sum(run.tool_calls.values())
                )
                trace.fields["agent_steps"] = [
                    {"action": s.action or "parse_error", "thought": (s.thought or "")[:60]}
                    for s in run.steps
                ]
        else:
            with TRACER.span(trace, "retrieve", top_n=top_n):
                hits = retriever.retrieve(question, top_k=max(res["candidate_k"], top_n))
            contexts, titles, used = build_contexts(hits, budget, top_n)
            with TRACER.span(trace, "generate", chars=used):
                gen = generator.answer(question, contexts, titles, mode)
                answer_text = gen.answer
                citations = list(gen.citations)
        return {
            "contexts": contexts,
            "titles": titles,
            "answer": answer_text,
            "citations": citations,
            "used_chars": used,
        }

    def _structured(
        generator: Any,
        question: str,
        contexts: List[str],
        titles: List[str],
        answer_text: str,
        citations: List[int],
        used: int,
        unsupported: List[str],
        grounding: float,
        history: str = "",
    ) -> Optional[Dict[str, Any]]:
        """让模型按 schema 回答；模型不配合时退化为服务端组装，不丢回答。"""
        raw = None
        if hasattr(generator, "chat"):
            prompt = (
                f"{build_context_block(contexts, titles)}\n\n"
                + (f"{history}\n\n" if history else "")
                + f"问题：{question}\n\n{render_json_instruction(len(contexts))}"
            )
            try:
                raw = generator.chat([{"role": "user", "content": prompt}])
            except Exception:
                raw = None
        model = parse_structured_answer(
            raw or "",
            n_contexts=len(contexts),
            titles=titles,
            used_chars=used,
            grounding=grounding,
            unsupported=unsupported,
        )
        if model is None:  # 退化：用已有文本答案组装，字段仍然完整
            model = parse_structured_answer(
                json.dumps({"answer": answer_text, "citations": citations}, ensure_ascii=False),
                n_contexts=len(contexts),
                titles=titles,
                used_chars=used,
                grounding=grounding,
                unsupported=unsupported,
            )
        return model.model_dump() if model else None

    @app.post("/v1/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest, x_api_key: Optional[str] = Header(default=None)) -> AnswerResponse:
        _auth(x_api_key)
        started = time.time()
        request_id = uuid.uuid4().hex[:12]
        trace = TRACER.new_trace("answer", trace_id=request_id, mode=req.mode)
        cache_key = json.dumps(
            [req.question, req.mode, req.top_n, req.budget, req.min_read], ensure_ascii=False
        )
        if req.use_cache:
            hit = _cache_get(cache_key)
            if hit is not None:
                payload = dict(hit)
                payload["request_id"] = request_id
                payload["cached"] = True
                payload["latency_ms"] = int((time.time() - started) * 1000)
                _log("answer", request_id=request_id, cached=True)
                return AnswerResponse(**payload)

        res = resources()
        retriever = res["retriever"]
        try:
            out = _pipeline(trace, res, req.question, req.mode, req.top_n, req.budget, req.min_read)
        except Exception as exc:
            _log("answer_error", request_id=request_id, error=str(exc))
            TRACER.finish(trace)
            raise HTTPException(status_code=503, detail="模型或检索服务暂时不可用")
        contexts = out["contexts"]
        titles = out["titles"]
        answer_text = out["answer"]
        citations = out["citations"]
        used = out["used_chars"]

        with TRACER.span(trace, "verify", contexts=len(contexts)):
            result = verify_answer(answer_text, contexts, mode="flag")
        payload = {
            "request_id": request_id,
            "answer": answer_text,
            "refused": result.refused if hasattr(result, "refused") else False,
            "citations": citations,
            "titles": titles,
            "contexts": contexts,
            "verdicts": [
                Verdict(
                    sentence=v.text,
                    supported=v.supported,
                    coverage=round(v.coverage, 3),
                    unsupported_numbers=list(v.unsupported_numbers),
                ).model_dump()
                for v in result.verdicts
            ],
            "retrieval_mode": res["mode"],
            "generator": type(res["generator"]).__name__,
            "used_chars": used,
            "latency_ms": int((time.time() - started) * 1000),
            "cached": False,
        }
        if req.use_cache:
            _cache_put(cache_key, dict(payload))
        _log(
            "answer",
            request_id=request_id,
            mode=req.mode,
            used_chars=used,
            latency_ms=payload["latency_ms"],
            cached=False,
        )
        TRACER.finish(trace)
        return AnswerResponse(**payload)

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None)) -> ChatResponse:
        """多轮问答：先做指代消解，再走同一条链；可选按 schema 返回结构化答案。"""
        _auth(x_api_key)
        started = time.time()
        request_id = uuid.uuid4().hex[:12]
        trace = TRACER.new_trace("chat", trace_id=request_id, session=req.session_id)

        session = SESSIONS.get(req.session_id)
        with TRACER.span(trace, "session"):
            rewritten = session.rewrite(req.question)
            history = session.history_block()

        cache_key = json.dumps(
            [req.session_id, req.question, req.mode, req.top_n, req.budget, req.response_format],
            ensure_ascii=False,
        )
        if req.use_cache:
            hit = _cache_get(cache_key)
            if hit is not None:
                payload = dict(hit)
                payload["request_id"] = request_id
                payload["cached"] = True
                payload["latency_ms"] = int((time.time() - started) * 1000)
                _log("chat", request_id=request_id, cached=True)
                TRACER.finish(trace)
                return ChatResponse(**payload)

        res = resources()
        try:
            out = _pipeline(trace, res, rewritten, req.mode, req.top_n, req.budget)
        except Exception as exc:
            _log("chat_error", request_id=request_id, error=str(exc))
            TRACER.finish(trace)
            raise HTTPException(status_code=503, detail="模型或检索服务暂时不可用")

        with TRACER.span(trace, "verify", contexts=len(out["contexts"])):
            result = verify_answer(out["answer"], out["contexts"], mode="flag")
        claims = [v for v in result.verdicts if getattr(v, "is_claim", True)]
        grounding = (sum(1 for v in claims if v.supported) / len(claims)) if claims else 0.0

        structured = None
        if req.response_format == "json":
            with TRACER.span(trace, "structured"):
                structured = _structured(
                    res["generator"],
                    req.question,
                    out["contexts"],
                    out["titles"],
                    out["answer"],
                    out["citations"],
                    out["used_chars"],
                    list(result.unsupported_sentences),
                    grounding,
                    history=history,
                )

        session.add(
            req.question,
            out["answer"],
            titles=out["titles"],
            citations=out["citations"],
            contexts=len(out["contexts"]),
        )
        payload = {
            "request_id": request_id,
            "answer": out["answer"],
            "refused": result.refused if hasattr(result, "refused") else False,
            "citations": out["citations"],
            "titles": out["titles"],
            "contexts": out["contexts"],
            "verdicts": [
                Verdict(
                    sentence=v.text,
                    supported=v.supported,
                    coverage=round(v.coverage, 3),
                    unsupported_numbers=list(v.unsupported_numbers),
                ).model_dump()
                for v in result.verdicts
            ],
            "retrieval_mode": res["mode"],
            "generator": type(res["generator"]).__name__,
            "used_chars": out["used_chars"],
            "latency_ms": int((time.time() - started) * 1000),
            "cached": False,
            "rewritten_query": rewritten,
            "history_used": bool(history),
            "structured": structured,
        }
        if req.use_cache:
            _cache_put(cache_key, dict(payload))
        _log(
            "chat",
            request_id=request_id,
            session=req.session_id,
            rewritten=rewritten != req.question,
            latency_ms=payload["latency_ms"],
        )
        TRACER.finish(trace)
        return ChatResponse(**payload)

    return app


app = create_app()


def main() -> int:
    ap = argparse.ArgumentParser(description="启动问答服务")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
