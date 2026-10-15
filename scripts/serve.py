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
from medical_rag.generate import DeepSeekGenerator, EchoGenerator, load_generator  # noqa: E402
from medical_rag.verify import verify_answer  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402
from run_generation import build_contexts  # noqa: E402

CACHE_MAX = int(os.environ.get("SERVICE_CACHE_MAX", "128"))
_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


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


class StatsResponse(BaseModel):
    cache_size: int
    cache_max: int
    stage: str
    index_chunks: int = 0


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
        )

    @app.post("/v1/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest, x_api_key: Optional[str] = Header(default=None)) -> AnswerResponse:
        _auth(x_api_key)
        started = time.time()
        request_id = uuid.uuid4().hex[:12]
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
        retriever, generator = res["retriever"], res["generator"]
        contexts: List[str] = []
        titles: List[str] = []
        answer_text = ""
        used = 0
        citations: List[int] = []

        try:
            if req.mode == "agent":
                from medical_rag.agent import make_agent
                from run_agent import offline_policy

                use_api = isinstance(generator, DeepSeekGenerator)
                agent, _ = make_agent(
                    generator.chat if use_api else offline_policy,
                    retriever,
                    reranker=getattr(retriever, "reranker", None),
                    max_steps=8,
                    budget_chars=req.budget,
                    min_read=req.min_read,
                )
                run = agent.run(req.question)
                contexts, titles, answer_text = run.contexts, run.titles, run.answer
                used = run.used_chars
            else:
                hits = retriever.retrieve(req.question, top_k=max(res["candidate_k"], req.top_n))
                contexts, titles, used = build_contexts(hits, req.budget, req.top_n)
                gen = generator.answer(req.question, contexts, titles, req.mode)
                answer_text = gen.answer
                citations = list(gen.citations)
        except Exception as exc:
            _log("answer_error", request_id=request_id, error=str(exc))
            raise HTTPException(status_code=503, detail="模型或检索服务暂时不可用")

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
            "generator": type(generator).__name__,
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
        return AnswerResponse(**payload)

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
