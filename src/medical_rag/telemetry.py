# -*- coding: utf-8 -*-
"""轻量可观测性：一次请求一条 trace，一条 trace 若干 span。

为什么不用现成的 APM：这里要观测的不是「接口慢不慢」，而是
**一次问答内部的时间去哪了**——检索多少毫秒、精排多少、生成多少、校验多少、
Agent 每一步调用了什么工具。这种粒度用通用 APM 打点很啰嗦，
而自己写只需要一个 request_id + 一串 span。

产物是一行 JSON（可打到 stdout，也可落 JSONL 文件），字段固定，方便直接 grep：
    {"trace_id": "...", "event": "answer", "spans": [...], "total_ms": 1234}
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    started: float
    ended: float = 0.0
    fields: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ms(self) -> float:
        end = self.ended or time.perf_counter()
        return round((end - self.started) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ms": self.ms,
            "fields": self.fields,
            **({"error": self.error} if self.error else {}),
        }


@dataclass
class Trace:
    trace_id: str
    event: str = "request"
    spans: List[Span] = field(default_factory=list)
    fields: Dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event": self.event,
            "spans": [s.to_dict() for s in self.spans],
            "fields": self.fields,
            "total_ms": self.total_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def span_ms(self, name: str) -> float:
        for s in self.spans:
            if s.name == name:
                return s.ms
        return 0.0


class Tracer:
    """收集 trace 并可选落盘 JSONL。不接外部服务，零配置即可用。"""

    def __init__(self, sink: Optional[Path] = None, echo: bool = False, keep: int = 200) -> None:
        self.sink = Path(sink) if sink else None
        self.echo = echo
        self.keep = keep
        self.traces: List[Trace] = []

    def new_trace(self, event: str = "request", trace_id: Optional[str] = None, **fields: Any) -> Trace:
        trace = Trace(trace_id or uuid.uuid4().hex[:12], event=event, fields=dict(fields))
        self.traces.append(trace)
        if len(self.traces) > self.keep:
            self.traces = self.traces[-self.keep :]
        return trace

    @contextmanager
    def span(self, trace: Trace, name: str, **fields: Any) -> Iterator[Span]:
        span = Span(name=name, started=time.perf_counter(), fields=dict(fields))
        trace.spans.append(span)
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.ended = time.perf_counter()

    def finish(self, trace: Trace) -> Dict[str, Any]:
        payload = trace.to_dict()
        if self.echo:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        if self.sink is not None:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def summary(self, event: Optional[str] = None) -> Dict[str, Any]:
        """把若干条 trace 聚合成一份统计，用于 /v1/stats 之类的观测接口。"""
        rows = [t for t in self.traces if event is None or t.event == event]
        if not rows:
            return {"count": 0}
        totals = [t.total_ms for t in rows]
        per_span: Dict[str, List[float]] = {}
        for t in rows:
            for s in t.spans:
                per_span.setdefault(s.name, []).append(s.ms)
        return {
            "count": len(rows),
            "total_ms_mean": round(sum(totals) / len(totals), 2),
            "total_ms_p95": round(sorted(totals)[max(int(len(totals) * 0.95) - 1, 0)], 2),
            "span_ms_mean": {k: round(sum(v) / len(v), 2) for k, v in per_span.items()},
        }


def tracer_from_env() -> Tracer:
    """环境变量驱动：TRACE_JSONL 落盘，TRACE_ECHO=1 打屏，都不设就只留内存。"""
    import os

    sink = os.environ.get("TRACE_JSONL", "")
    return Tracer(
        sink=Path(sink) if sink else None,
        echo=os.environ.get("TRACE_ECHO", "") == "1",
    )
