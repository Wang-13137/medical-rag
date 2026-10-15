# -*- coding: utf-8 -*-
"""可观测性模块的测试：trace / span / 落盘 / 聚合。"""

from __future__ import annotations

import json
import time

from medical_rag.telemetry import Span, Trace, Tracer


def test_span_measures_elapsed_time():
    span = Span(name="retrieve", started=time.perf_counter())
    time.sleep(0.01)
    span.ended = time.perf_counter()
    assert span.ms >= 10


def test_span_records_error_and_reraises():
    tracer = Tracer()
    trace = tracer.new_trace("t")
    try:
        with tracer.span(trace, "generate") as span:
            raise RuntimeError("模型超时")
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("异常应当被重新抛出")
    assert "RuntimeError" in trace.spans[0].error


def test_span_keeps_fields():
    tracer = Tracer()
    trace = tracer.new_trace("t")
    with tracer.span(trace, "retrieve", top_k=8) as span:
        span.fields["hits"] = 5
    assert trace.spans[0].fields["top_k"] == 8
    assert trace.spans[0].fields["hits"] == 5


def test_trace_serializes_to_json():
    tracer = Tracer()
    trace = tracer.new_trace("answer", trace_id="abc123", mode="strict")
    with tracer.span(trace, "retrieve"):
        pass
    payload = json.loads(trace.to_json())
    assert payload["trace_id"] == "abc123"
    assert payload["event"] == "answer"
    assert payload["spans"][0]["name"] == "retrieve"
    assert payload["total_ms"] >= 0


def test_trace_span_ms_lookup():
    trace = Trace(trace_id="x")
    trace.spans.append(Span(name="verify", started=time.perf_counter(), ended=time.perf_counter()))
    assert trace.span_ms("verify") >= 0
    assert trace.span_ms("missing") == 0.0


def test_tracer_finish_writes_jsonl(tmp_path):
    sink = tmp_path / "traces.jsonl"
    tracer = Tracer(sink=sink)
    trace = tracer.new_trace("chat", session="s1")
    with tracer.span(trace, "session"):
        pass
    tracer.finish(trace)
    rows = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["fields"]["session"] == "s1"


def test_tracer_keeps_only_recent_traces():
    tracer = Tracer(keep=3)
    for _ in range(10):
        tracer.new_trace("answer")
    assert len(tracer.traces) == 3


def test_summary_aggregates_spans():
    tracer = Tracer()
    for _ in range(3):
        trace = tracer.new_trace("answer")
        with tracer.span(trace, "retrieve"):
            pass
        with tracer.span(trace, "generate"):
            pass
    summary = tracer.summary()
    assert summary["count"] == 3
    assert set(summary["span_ms_mean"]) == {"retrieve", "generate"}
    assert tracer.summary("nothing")["count"] == 0


def test_summary_empty_tracer():
    assert Tracer().summary()["count"] == 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])
