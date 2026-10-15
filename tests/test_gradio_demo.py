"""Gradio 演示的离线测试：只测纯渲染函数与界面能否构建，不启动服务、不加载模型。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gradio_demo as gd  # noqa: E402


def _hit(text: str, doc: str = "指南A.txt", score: float = 0.8, head=None, before=None):
    hit = SimpleNamespace(
        chunk=SimpleNamespace(text=text, doc_name=doc, heading_path=head or ["概述"]),
        score=score,
    )
    if before is not None:
        hit.first_stage_rank = before
    return hit


def test_render_sources_numbers_matches_citations():
    md = gd.render_sources(["甲。", "乙。"], ["来源：指南A", "来源：指南B"])
    assert "[1]" in md and "[2]" in md
    assert "指南B" in md


def test_render_sources_empty_context():
    assert "没有进入上下文" in gd.render_sources([], [])


def test_render_sources_truncates_long_text():
    md = gd.render_sources(["丙" * 500], ["来源：指南A"])
    assert "…" in md
    assert len(md) < 500


def test_render_retrieval_table_shows_before_and_after_rank():
    rows = gd.render_retrieval_table([_hit("甲。", before=4), _hit("乙。", before=1)], top_n=8)
    assert rows[0][2] == 4
    assert rows[1][2] == 1
    assert rows[0][0] == 1  # 精排后名次重新编号


def test_render_retrieval_table_tolerates_missing_first_stage():
    rows = gd.render_retrieval_table([_hit("甲。")], top_n=8)
    assert rows[0][2] == "—"


def test_render_verdicts_flags_unsupported_number():
    contexts = ["推荐起始剂量为 5 mg，每日一次。"]
    rows = gd.render_verdicts("推荐起始剂量为 50 mg。", contexts)
    assert rows
    unsupported = [r for r in rows if "无依据" in r[1]]
    assert unsupported
    assert "50" in unsupported[0][3]


def test_render_verdicts_supported_sentence_passes():
    contexts = ["推荐起始剂量为 5 mg，每日一次。"]
    rows = gd.render_verdicts("推荐起始剂量为 5 mg。", contexts)
    assert rows and rows[0][1] == "有依据"


def test_render_verdicts_empty_answer():
    assert gd.render_verdicts("", ["甲。"]) == []


def test_answer_once_rejects_empty_question():
    answer, sources, *_ = gd.answer_once("  ", gd.MODE_PIPELINE_STRICT, 8, 2400, 5, True)
    assert "请输入问题" in answer
    assert sources == ""


def test_load_examples_contains_unanswerable():
    examples = gd.load_examples()
    assert examples, "benchmark_v1.json 应该存在且非空"
    # 样例里必须混入应拒答题，否则演示不出拒答行为
    bench = __import__("json").load(open(ROOT / "data/eval/benchmark_v1.json", encoding="utf-8"))
    unanswerable_q = {x["question"] for x in bench["items"] if x.get("type") == "unanswerable"}
    assert any(e[0] in unanswerable_q for e in examples)


def test_build_demo_constructs_blocks():
    pytest.importorskip("gradio")
    demo = gd.build_demo()
    assert demo is not None
    # 构建过程不应自动启动服务
    assert not hasattr(demo, "server_thread_live") or not demo.server_thread_live
