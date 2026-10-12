"""生成端与评测指标的单元测试。

刻意不使用真实模型与网络：生成用 EchoGenerator 替身，重排等模块不出现在本测试里。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.generate import (  # noqa: E402
    EchoGenerator,
    build_context_block,
    detect_refusal,
    parse_citations,
)


def test_parse_citations_dedup_and_order():
    text = "血压目标 <130/80 mmHg [1]，合并糖尿病者更低 [2][1]。"
    assert parse_citations(text) == [1, 2]


def test_parse_citations_respects_max_index():
    assert parse_citations("[1][5][2]", max_index=3) == [1, 2]


def test_detect_refusal():
    assert detect_refusal("根据提供的指南片段无法回答该问题。缺少相关内容。")
    assert detect_refusal("  无法回答 ")
    assert not detect_refusal("推荐首选手术治疗 [1]。")


def test_build_context_block_numbering():
    block = build_context_block(["甲", "乙"], titles=["来源：A", "来源：B"])
    assert block.startswith("[1] 来源：A\n甲")
    assert "[2] 来源：B\n乙" in block


def test_build_context_block_without_titles():
    block = build_context_block(["甲"])
    assert block == "[1]\n甲"


def test_echo_generator_offline_and_cites_first_context():
    gen = EchoGenerator()
    res = gen.answer("什么？", ["上下文一", "上下文二"], prompt_mode="strict")
    assert res.model == "echo-offline"
    assert res.citations == [1]
    assert "上下文一" in res.answer
    assert not res.refused


def test_echo_generator_refuses_when_no_context():
    gen = EchoGenerator()
    res = gen.answer("什么？", [], prompt_mode="strict")
    assert res.refused
    assert res.citations == []
    assert res.error is None


# ---- 评测指标相关：直接测 scripts/eval_generation.py 的纯函数 ----

from eval_generation import citation_stats, extract_numbers, normalize  # noqa: E402


def test_normalize_strips_width_and_space():
    assert normalize("１ ４ ０ ｍｍＨｇ") == normalize("140mmHg")


def test_extract_numbers_drops_citations_and_list_markers():
    nums = extract_numbers("1. 血压<140/90 mmHg [2]\n2. 血糖7.0 mmol/L")
    # 引用编号 [2] 与列表序号 1./2. 都不应被当成事实数字
    assert nums == ["140", "90", "7.0"]
    assert nums.count("90") == 1
    assert nums.count("7.0") == 1
    assert nums.count("1") == 0
    assert nums.count("2") == 0


def test_extract_numbers_parenthesized_index():
    assert extract_numbers("（1）首选 [3] 140/90") == ["140", "90"]


def test_citation_stats_supported_and_dangling():
    rec = {
        "citations": [1, 3, 9],
        "contexts": ["目标值130/80 mmHg", "无关片段", "另一个支持片段30/80"],
        "answer_keys": ["130/80 mmHg"],
    }
    st = citation_stats(rec)
    # 引用的 3 条里有 1 条含关键串；编号 9 超出上下文范围，计入悬空引用
    assert st["citation_rate"] == 1.0
    assert st["citation_precision"] == pytest.approx(0.5)
    assert st["citation_dangling"] == 1


def test_citation_stats_no_citation():
    rec = {"citations": [], "contexts": ["有内容"], "answer_keys": ["x"]}
    assert citation_stats(rec)["citation_rate"] == 0.0


def test_citation_stats_unanswerable_has_no_keys():
    rec = {"citations": [1], "contexts": ["有内容"], "answer_keys": []}
    st = citation_stats(rec)
    assert st["citation_rate"] == 1.0
    assert st["citation_precision"] == 0.0
