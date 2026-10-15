# -*- coding: utf-8 -*-
"""切片逻辑测试：切分数量、重叠正确性、来源信息保留、标题识别。"""

from __future__ import annotations

import pytest

from medical_rag.chunking import (
    build_sections,
    chunk_document,
    chunk_documents_from_config,
    detect_heading,
    split_sentences,
)
from medical_rag.ingest import Document, LineBlock


def _make_doc(lines, doc_name="样本.md", file_type="md", pages=None) -> Document:
    """构造一个内存里的 Document，避免测试依赖磁盘文件。"""
    blocks = [
        LineBlock(text=text, page=None if pages is None else pages[index])
        for index, text in enumerate(lines)
    ]
    return Document(
        doc_id="样本",
        doc_name=doc_name,
        source_path=f"D:/语料/{doc_name}",
        file_type=file_type,
        lines=blocks,
    )


# ---------------------------------------------------------------- 1. 长文本应被切成多个切片

def test_long_text_is_split_into_multiple_chunks():
    body = "血糖长期控制不佳会显著增加微血管与大血管并发症的发生风险。" * 20  # 约 560 字
    doc = _make_doc([body])

    # 字符滑窗路径（sentence_aware=False）
    chunks = chunk_document(
        doc, chunk_size=200, chunk_overlap=50, prepend_heading_path=False, sentence_aware=False
    )

    assert len(chunks) > 1, "长文本应被切成多个切片"
    assert all(len(chunk.text) <= 200 for chunk in chunks), "单个切片长度不得超过 chunk_size"
    # 内容完整：首尾应与原文一致，说明没有丢字
    assert chunks[0].text.startswith(body[:20])
    assert chunks[-1].text.endswith(body[-20:])


# ---------------------------------------------------------------- 2. 相邻切片重叠正确

def test_chunk_overlap_exact():
    body = "".join(f"这是第{index}句用于控制文本长度的说明性句子。" for index in range(60))
    assert len(body) > 600, f"构造文本过短：{len(body)}"

    size, overlap = 200, 50
    doc = _make_doc([body])
    chunks = chunk_document(
        doc,
        chunk_size=size,
        chunk_overlap=overlap,
        prepend_heading_path=False,
        sentence_aware=False,
    )
    assert len(chunks) > 2

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-overlap:] == current.text[:overlap], "相邻切片的重叠区域必须逐字符一致"
        assert current.char_start - previous.char_start == size - overlap, "窗口步长应等于 chunk_size - overlap"

    for chunk in chunks:
        assert chunk.char_end - chunk.char_start == len(chunk.text), "字符区间应与切片文本长度吻合"


# ---------------------------------------------------------------- 3. 来源信息与标题层级保留

def test_heading_and_source_are_preserved():
    body = "长期血压管理需结合家庭自测血压与动态血压监测结果综合判断。" * 12
    lines = ["# 心血管疾病", "## 降压治疗的目标值", body, "### 合并糖尿病者", "目标值应个体化设定。"]
    doc = _make_doc(lines)

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=50)

    assert chunks, "应产出至少一个切片"
    for chunk in chunks:
        assert chunk.doc_name == "样本.md"
        assert chunk.source_path == "D:/语料/样本.md"
        assert len(chunk.chunk_id) == 16, "chunk_id 应为稳定的 16 位摘要"
        assert chunk.heading_path, "标题层级路径不能丢失"

    target = [chunk for chunk in chunks if chunk.heading_path[-1] == "降压治疗的目标值"]
    assert target, "应能按标题找回所属小节"
    assert "心血管疾病" in target[0].heading_path, "应保留一级标题作为祖先链"
    assert target[0].text.startswith("心血管疾病 > 降压治疗的目标值"), "标题路径应写进切片正文"
    assert target[0].citation() == "样本.md md · 心血管疾病 > 降压治疗的目标值"

    child = [chunk for chunk in chunks if chunk.heading_path[-1] == "合并糖尿病者"]
    assert len(child) == 1, "短小节应独立成片"
    assert child[0].text.endswith("目标值应个体化设定。")


def test_pdf_page_number_is_kept():
    doc = _make_doc(
        ["第一章 总论", "第一句内容。", "第二句内容。", "第三句内容。"],
        doc_name="指南.pdf",
        file_type="pdf",
        pages=[1, 1, 2, 2],
    )
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=80)
    assert chunks
    assert all(chunk.page == 1 for chunk in chunks), "切片应记住正文起始页码"
    assert chunks[0].file_type == "pdf"
    assert "P1" in chunks[0].citation(), "引用串应带页码"


# ---------------------------------------------------------------- 4. 短切片不拆分

def test_short_section_keeps_single_chunk():
    doc = _make_doc(["## 简要说明", "这是短短一句话。"])
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=80, prepend_heading_path=False)
    assert len(chunks) == 1
    assert chunks[0].text == "这是短短一句话。"


# ---------------------------------------------------------------- 5. 标题识别规则

def test_detect_heading_rules():
    assert detect_heading("# 用药安全")[1] == "用药安全"
    assert detect_heading("## 2型糖尿病的诊断标准")[1] == "2型糖尿病的诊断标准"
    assert detect_heading("第三章 检验科质量")[1] == "检验科质量"
    assert detect_heading("一、基本概念")[1] == "基本概念"
    assert detect_heading("（二）血糖监测")[1] == "血糖监测"
    assert detect_heading("2.1 血糖控制目标")[1] == "血糖控制目标"
    assert detect_heading("患者应定期复查，这是长期随访的重要一环。") is None, "以句号收尾的不是标题"
    assert detect_heading("短" * 61) is None, "超长行不是标题"


def test_build_sections_groups_body_under_heading():
    doc = _make_doc(["# 指南", "前言一句话。", "## 血压测量", "方法一。", "方法二。"])
    sections = build_sections(doc)
    bodies = [section.body for section in sections]
    assert "前言一句话。" in bodies
    assert "方法一。\n方法二。" in bodies
    assert sections[-1].heading_path == ["指南", "血压测量"]


# ---------------------------------------------------------------- 6. 配置驱动与参数校验

def test_chunk_documents_from_config_uses_yaml(config):
    """默认配置走句边界对齐：切点落在句末，重叠按整句回退。"""
    body = "".join(f"第{index}条推荐意见内容用于占位。" for index in range(40))
    doc = _make_doc(["## 概述", body])
    chunks = chunk_documents_from_config([doc], config)

    size = int(config.get("chunking.chunk_size"))
    heading_budget = 60  # 标题前缀占用的字符余量

    assert len(chunks) > 1
    assert all(len(chunk.text) <= size + heading_budget for chunk in chunks)
    # 每个切片都以完整句子收尾，不会从句子中间断开
    assert all(chunk.text.endswith("。") for chunk in chunks)


def test_config_can_fall_back_to_char_window(root_dir):
    """关掉 sentence_aware 后应退回固定步长的字符滑窗。"""
    from medical_rag.config import load_config

    cfg = load_config(root_dir / "configs" / "config.yaml", {"chunking.sentence_aware": False})
    doc = _make_doc(["## 概述", "甲" * 1000])

    chunks = chunk_documents_from_config([doc], cfg)
    size = int(cfg.get("chunking.chunk_size"))
    overlap = int(cfg.get("chunking.chunk_overlap"))
    assert chunks[1].char_start - chunks[0].char_start == size - overlap


def test_invalid_overlap_raises():
    doc = _make_doc(["甲" * 500])
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------- 7. 句边界对齐切片

def test_split_sentences_keeps_trailing_punctuation():
    text = "首选生活方式干预。若效果不佳，加用二甲双胍。"
    spans = split_sentences(text)
    assert [text[start:end] for start, end in spans] == [
        "首选生活方式干预。",
        "若效果不佳，加用二甲双胍。",
    ]


def test_split_sentences_ignores_decimal_point():
    """剂量里的小数点不能当句末，否则 "0.5 g" 会被剖开。"""
    text = "起始剂量为 0.5 g，每日两次。根据血糖调整。"
    spans = split_sentences(text)
    assert len(spans) == 2
    assert text[spans[0][0] : spans[0][1]] == "起始剂量为 0.5 g，每日两次。"


def test_sentence_aware_never_cuts_inside_sentence():
    """句对齐的核心保证：任何切片都以句末收尾，长推荐句整句成片。"""
    long_recommendation = (
        "对于幽门螺杆菌根除治疗，推荐使用铋剂四联方案，"
        "标准剂量质子泵抑制剂联合两种抗生素与铋剂，疗程为 14 d。"
    )
    filler = "其余内容用于把文本撑到需要切分的程度。" * 6
    body = filler + long_recommendation
    doc = _make_doc([body])

    chunks = chunk_document(
        doc, chunk_size=150, chunk_overlap=30, prepend_heading_path=False, sentence_aware=True
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith("。"), f"切片应以句末收尾，实际：{chunk.text[-20:]!r}"

    # 关键串必须在某一个切片里完整出现，而不是跨切片拼起来才完整
    joined = [chunk.text for chunk in chunks if "疗程为 14 d" in chunk.text]
    assert joined, "完整推荐句应在单个切片内可检索到"
    assert all("推荐使用铋剂四联方案" in text for text in joined)


def test_sentence_aware_makes_progress_and_covers_text():
    """句窗口必须单调前进且首尾不丢字（防止回退逻辑写错导致死循环或漏字）。"""
    body = "".join(f"第{index}条推荐意见，内容用于占位。" for index in range(80))
    doc = _make_doc([body])
    chunks = chunk_document(
        doc, chunk_size=200, chunk_overlap=60, prepend_heading_path=False, sentence_aware=True
    )
    assert len(chunks) > 3
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(body)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.char_start > previous.char_start, "窗口必须单调前进"
        assert current.char_start < previous.char_end, "相邻窗口应保持重叠"


def test_extremely_long_sentence_is_capped_by_hard_max():
    """没有句末标点的超长段落不能无限膨胀，需按 hard_max_chars 兜底切开。"""
    doc = _make_doc(["甲" * 2000])
    chunks = chunk_document(
        doc,
        chunk_size=300,
        chunk_overlap=60,
        prepend_heading_path=False,
        sentence_aware=True,
        hard_max_chars=800,
    )
    assert len(chunks) > 1
    assert max(len(chunk.text) for chunk in chunks) <= 800
