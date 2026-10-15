# -*- coding: utf-8 -*-
"""逐句蕴含校验测试：数字溯源、覆盖率、非事实句豁免、三种闸门动作。"""

from __future__ import annotations

import pytest

from medical_rag.verify import (
    DEFAULT_COVERAGE_THRESHOLD,
    REFUSAL_TEXT,
    coverage_score,
    extract_numbers,
    is_claim_sentence,
    split_answer_sentences,
    verify_answer,
    verify_sentence,
)

CONTEXT = [
    "推荐使用铋剂四联方案，疗程为 14 d。根除治疗应在停药 4 周后复查。",
    "对于青霉素过敏者，可选用含左氧氟沙星的方案。",
]


# ---------------------------------------------------------------- 1. 句子切分与数字抽取

def test_split_answer_sentences_keeps_citation_with_sentence():
    answer = "推荐疗程为 14 d[1]。停药 4 周后复查[1]。"
    assert split_answer_sentences(answer) == ["推荐疗程为 14 d[1]。", "停药 4 周后复查[1]。"]


def test_extract_numbers_drops_citation_and_list_index():
    assert extract_numbers("1. 疗程 14 d[1]（2）复查") == ["14"]


# ---------------------------------------------------------------- 2. 非事实句豁免

def test_is_claim_sentence_rejects_boilerplate():
    assert not is_claim_sentence("根据提供的指南片段无法回答该问题。")
    assert not is_claim_sentence("根据提供的指南片段：")
    assert not is_claim_sentence("疗程为 14 d。")  # 少于 12 字，不构成独立事实陈述
    assert is_claim_sentence("推荐使用铋剂四联方案，疗程为 14 d[1]。")


def test_boilerplate_sentence_is_never_flagged_unsupported():
    """引导语在语料里必然找不到依据，不能因此判它编造。"""
    verdict = verify_sentence("根据提供的指南片段无法回答该问题。", CONTEXT)
    assert verdict.is_claim is False
    assert verdict.supported is True


# ---------------------------------------------------------------- 3. 数字溯源

def test_supported_number_passes():
    verdict = verify_sentence("推荐使用铋剂四联方案，疗程为 14 d[1]。", CONTEXT)
    assert verdict.unsupported_numbers == []
    assert verdict.supported is True


def test_unsupported_number_is_caught():
    """把疗程改成 30 d：上下文里没有这个数，必须被抓出来。"""
    verdict = verify_sentence("推荐使用铋剂四联方案，疗程为 30 d[1]。", CONTEXT)
    assert verdict.unsupported_numbers == ["30"]
    assert verdict.supported is False


def test_number_in_other_chunk_does_not_trigger_weak_citation():
    """数字在别的片段里找到，说明引用标得不准，但不等于编造。"""
    sentence = "对于青霉素过敏者，可选用含左氧氟沙星的方案[1]。"
    verdict = verify_sentence(sentence, CONTEXT)
    assert verdict.supported is True, "内容在上下文里有依据，不应判为编造"
    assert verdict.weak_citation is True, "但引用编号指向了不含该内容的片段"


# ---------------------------------------------------------------- 4. 覆盖率

def test_coverage_is_one_for_verbatim_sentence():
    sentence = "推荐使用铋剂四联方案，疗程为 14 d。"
    assert coverage_score(sentence, CONTEXT[0]) > 0.9


def test_coverage_drops_for_unrelated_sentence():
    sentence = "该方案对妊娠期高血压患者的降压目标应控制在 130/80 mmHg 以下。"
    assert coverage_score(sentence, "\n".join(CONTEXT)) < DEFAULT_COVERAGE_THRESHOLD


# ---------------------------------------------------------------- 5. 三种闸门动作

def test_flag_mode_does_not_modify_answer():
    answer = "推荐使用铋剂四联方案，疗程为 14 d[1]。另外疗程也可以是 90 d[1]。"
    result = verify_answer(answer, CONTEXT, mode="flag")
    assert result.action == "flagged"
    assert result.revised_answer == answer
    assert result.refused is False


def test_prune_mode_drops_unsupported_sentence():
    answer = "推荐使用铋剂四联方案，疗程为 14 d[1]。推荐使用铋剂四联方案，疗程为 90 d[1]。"
    result = verify_answer(answer, CONTEXT, mode="prune")
    assert result.action == "pruned"
    assert "90 d" not in result.revised_answer
    assert "14 d" in result.revised_answer


def test_refuse_mode_replaces_whole_answer():
    answer = "推荐使用铋剂四联方案，疗程为 90 d[1]。"
    result = verify_answer(answer, CONTEXT, mode="refuse")
    assert result.action == "refused"
    assert result.refused is True
    assert result.revised_answer == REFUSAL_TEXT


def test_prune_falls_back_to_refusal_when_little_remains():
    """删掉无依据句后剩不下东西时，应整条拒答而不是输出残缺答案。"""
    answer = "推荐使用铋剂四联方案疗程为 90 d[1]。停药后可立即复查幽门螺杆菌[1]。"
    result = verify_answer(answer, CONTEXT, mode="prune", min_keep_ratio=0.9)
    assert result.action == "refused"
    assert result.revised_answer == REFUSAL_TEXT


def test_refusal_answer_passes_untouched():
    """答案本身就是拒答时，闸门不该再把它当成编造来处理。"""
    result = verify_answer(REFUSAL_TEXT, CONTEXT, mode="refuse")
    assert result.action == "pass"
    assert result.revised_answer == REFUSAL_TEXT


def test_clean_answer_passes():
    answer = "推荐使用铋剂四联方案，疗程为 14 d[1]。根除治疗应在停药 4 周后复查[1]。"
    result = verify_answer(answer, CONTEXT, mode="prune")
    assert result.action == "pass"
    assert result.unsupported_sentences == []


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        verify_answer("推荐使用铋剂四联方案，疗程为 14 d[1]。", CONTEXT, mode="whatever")
