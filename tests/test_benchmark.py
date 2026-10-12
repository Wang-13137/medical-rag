"""评测集本身的单元测试：确保题库不会被悄悄改动成「凭空写的答案」。

这里不依赖 2.2GB 的重排模型，只做静态校验，保证能在 CI 里跑。
"""

from __future__ import annotations

import json
import sys
import unicodedata
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH = ROOT / "data" / "eval" / "benchmark_v1.json"
SOURCE = ROOT / "data" / "eval" / "queries_v1.yaml"
CORPUS = ROOT / "data" / "clean"


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", folded)


@pytest.fixture(scope="module")
def benchmark():
    if not BENCH.exists():
        pytest.skip("benchmark_v1.json 尚未生成，先运行 scripts/build_benchmark.py")
    return json.loads(BENCH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists() or not list(CORPUS.glob("*.txt")):
        pytest.skip("data/clean 语料不存在")
    return {
        p.name: normalize(p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(CORPUS.glob("*.txt"))
    }


def test_source_valid_yaml():
    """标注源文件必须能解析，否则整条构建链路失效。"""
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    assert "items" in data and data["items"]


def test_ids_unique_and_total(benchmark):
    ids = [item["id"] for item in benchmark["items"]]
    assert len(ids) == len(set(ids)), "存在重复题号"
    assert len(ids) == benchmark["metadata"]["total_items"]


def test_answerable_has_gold_doc_and_keys(benchmark):
    for item in benchmark["items"]:
        if item["type"] == "unanswerable":
            continue
        assert item.get("gold_doc"), f"{item['id']} 缺少 gold_doc"
        assert item.get("answer_keys"), f"{item['id']} 缺少答案关键串"
        assert item.get("gold_answer"), f"{item['id']} 缺少人读标准答案"


def test_every_answer_key_exists_in_gold_doc(benchmark, corpus):
    """核心断言：每个关键串都必须能在 gold_doc 原文里逐字搜到。"""
    failures = []
    for item in benchmark["items"]:
        if item["type"] == "unanswerable":
            continue
        norm = corpus.get(item["gold_doc"])
        if norm is None:
            failures.append(f"{item['id']}: gold_doc 不在语料中")
            continue
        for key in item["answer_keys"]:
            if normalize(key) not in norm:
                failures.append(f"{item['id']}: 关键串未在原文命中 -> {key}")
    assert not failures, "\n".join(failures)


def test_refusal_terms_absent_from_corpus(benchmark, corpus):
    """应拒答题的主题词必须在语料里零出现，否则这条「无依据」的声称不成立。"""
    bad = []
    for item in benchmark["items"]:
        if item["type"] != "unanswerable":
            continue
        for term in item.get("probe_terms") or []:
            hits = sum(text.count(normalize(term)) for text in corpus.values())
            if hits:
                bad.append(f"{item['id']}: 主题词「{term}」在语料出现 {hits} 次")
    assert not bad, "\n".join(bad)


def test_refusals_have_reason(benchmark):
    for item in benchmark["items"]:
        if item["type"] == "unanswerable":
            assert item.get("refuse_reason"), f"{item['id']} 需要写清拒答理由"


def test_type_values_are_known(benchmark):
    allowed = {"fact", "numeric", "disambiguation", "unanswerable"}
    for item in benchmark["items"]:
        assert item["type"] in allowed, f"{item['id']} 题型未定义：{item['type']}"


def test_each_corpus_doc_is_asked_about(benchmark, corpus):
    """每份入库指南都至少有一道题，避免语料里躺着没人测的文档。"""
    covered = {item["gold_doc"] for item in benchmark["items"] if item.get("gold_doc")}
    missing = sorted(set(corpus) - covered)
    assert not missing, f"以下文档没有任何评测题：{missing}"
