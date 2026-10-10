# -*- coding: utf-8 -*-
"""向量化测试：维度、bge query 指令前缀、一致性与批量推理。"""

from __future__ import annotations

import numpy as np

from medical_rag.embedding import DEFAULT_QUERY_INSTRUCTION

EXPECTED_DIM = 512  # bge-small-zh-v1.5 的输出维度


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


# ---------------------------------------------------------------- 1. 维度与归一化

def test_embedding_dimension_is_512(embedder):
    vectors = embedder.encode_documents(["2型糖尿病的诊断标准"])

    assert embedder.dimension == EXPECTED_DIM, "bge-small-zh-v1.5 应为 512 维"
    assert vectors.shape == (1, EXPECTED_DIM)
    assert vectors.dtype == np.float32
    assert abs(np.linalg.norm(vectors[0]) - 1.0) < 1e-4, "开启 normalize 后向量应为单位向量"


# ---------------------------------------------------------------- 2. query 前缀处理

def test_query_instruction_only_applied_on_query_side(embedder):
    text = "2型糖尿病血糖控制目标"

    query_vec = embedder.encode_queries([text])[0]
    doc_vec = embedder.encode_documents([text])[0]
    prefixed_doc_vec = embedder.encode_documents([embedder.query_instruction + text])[0]

    assert embedder.query_instruction == DEFAULT_QUERY_INSTRUCTION
    assert embedder.query_instruction, "必须配置 query 指令前缀"

    # query 侧编码结果应与「手动加前缀再做文档侧编码」完全一致
    assert np.allclose(query_vec, prefixed_doc_vec, atol=1e-5), "query 侧应自动拼上指令前缀"
    # 且与「不加前缀」明显不同，说明前缀确实生效
    assert not np.allclose(query_vec, doc_vec, atol=1e-3)
    assert 0.5 < _cosine(query_vec, doc_vec) < 0.999

    # 重复前缀不应被叠加一次以上
    double_prefixed = embedder.encode_queries([embedder.query_instruction + text])[0]
    assert np.allclose(double_prefixed, query_vec, atol=1e-5)


# ---------------------------------------------------------------- 3. 相同文本向量一致

def test_same_text_produces_identical_vectors(embedder):
    text = "华法林的抗凝监测指标是国际标准化比值 INR"

    repeated = embedder.encode_documents([text, text, text])
    assert np.allclose(repeated[0], repeated[1], atol=1e-6)
    assert np.allclose(repeated[1], repeated[2], atol=1e-6)

    queries = embedder.encode_queries([text, text])
    assert np.allclose(queries[0], queries[1], atol=1e-6)


# ---------------------------------------------------------------- 4. 批量推理

def test_batch_encoding_shapes_and_order(embedder):
    texts = ["他汀类药物的使用注意事项", "社区获得性肺炎的诊断要点", "血清电解质的参考区间"]

    vectors = embedder.encode_documents(texts)
    assert vectors.shape == (3, EXPECTED_DIM)

    batch_size = int(embedder.batch_size)
    assert batch_size >= 1

    # 批内顺序必须与输入一致：用文本自身做近邻检索应当命中自己
    for index, text in enumerate(texts):
        scores = vectors @ embedder.encode_documents([text])[0]
        assert int(np.argmax(scores)) == index, "批量输出顺序应与输入一致"


# ---------------------------------------------------------------- 5. 语义区分度

def test_semantic_separation(embedder):
    anchor = embedder.encode_documents(["慢性肾脏病的分期依据估算肾小球滤过率"])[0]
    related = embedder.encode_documents(["肾功能评估常用 eGFR 作为分期指标"])[0]
    unrelated = embedder.encode_documents(["儿童预防接种的时间安排"])[0]

    assert _cosine(anchor, related) > _cosine(anchor, unrelated), "语义相近文本应更相似"


# ---------------------------------------------------------------- 6. 配置一致性

def test_embedder_reads_from_config(config, embedder):
    assert embedder.model_name == str(config.get("embedding.model_name"))
    assert embedder.query_instruction == str(config.get("embedding.query_instruction"))
    assert embedder.batch_size == int(config.get("embedding.batch_size"))
    assert embedder.resolved_model_dir, "应记录实际加载的模型路径，便于离线复现"


def test_empty_batch_returns_empty_result(embedder):
    """空批次不应崩溃。sentence-transformers 返回一维空数组，行数可作为判断依据。"""
    vectors = embedder.encode_documents([])
    assert vectors.shape == (0,)
    assert vectors.dtype == np.float32
