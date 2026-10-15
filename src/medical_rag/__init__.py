# -*- coding: utf-8 -*-
"""中文医疗 RAG 系统（W1–W5）的模块集合。

按出场顺序分层：
- 基础件：文档解析（ingest）、切片（chunking）、向量化（embedding）、索引与检索（index / retrieve）
- W2：BM25 混合检索（hybrid）、cross-encoder 精排（rerank）
- W3：受限提示词的答案合成与引用标注（generate）
- W4：逐句蕴含校验闸门（verify）
- W5：手写 ReAct 智能体层（agent）

参数集中在 configs/config.yaml，评测产物集中在 reports/。
"""

__version__ = "0.5.0-w5"

__all__ = [
    "config",
    "ingest",
    "chunking",
    "embedding",
    "index",
    "retrieve",
    "hybrid",
    "rerank",
    "generate",
    "verify",
    "agent",
]
