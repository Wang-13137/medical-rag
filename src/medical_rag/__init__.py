# -*- coding: utf-8 -*-
"""中文医疗 RAG 系统的 W1 模块集合。

提供四个基础件：文档解析（ingest）、切片（chunking）、向量化（embedding）、
索引与检索（index / retrieve）。本阶段不含生成、重排与评测。
"""

__version__ = "0.1.0-w1"

__all__ = ["config", "ingest", "chunking", "embedding", "index", "retrieve"]
