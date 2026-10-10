# -*- coding: utf-8 -*-
"""pytest 公共夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.embedding import BGEEmbedder  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT_DIR


@pytest.fixture(scope="session")
def config():
    """默认配置对象。"""
    return load_config(ROOT_DIR / "configs" / "config.yaml")


@pytest.fixture(scope="session")
def embedder(config):
    """跨用例复用的向量化器，避免重复加载模型（bge-small 约 0.4 秒，重复加载浪费显存无关时间）。"""
    return BGEEmbedder.from_config(config, verbose=False)
