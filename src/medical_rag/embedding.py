# -*- coding: utf-8 -*-
"""向量化层：SentenceTransformer 加载 bge-small-zh-v1.5 并批量推理。

两条容易踩错的约定
------------------
1. bge 系列做检索时，query 侧要加指令前缀
   「为这个句子生成表示以用于检索相关文章：」，文档侧不加前缀。
   两侧处理不一致会导致召回质量明显下降。
2. 输出向量做 L2 归一化，下游 FAISS 用 IndexFlatIP，
   内积即余弦相似度，取值区间统一在 [-1, 1]。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# bge 官方推荐的检索指令；配置中的 embedding.query_instruction 会覆盖它
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder:
    """对 bge 系列模型的薄封装，区分 query 与 document 两种编码路径。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        max_seq_length: int = 512,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        model: Any = None,
        local_model_dir: str | Path | None = None,
        show_progress_bar: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.query_instruction = query_instruction
        self.show_progress_bar = show_progress_bar
        self.local_model_dir = str(local_model_dir) if local_model_dir else None
        # 实际加载路径（本地快照目录或 Hub 模型名），加载后写入
        self.resolved_model_dir: Optional[str] = None
        # 模型对象可由外部注入（测试场景），否则走懒加载
        self._model = model

    # ------------------------------------------------------------ 构造入口

    @classmethod
    def from_config(cls, cfg, verbose: bool = True) -> "BGEEmbedder":
        """从配置对象构造，并顺便把 torch 线程数 set 成配置的 runtime.num_threads。"""
        embedder = cls(
            model_name=str(cfg.get("embedding.model_name", "BAAI/bge-small-zh-v1.5")),
            device=str(cfg.get("embedding.device", "cpu")),
            batch_size=int(cfg.get("embedding.batch_size", 16)),
            normalize=bool(cfg.get("embedding.normalize", True)),
            max_seq_length=int(cfg.get("embedding.max_seq_length", 512)),
            query_instruction=str(cfg.get("embedding.query_instruction", DEFAULT_QUERY_INSTRUCTION)),
            local_model_dir=cfg.get("embedding.local_model_dir") or None,
            show_progress_bar=bool(cfg.get("runtime.show_progress_bar", False)),
        )
        embedder._limit_threads(int(cfg.get("runtime.num_threads", 0) or 0))
        model_dir = embedder.resolve_model_dir(hint_localok=True, verbose=verbose)
        embedder.load(model_dir, verbose=verbose)
        return embedder

    @staticmethod
    def _limit_threads(num_threads: int) -> None:
        """CPU 机器上限制线程数，避免 12 核全开把机器跑满。"""
        if num_threads <= 0:
            return
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
        try:
            import torch

            torch.set_num_threads(num_threads)
        except Exception:
            pass

    # ------------------------------------------------------------ 模型加载

    @property
    def model(self) -> Any:
        """懒加载：真正用到时才下载 / 读取模型权重。"""
        if self._model is None:
            model_dir = self.resolve_model_dir()
            self.load(model_dir)
        return self._model

    def resolve_model_dir(self, hint_localok: bool = False, verbose: bool = False) -> str:
        """确定模型加载路径，优先级：显式本地目录 > HF 缓存 > 联网下载 > ModelScope。

        返回一个可直接交给 SentenceTransformer 的路径（本地目录或 Hub 模型名）。
        """
        if self.local_model_dir and Path(self.local_model_dir).exists():
            if verbose:
                print(f"[embedding] 使用配置的本地模型目录：{self.local_model_dir}")
            return str(self.local_model_dir)

        # ModelScope 缓存优先：国内网络下通常由 modelscope 下载成功，
        # 命中后完全离线，无需再访问 huggingface.co
        try:
            from modelscope import snapshot_download as ms_download

            ms_path = ms_download(self.model_name, local_files_only=True)
            if verbose:
                print(f"[embedding] 命中 ModelScope 缓存：{ms_path}")
            return ms_path
        except Exception:
            pass

        local = self._try_offline_path()
        if local:
            if verbose:
                print(f"[embedding] 命中本地缓存：{local}")
            return local

        # 走联网下载：先按当前 HF_ENDPOINT（可能已设为 hf-mirror）
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id=self.model_name)
            if verbose:
                print(f"[embedding] 已从 HuggingFace 下载到：{path}")
            return path
        except Exception as exc:
            if verbose:
                print(f"[embedding] HuggingFace 下载失败（{type(exc).__name__}），改走 ModelScope")

        path = self._download_via_modelscope(verbose=verbose)
        return path

    def _try_offline_path(self) -> Optional[str]:
        """只读本地 Hub 缓存，不发起任何网络请求。"""
        try:
            from huggingface_hub import local_dir_exists

            if local_dir_exists(self.model_name):
                from huggingface_hub import snapshot_download

                return snapshot_download(repo_id=self.model_name, local_files_only=True)
        except Exception:
            return None
        return None

    def _download_via_modelscope(self, verbose: bool = False) -> str:
        """国内镜像兜底方案：从 ModelScope 拉取同名模型。"""
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace 下载失败，且未安装 modelscope。"
                "请执行 pip install modelscope 后重试，或手动下载模型并填写 "
                "embedding.local_model_dir。"
            ) from exc

        # ModelScope 上 BAAI 组织下的同名仓库
        ms_name = self.model_name if self.model_name.startswith("BAAI/") else f"BAAI/{self.model_name}"
        path = snapshot_download(ms_name)
        if verbose:
            print(f"[embedding] 已从 ModelScope 下载到：{path}")
        return path

    def load(self, model_dir: Optional[str] = None, verbose: bool = True) -> None:
        """加载 SentenceTransformer 并关掉 dropout 之类的训练期随机性。"""
        from sentence_transformers import SentenceTransformer

        target = model_dir or self.model_name
        self._model = SentenceTransformer(target, device=self.device)
        self._model.max_seq_length = self.max_seq_length
        self._model.eval()
        self.resolved_model_dir = str(target)
        if verbose:
            print(f"[embedding] 模型已加载：{target}（device={self.device}）")

    @property
    def dimension(self) -> int:
        """向量维度；bge-small-zh-v1.5 为 512。"""
        model = self.model
        # sentence-transformers 6.x 起改名为 get_embedding_dimension，兼容旧版本
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------ 编码

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        payload: List[str] = list(texts)
        if is_query and self.query_instruction:
            # 仅在 query 侧拼前缀；已带前缀的文本不重复添加
            payload = [
                text if text.startswith(self.query_instruction) else f"{self.query_instruction}{text}"
                for text in payload
            ]
        vectors = self.model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        # defensive: 少数情况下 batch_size=0 会返回 list
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """文档侧编码：不加任何前缀。"""
        return self._encode(texts, is_query=False)

    def encode_queries(self, texts: Sequence[str] | str) -> np.ndarray:
        """查询侧编码：自动补 query 指令前缀。"""
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts, is_query=True)

    # 兼容旧写法的别名
    encode = encode_documents

    def metadata(self) -> Dict[str, Any]:
        """落盘到索引元数据里的模型信息，便于后续排查"索引与模型不匹配"。"""
        return {
            "model_name": self.model_name,
            "resolved_model_dir": self.resolved_model_dir,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "query_instruction": self.query_instruction,
            "batch_size": self.batch_size,
            "device": self.device,
            "local_model_dir": self.local_model_dir,
            "resolved_model_dir": getattr(self._model, "_model_card_vars", {}) and None or None,
        }
