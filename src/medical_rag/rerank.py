# -*- coding: utf-8 -*-
"""二阶段检索的第二阶段：cross-encoder 重排。

为什么需要它（这一节是上一轮实验的直接结论）：
    BM25 + 向量的 RRF 混合在本语料上**反而变差**了（关键串全中率 0.500 → 0.400）。
    诊断下来原因是：中文临床指南里「基层医疗卫生机构」「推荐使用」这类短语在单份文档内部
    反复出现，BM25 会被这种高频短语主导，把「适用范围」「服务要求」这类章节顶到前排，
    RRF 又把它们的排名掺进最终结果，真正写着推荐条款的片段被挤出上下文预算。

    也就是说：**问题的本质是「同一份文档内部的段落判别」，不是「找对文档」**。
    稀疏的字面匹配解决不了这件事，需要的是把 query 和候选段拼在一起做一次交互式打分——
    也就是 cross-encoder（bge-reranker-base），它对「这段到底答没答这个问题」更敏感。

用法上是标准的两阶段：
    第一阶段用向量召回 candidate_k 条（宽进），第二阶段 cross-encoder 精排，返回 top_k 条。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .index import SearchHit


class CrossEncoderReranker:
    """bge-reranker-base 打分器，输入 (query, passage) 对，输出相关性 logits。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        local_model_dir: Optional[Path | str] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_model_dir = str(local_model_dir) if local_model_dir else None
        self._load(verbose)

    def _load(self, verbose: bool) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("重排需要 transformers 与 torch，请先安装") from exc

        self.torch = torch
        source = self.local_model_dir or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.eval()
        if self.device != "cpu":
            self.model.to(self.device)
        if verbose:
            print(f"[rerank] 模型已加载：{source}（device={self.device}）")

    # ------------------------------------------------------------ 打分

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """返回每个候选段落的相关性分数，越大越相关。"""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [(query, t) for t in texts[start : start + self.batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**encoded).logits.squeeze(-1)
            batch_scores = logits.float().numpy().reshape(-1)
            scores.extend(float(s) for s in batch_scores)
        return np.asarray(scores, dtype=np.float32)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "local_model_dir": self.local_model_dir,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }


class RerankRetriever:
    """第一阶段召回 + 第二阶段 cross-encoder 精排。"""

    def __init__(self, first_stage, reranker: CrossEncoderReranker, candidate_k: int = 20, top_k: int = 5):
        self.first_stage = first_stage
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.index = getattr(first_stage, "index", None)

    def _first_stage_hits(self, query: str, k: int) -> List[SearchHit]:
        hits = self.first_stage.retrieve(query, top_k=k)
        # 第一阶段可能返回不足 k 条，这里不再额外补。真实召回够宽就行。
        return list(hits)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchHit]:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        final_k = int(top_k or self.top_k)
        candidates = self._first_stage_hits(query, max(self.candidate_k, final_k))
        if not candidates:
            return []
        scores = self.reranker.score(query, [h.chunk.text for h in candidates])
        order = np.argsort(-scores)[:final_k]
        return [
            SearchHit(chunk=candidates[i].chunk, score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    def explain(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """输出每条结果的 cross-encoder 分数与它在第一阶段的原始排名。"""
        final_k = int(top_k or self.top_k)
        candidates = self._first_stage_hits(query, max(self.candidate_k, final_k))
        if not candidates:
            return []
        scores = self.reranker.score(query, [h.chunk.text for h in candidates])
        first_rank = {id(h.chunk): r for r, h in enumerate(candidates, start=1)}
        order = np.argsort(-scores)[:final_k]
        rows: List[Dict[str, Any]] = []
        for rank, i in enumerate(order, start=1):
            chunk = candidates[i].chunk
            rows.append(
                {
                    "rank": rank,
                    "rerank_score": float(scores[i]),
                    "first_stage_rank": first_rank.get(id(chunk)),
                    "doc_name": chunk.doc_name,
                    "text": chunk.text,
                }
            )
        return rows

    def format_hits(self, hits: Sequence[SearchHit], text_width: int = 200) -> str:
        lines: List[str] = []
        for hit in hits:
            preview = hit.chunk.text.replace("\n", " ")[:text_width]
            lines.append(f"[{hit.rank}] rerank={hit.score:.4f}  {hit.chunk.citation()}")
            lines.append(f"    {preview}")
        return "\n".join(lines)
