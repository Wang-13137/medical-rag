# -*- coding: utf-8 -*-
"""结构化输出：让模型按 schema 回答，而不是吐一段无法解析的自由文本。

为什么值得做：RAG 的下游是程序，不是人。自由文本答案要被引用解析、被逐句校验、
被前端渲染、被 Agent 消费，每一步都在猜。约定 schema 之后，
「答案 - 引用 - 依据」三者的对应关系是显式的，幻觉检测也能直接落在字段上。

字段设计上刻意**不包含模型自报的置信度**：自报分数没有校准过，
拿来当闸门只会制造错觉。有依据句子的比例由校验器算（grounding），不是模型说的。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_OBJ_RE = re.compile(r"\{.*\}", re.S)


class StructuredAnswer(BaseModel):
    """一次回答的结构化产物。"""

    answer: str = Field(..., description="最终答案正文，可含 [n] 引用编号")
    citations: List[int] = Field(default_factory=list, description="引用了哪些片段，编号从 1 开始")
    refused: bool = Field(False, description="是否因证据不足而拒答")
    unsupported: List[str] = Field(default_factory=list, description="找不到依据的句子")
    titles: List[str] = Field(default_factory=list, description="被引用的出处标题")
    used_chars: int = Field(0, description="上下文实际用掉的字符数")
    grounding: float = Field(0.0, ge=0.0, le=1.0, description="有依据句子占比，由校验器计算")

    def clamp_citations(self, n_contexts: int) -> "StructuredAnswer":
        """引用编号越界的直接丢掉，避免下游拿 [7] 去索引 3 条上下文。"""
        self.citations = [c for c in self.citations if 1 <= c <= n_contexts]
        return self


def render_json_instruction(n_contexts: int) -> str:
    """生成给模型的 schema 说明；引用上限随本轮上下文条数变化。"""
    return (
        "只输出一个 JSON 对象，不要有任何解释性文字、不要加代码块标记。字段：\n"
        '{"answer": "答案正文，引用写成 [1] [2]，编号只能是 1 到 '
        f'{max(n_contexts, 1)} 之间的整数",\n'
        ' "citations": [1, 2],\n'
        ' "refused": false}\n'
        "证据不足无法回答时，answer 里写明确的拒答说明，并把 refused 置为 true。"
    )


JSON_INSTRUCTION = render_json_instruction(8)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里抠出 JSON 对象：先剥代码块围栏，再退化为首尾花括号匹配。"""
    if not text:
        return None
    raw = text.strip()
    for candidate in _FENCE_RE.findall(raw):
        candidate = candidate.strip()
        if candidate.startswith("{"):
            raw = candidate
            break
    else:
        match = _OBJ_RE.search(raw)
        if match:
            raw = match.group(0)
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_structured_answer(
    text: str,
    n_contexts: int = 0,
    titles: Optional[List[str]] = None,
    used_chars: int = 0,
    grounding: float = 0.0,
    unsupported: Optional[List[str]] = None,
) -> Optional[StructuredAnswer]:
    """把模型输出解析成 StructuredAnswer，解析不出来返回 None（调用方决定退化策略）。"""
    payload = extract_json(text)
    if payload is None:
        return None
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        return None
    citations = payload.get("citations")
    if not isinstance(citations, list):
        # 没给 citations 就从正文里解析，兼容只输出 answer 的情况
        from .generate import parse_citations  # 局部导入，避免循环依赖

        citations = parse_citations(answer, max_index=n_contexts or None)
    model = StructuredAnswer(
        answer=answer,
        citations=[int(c) for c in citations if isinstance(c, (int, float))],
        refused=bool(payload.get("refused", False)),
        unsupported=list(unsupported or []),
        titles=list(titles or []),
        used_chars=int(used_chars or 0),
        grounding=float(grounding or 0.0),
    )
    if n_contexts:
        model.clamp_citations(n_contexts)
    return model


def to_json(model: StructuredAnswer) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False)
