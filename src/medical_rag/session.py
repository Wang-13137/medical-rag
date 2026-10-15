# -*- coding: utf-8 -*-
"""多轮会话记忆：把「上一条问的是什么」保留下来，让追问能被检索到。

RAG 系统最常见的多轮故障不是记不住答案，而是**追问里的指代消失**：
用户第二轮问「那疗程多久？」「这个药孕妇能用吗？」，检索器拿到一句没有主体的话，
召回质量直接掉到随机水平。这里做的三件事：

1. rewrite：把指代型追问改写成一个可独立检索的 query（拼回上一轮的主题词）；
2. history_block：给生成器一段压缩过的历史，避免整段对话原样塞进上下文；
3. 出处偏好：同一份指南上一轮被引用过，本轮命中时更值得读（仅提示，不硬性过滤）。

刻意不做的事：不把整段历史拼进检索 query（会稀释主题），
不做向量化的「记忆库」（本语料规模下收益不稳定，且无法解释）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .hybrid import tokenize

# 出现这些词，基本可以判定这一轮是追问而不是新话题
REFERENCE_WORDS = (
    "它", "这个", "那个", "其", "上述", "该", "这种", "这类", "前者", "后者",
    "这种药", "这个病", "这个方案", "那如果", "那么", "另外", "还有", "再问",
)
# 这些词单独出现时没有检索价值，只作追问信号
WEAK_QUERY_WORDS = ("剂量", "疗程", "用量", "用法", "多久", "禁忌", "注意", "复查", "能不能", "可以吗")

STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "有", "对", "吗", "呢", "怎么", "如何",
    "什么", "哪些", "请问", "需要", "可以", "患者", "应该", "是否", "一个", "这种",
}


@dataclass
class Turn:
    question: str
    answer: str = ""
    titles: List[str] = field(default_factory=list)
    citations: List[int] = field(default_factory=list)
    contexts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "titles": list(self.titles),
            "citations": list(self.citations),
            "contexts": self.contexts,
        }


class SessionMemory:
    """一轮一存，只保留最近 max_turns 轮，超出即丢最早的。"""

    def __init__(self, session_id: str = "", max_turns: int = 5, history_chars: int = 600) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self.history_chars = history_chars
        self.turns: List[Turn] = []

    # ------------------------------------------------------------ 写入

    def add(
        self,
        question: str,
        answer: str = "",
        titles: Optional[Sequence[str]] = None,
        citations: Optional[Sequence[int]] = None,
        contexts: int = 0,
    ) -> Turn:
        turn = Turn(
            question=question,
            answer=answer,
            titles=list(titles or []),
            citations=list(citations or []),
            contexts=contexts,
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
        return turn

    # ------------------------------------------------------------ 指代消解

    def is_followup(self, query: str) -> bool:
        if not self.turns:
            return False
        compact = query.replace(" ", "")
        if any(w in compact for w in REFERENCE_WORDS):
            return True
        # 「疗程多久？」这类没有主语的短问，也算追问
        if len(compact) <= 12 and any(w in compact for w in WEAK_QUERY_WORDS):
            return True
        return False

    def topic_terms(self, limit: int = 6) -> List[str]:
        """从最近一轮问题里抽主题词：去掉停用词与追问信号词。"""
        if not self.turns:
            return []
        text = self.turns[-1].question
        terms: List[str] = []
        for tok in tokenize(text):
            if len(tok) < 2 or tok in STOPWORDS or tok in WEAK_QUERY_WORDS:
                continue
            if tok in REFERENCE_WORDS:
                continue
            if tok not in terms:
                terms.append(tok)
            if len(terms) >= limit:
                break
        return terms

    def rewrite(self, query: str) -> str:
        """把追问改写成能独立检索的 query；非追问原样返回。"""
        if not self.is_followup(query):
            return query
        terms = self.topic_terms()
        if not terms:
            return query
        return " ".join(terms) + " " + query.strip()

    # ------------------------------------------------------------ 历史

    def history_block(self) -> str:
        """压缩后的对话历史，供生成器消歧；不超过 history_chars。

        只统计**已完成**的轮次：调用方应在把本轮 add 进来之前取历史，
        否则本轮的问题会被当成历史又喂一遍给模型。
        """
        if not self.turns:
            return ""
        lines: List[str] = []
        used = 0
        for turn in reversed(self.turns):
            snippet = (turn.answer or "").replace("\n", " ").strip()
            snippet = re.sub(r"\[\d+\]", "", snippet)
            line = f"问：{turn.question}\n答：{snippet[:120]}"
            if used + len(line) > self.history_chars and lines:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        return "【对话历史，由早到晚】\n" + "\n".join(reversed(lines))

    # ------------------------------------------------------------ 出处偏好

    def preferred_titles(self) -> List[str]:
        """上一轮引用过的出处，本轮命中同一出处时可优先读。"""
        if not self.turns:
            return []
        return list(self.turns[-1].titles)

    def rank_contexts(self, contexts: Sequence[str], titles: Sequence[str]) -> List[int]:
        """返回按下标排序的阅读顺序：本轮顺序为主，上一轮引用过的出处提前。"""
        preferred = set(self.preferred_titles())
        if not preferred:
            return list(range(len(contexts)))
        score = []
        for i, title in enumerate(titles):
            score.append((0 if title in preferred else 1, i))
        return [i for _, i in sorted(score)]

    # ------------------------------------------------------------ 序列化

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "max_turns": self.max_turns,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionMemory":
        mem = cls(payload.get("session_id", ""), int(payload.get("max_turns", 5)))
        for t in payload.get("turns") or []:
            mem.add(
                t.get("question", ""),
                t.get("answer", ""),
                t.get("titles") or [],
                t.get("citations") or [],
                int(t.get("contexts") or 0),
            )
        return mem


class SessionStore:
    """按 session_id 存放会话；容量上限后淘汰最久未更新的。"""

    def __init__(self, capacity: int = 200, max_turns: int = 5) -> None:
        self.capacity = capacity
        self.max_turns = max_turns
        self._sessions: Dict[str, SessionMemory] = {}
        self._order: List[str] = []

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id, self.max_turns)
            self._order.append(session_id)
            self._evict()
        else:
            self._order.remove(session_id)
            self._order.append(session_id)
        return self._sessions[session_id]

    def _evict(self) -> None:
        while len(self._order) > self.capacity:
            oldest = self._order.pop(0)
            self._sessions.pop(oldest, None)

    def __len__(self) -> int:
        return len(self._sessions)
