# -*- coding: utf-8 -*-
"""用 LangGraph 把同一套问答流程写成「固定编排图」，与手写 ReAct 形成对照。

对照的意义不是比谁分数高，而是把两种控制流的代价摆在同一张表上：

- 手写 ReAct（agent.py）：LLM 自己决定检索几次、读几条、什么时候收尾。
  可控性差（可能只读 2 条就收尾），但不需要事先假定流程。
- LangGraph 固定图（本模块）：节点与边写死，检索 -> 精排 -> 读 -> 生成 -> 校验，
  校验不过就剪掉无依据句子，最多改一轮。可控、可复现，代价是丧失自适应。

两者共用同一批底层能力（同一个 retriever、同一个精排器、同一个校验器），
所以对照里出现的差异只可能来自控制流本身。

用法：
    from medical_rag.langgraph_flow import build_graph
    graph = build_graph(retriever, generator)
    result = graph.run("妊娠合并急性阑尾炎首选手术还是保守治疗？")
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # LangGraph 是可选依赖，没装时给出明确报错而不是 import 崩掉
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - 仅在没有依赖时触发
    raise ImportError("需要 LangGraph：pip install langgraph") from exc

from typing_extensions import TypedDict

from .verify import verify_answer


class GraphState(TypedDict, total=False):
    query: str
    evidence: List[Dict[str, Any]]
    contexts: List[str]
    titles: List[str]
    answer: str
    unsupported: List[str]
    revisions: int
    refused: bool
    used_chars: int
    llm_calls: int
    trace: List[Dict[str, Any]]
    stop_reason: str


def _hits_to_items(hits: Sequence[Any]) -> List[Dict[str, Any]]:
    items = []
    for h in hits:
        chunk = getattr(h, "chunk", None)
        head = " > ".join(getattr(chunk, "heading_path", None) or [])
        doc = getattr(chunk, "doc_name", "") or ""
        items.append(
            {
                "text": getattr(chunk, "text", "") or "",
                "title": f"来源：{doc}" + (f" · {head}" if head else ""),
                "score": float(getattr(h, "score", 0.0) or 0.0),
            }
        )
    return items


class GuidelineGraph:
    """固定编排图。compile 之后 run 一次即对一题完成全流程。"""

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        keyword_search: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        reranker: Any = None,
        candidate_k: int = 20,
        top_n: int = 8,
        budget_chars: int = 2400,
        max_revisions: int = 1,
        mode: str = "strict",
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.keyword_search = keyword_search
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_n = top_n
        self.budget_chars = budget_chars
        self.max_revisions = max_revisions
        self.mode = mode
        self.app = self._compile()

    # ------------------------------------------------------------ 节点

    def _log(self, state: GraphState, node: str, **fields: Any) -> List[Dict[str, Any]]:
        trace = list(state.get("trace") or [])
        trace.append({"node": node, **fields})
        return trace

    def node_retrieve(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]
        k = max(self.candidate_k, self.top_n)
        items = _hits_to_items(self.retriever.retrieve(query, top_k=k))
        if self.keyword_search is not None:
            seen = {it["text"] for it in items}
            for it in self.keyword_search(query, k):
                if it.get("text") and it["text"] not in seen:
                    items.append(
                        {
                            "text": it["text"],
                            "title": it.get("title") or "",
                            "score": float(it.get("score") or 0.0),
                        }
                    )
                    seen.add(it["text"])
        return {
            "evidence": items,
            "trace": self._log(state, "retrieve", count=len(items)),
        }

    def node_rerank(self, state: GraphState) -> Dict[str, Any]:
        evidence = list(state.get("evidence") or [])
        if self.reranker is None or not evidence:
            return {"evidence": evidence, "trace": self._log(state, "rerank", skipped=True)}
        scores = [float(s) for s in self.reranker.score(state["query"], [e["text"] for e in evidence])]
        ranked = [e for _, e in sorted(zip(scores, evidence), key=lambda p: p[0], reverse=True)]
        for e, s in zip(ranked, sorted(scores, reverse=True)):
            e["score"] = s
        return {"evidence": ranked, "trace": self._log(state, "rerank", count=len(ranked))}

    def node_read(self, state: GraphState) -> Dict[str, Any]:
        contexts: List[str] = []
        titles: List[str] = []
        used = 0
        for e in state.get("evidence") or []:
            if len(contexts) >= self.top_n:
                break
            text = e.get("text") or ""
            if not text:
                continue
            if used + len(text) > self.budget_chars and contexts:
                break
            contexts.append(text)
            titles.append(e.get("title") or "")
            used += len(text)
        return {
            "contexts": contexts,
            "titles": titles,
            "used_chars": used,
            "trace": self._log(state, "read", count=len(contexts), used_chars=used),
        }

    def node_generate(self, state: GraphState) -> Dict[str, Any]:
        gen = self.generator.answer(state["query"], state.get("contexts") or [], state.get("titles") or [], self.mode)
        return {
            "answer": getattr(gen, "answer", "") or "",
            "llm_calls": int(state.get("llm_calls") or 0) + 1,
            "trace": self._log(state, "generate", chars=len(getattr(gen, "answer", "") or "")),
        }

    def node_verify(self, state: GraphState) -> Dict[str, Any]:
        result = verify_answer(state.get("answer") or "", state.get("contexts") or [], mode="flag")
        return {
            "unsupported": list(result.unsupported_sentences),
            "refused": bool(getattr(result, "refused", False)),
            "trace": self._log(state, "verify", unsupported=len(result.unsupported_sentences)),
        }

    def node_revise(self, state: GraphState) -> Dict[str, Any]:
        """剪掉无依据句子。这里不再调 LLM，控制的是「成本可控的修正」。"""
        result = verify_answer(state.get("answer") or "", state.get("contexts") or [], mode="prune")
        revised = result.revised_answer or ""
        return {
            "answer": revised,
            "revisions": int(state.get("revisions") or 0) + 1,
            "unsupported": list(result.unsupported_sentences),
            "refused": bool(getattr(result, "refused", False)),
            "trace": self._log(state, "revise", chars=len(revised)),
        }

    def node_finish(self, state: GraphState) -> Dict[str, Any]:
        return {"stop_reason": "finish", "trace": self._log(state, "finish")}

    # ------------------------------------------------------------ 边

    def _after_verify(self, state: GraphState) -> str:
        unsupported = state.get("unsupported") or []
        if not unsupported:
            return "finish"
        if int(state.get("revisions") or 0) >= self.max_revisions:
            return "finish"
        return "revise"

    def _compile(self) -> Any:
        graph: StateGraph = StateGraph(GraphState)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("rerank", self.node_rerank)
        graph.add_node("read", self.node_read)
        graph.add_node("generate", self.node_generate)
        graph.add_node("verify", self.node_verify)
        graph.add_node("revise", self.node_revise)
        graph.add_node("finish", self.node_finish)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "read")
        graph.add_edge("read", "generate")
        graph.add_edge("generate", "verify")
        graph.add_conditional_edges(
            "verify",
            self._after_verify,
            {"revise": "revise", "finish": "finish"},
        )
        graph.add_edge("revise", "finish")
        graph.add_edge("finish", END)
        return graph.compile()

    # ------------------------------------------------------------ 运行

    def run(self, query: str) -> Dict[str, Any]:
        started = time.time()
        state: GraphState = {
            "query": query,
            "evidence": [],
            "contexts": [],
            "titles": [],
            "answer": "",
            "unsupported": [],
            "revisions": 0,
            "refused": False,
            "used_chars": 0,
            "llm_calls": 0,
            "trace": [],
            "stop_reason": "",
        }
        final = self.app.invoke(state)
        return {
            "query": query,
            "answer": final.get("answer") or "",
            "refused": bool(final.get("refused")),
            "contexts": list(final.get("contexts") or []),
            "titles": list(final.get("titles") or []),
            "used_chars": int(final.get("used_chars") or 0),
            "llm_calls": int(final.get("llm_calls") or 0),
            "nodes": [t["node"] for t in final.get("trace") or []],
            "trace": list(final.get("trace") or []),
            "unsupported": list(final.get("unsupported") or []),
            "revisions": int(final.get("revisions") or 0),
            "stop_reason": final.get("stop_reason") or "finish",
            "latency_sec": round(time.time() - started, 3),
        }


def build_graph(
    retriever: Any,
    generator: Any,
    keyword_search: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    reranker: Any = None,
    candidate_k: int = 20,
    top_n: int = 8,
    budget_chars: int = 2400,
    max_revisions: int = 1,
    mode: str = "strict",
) -> GuidelineGraph:
    return GuidelineGraph(
        retriever,
        generator,
        keyword_search=keyword_search,
        reranker=reranker,
        candidate_k=candidate_k,
        top_n=top_n,
        budget_chars=budget_chars,
        max_revisions=max_revisions,
        mode=mode,
    )
