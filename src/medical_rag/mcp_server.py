# -*- coding: utf-8 -*-
"""MCP 工具服务：把检索 / 精排 / 校验 / 问答能力暴露成标准工具协议。

为什么手写而不引 SDK：
- 这个仓库一贯的做法是「能手写就手写，再用框架做对照」，ReAct 主循环就是这么做的，
  协议层保持一致，面试时能讲清楚帧格式而不是只会调库；
- 依赖面小，不会因为上游 SDK 版本变动把服务搞挂。

实现的是 MCP 的 stdio 传输 + JSON-RPC 2.0 帧，支持：
    initialize / notifications/initialized / ping / tools/list / tools/call
工具结果同时给 content（文本）与 structuredContent（结构化），
后者是给上层 Agent 直接消费的，省掉一次解析。

用法：
    python scripts/run_mcp_server.py            # 进程内 stdio 服务
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/run_mcp_server.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .generate import parse_citations
from .verify import verify_answer

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = (PROTOCOL_VERSION, "2024-11-05")
SERVER_INFO = {"name": "medical-rag", "version": "0.6.0"}

# JSON-RPC 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[..., Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _schema(properties: Dict[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


class MCPToolServer:
    """无状态工具服务：每次 tools/call 相互独立，不保留工作集。"""

    def __init__(
        self,
        provider: Callable[[], Dict[str, Any]],
        server_info: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider = provider
        self.server_info = server_info or SERVER_INFO
        self._resources: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------ 资源

    def resources(self) -> Dict[str, Any]:
        if self._resources is None:
            self._resources = self.provider()
        return self._resources

    def reset(self) -> None:
        self._resources = None

    # ------------------------------------------------------------ 工具实现

    def _retriever(self) -> Any:
        return self.resources()["retriever"]

    def _hit_rows(self, hits: Sequence[Any]) -> List[Dict[str, Any]]:
        rows = []
        for h in hits:
            chunk = getattr(h, "chunk", None)
            head = " > ".join(getattr(chunk, "heading_path", None) or [])
            doc = getattr(chunk, "doc_name", "") or ""
            rows.append(
                {
                    "rank": int(getattr(h, "rank", 0) or 0),
                    "score": round(float(getattr(h, "score", 0.0) or 0.0), 4),
                    "source": f"{doc} · {head}" if head else doc,
                    "text": getattr(chunk, "text", "") or "",
                }
            )
        return rows

    def _tool_search(self, query: str = "", top_k: int = 8) -> Dict[str, Any]:
        if not query:
            return {"error": "缺少 query"}
        hits = self._retriever().retrieve(query, top_k=int(top_k))
        rows = self._hit_rows(hits)
        return {"query": query, "count": len(rows), "results": rows}

    def _tool_keyword(self, query: str = "", top_k: int = 8) -> Dict[str, Any]:
        fn = self.resources().get("keyword_fn")
        if fn is None:
            return {"available": False, "reason": "BM25 工具未启用（启动参数 --keyword）"}
        if not query:
            return {"error": "缺少 query"}
        items = fn(query, int(top_k))
        rows = [
            {
                "rank": i + 1,
                "score": round(float(it.get("score") or 0.0), 4),
                "source": it.get("title") or "",
                "text": (it.get("text") or "")[:400],
            }
            for i, it in enumerate(items)
        ]
        return {"query": query, "available": True, "count": len(rows), "results": rows}

    def _tool_rerank(self, query: str = "", top_k: int = 5) -> Dict[str, Any]:
        retriever = self._retriever()
        if not hasattr(retriever, "explain"):
            return {"available": False, "reason": "当前检索器未启用 cross-encoder 精排"}
        rows = retriever.explain(query, top_k=int(top_k))
        return {
            "available": True,
            "count": len(rows),
            "results": [
                {
                    "rank": r["rank"],
                    "rerank_score": round(float(r["rerank_score"]), 4),
                    "first_stage_rank": r["first_stage_rank"],
                    "doc_name": r["doc_name"],
                    "text": r["text"][:400],
                }
                for r in rows
            ],
        }

    def _tool_verify(self, answer: str = "", contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        ctx = list(contexts or [])
        if not answer:
            return {"error": "缺少待校验的 answer"}
        if not ctx:
            return {"error": "缺少 contexts（提供待校验依据的原文片段列表）"}
        result = verify_answer(answer, ctx, mode="flag")
        return {
            "refused": bool(getattr(result, "refused", False)),
            "unsupported_sentences": list(result.unsupported_sentences),
            "verdicts": [
                {
                    "sentence": v.text,
                    "supported": bool(v.supported),
                    "coverage": round(float(v.coverage), 3),
                    "unsupported_numbers": list(v.unsupported_numbers),
                }
                for v in result.verdicts
            ],
        }

    def _tool_answer(
        self,
        question: str = "",
        mode: str = "strict",
        top_n: int = 8,
        budget: int = 2400,
    ) -> Dict[str, Any]:
        if not question:
            return {"error": "缺少 question"}
        res = self.resources()
        retriever, generator = res["retriever"], res["generator"]
        hits = retriever.retrieve(question, top_k=max(int(res.get("candidate_k", 20)), int(top_n)))
        contexts: List[str] = []
        titles: List[str] = []
        used = 0
        for h in hits:
            if len(contexts) >= int(top_n):
                break
            text = getattr(h.chunk, "text", "") or ""
            if not text:
                continue
            if used + len(text) > int(budget) and contexts:
                break
            contexts.append(text)
            head = " > ".join(getattr(h.chunk, "heading_path", None) or [])
            doc = getattr(h.chunk, "doc_name", "") or ""
            titles.append(f"来源：{doc}" + (f" · {head}" if head else ""))
            used += len(text)
        gen = generator.answer(question, contexts, titles, mode)
        result = verify_answer(gen.answer, contexts, mode="flag")
        return {
            "answer": gen.answer,
            "refused": bool(getattr(result, "refused", False)),
            "citations": list(parse_citations(gen.answer, max_index=len(contexts))),
            "titles": titles,
            "used_chars": used,
            "retrieval_mode": res.get("mode", ""),
            "generator": type(generator).__name__,
            "unsupported_sentences": list(result.unsupported_sentences),
        }

    def _tool_stats(self) -> Dict[str, Any]:
        res = self.resources()
        index = getattr(res["retriever"], "index", None)
        chunks = len(getattr(index, "chunks", []) or [])
        return {
            "index_chunks": chunks,
            "retrieval_mode": res.get("mode", ""),
            "generator": type(res["generator"]).__name__,
            "stage": str(res["cfg"].get("project.stage", "")),
        }

    # ------------------------------------------------------------ 工具清单

    def tools(self) -> List[MCPTool]:
        return [
            MCPTool(
                "guideline_search",
                "语义检索中文临床指南片段，返回排名、分数、出处与正文",
                _schema(
                    {
                        "query": {"type": "string", "description": "检索问题"},
                        "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                    },
                    ["query"],
                ),
                self._tool_search,
            ),
            MCPTool(
                "keyword_search",
                "BM25 关键词检索，适合定位专有名词、药名与剂量数字",
                _schema(
                    {
                        "query": {"type": "string", "description": "检索关键词"},
                        "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                    },
                    ["query"],
                ),
                self._tool_keyword,
            ),
            MCPTool(
                "rerank_explain",
                "cross-encoder 精排，并给出每条结果在精排前的原始名次",
                _schema(
                    {
                        "query": {"type": "string", "description": "检索问题"},
                        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    },
                    ["query"],
                ),
                self._tool_rerank,
            ),
            MCPTool(
                "verify_answer",
                "逐句核对答案是否能在给定原文片段中找到依据，标出无来源的数字",
                _schema(
                    {
                        "answer": {"type": "string", "description": "待校验的答案"},
                        "contexts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "被引用的原文片段列表",
                        },
                    },
                    ["answer", "contexts"],
                ),
                self._tool_verify,
            ),
            MCPTool(
                "answer_question",
                "走完整条检索增强链路回答问题，返回答案、引用、出处与无依据句子",
                _schema(
                    {
                        "question": {"type": "string", "description": "用户问题"},
                        "mode": {"type": "string", "default": "strict", "enum": ["strict", "naive"]},
                        "top_n": {"type": "integer", "default": 8, "minimum": 1, "maximum": 12},
                        "budget": {"type": "integer", "default": 2400, "minimum": 400, "maximum": 4000},
                    },
                    ["question"],
                ),
                self._tool_answer,
            ),
            MCPTool(
                "index_stats",
                "查看索引规模、检索模式与当前使用的生成器",
                _schema({}, []),
                self._tool_stats,
            ),
        ]

    # ------------------------------------------------------------ JSON-RPC

    def handle(self, raw: str) -> Optional[str]:
        """处理一帧请求，返回一帧响应字符串；通知类返回 None。"""
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._reply_error(None, PARSE_ERROR, "JSON 解析失败")
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            return self._reply_error(None, INVALID_REQUEST, "不是合法的 JSON-RPC 2.0 请求")

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}
        is_notification = "id" not in req

        try:
            if method == "initialize":
                client_version = params.get("protocolVersion") or PROTOCOL_VERSION
                version = client_version if client_version in SUPPORTED_VERSIONS else PROTOCOL_VERSION
                return self._reply(
                    req_id,
                    {
                        "protocolVersion": version,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": self.server_info,
                    },
                )
            if method in ("notifications/initialized", "initialized"):
                return None
            if method == "ping":
                return self._reply(req_id, {})
            if method == "tools/list":
                return self._reply(req_id, {"tools": [t.to_dict() for t in self.tools()]})
            if method == "tools/call":
                return self._reply(req_id, self._call_tool(params))
            return self._reply_error(req_id, METHOD_NOT_FOUND, f"不支持的方法：{method}")
        except Exception as exc:  # 工具内部异常不该把服务打死
            if is_notification:
                return None
            return self._reply_error(req_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """工具级错误走 isError 结果帧，不升级成协议错误，连接因此不会被打断。"""
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        table = {t.name: t for t in self.tools()}
        if name not in table:
            return self._tool_error(f"未知工具：{name}，可用：{', '.join(sorted(table))}")
        if not isinstance(arguments, dict):
            return self._tool_error("arguments 必须是对象")
        try:
            payload = table[name].func(**arguments)
        except TypeError as exc:
            return self._tool_error(f"参数不对：{exc}")
        except Exception as exc:
            return self._tool_error(f"{type(exc).__name__}: {exc}")
        is_error = bool(payload.get("error"))
        text = json.dumps(payload, ensure_ascii=False)
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": payload,
            "isError": is_error,
        }

    @staticmethod
    def _tool_error(message: str) -> Dict[str, Any]:
        return {
            "content": [{"type": "text", "text": message}],
            "structuredContent": {"error": message},
            "isError": True,
        }

    @staticmethod
    def _reply(req_id: Any, result: Dict[str, Any]) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}, ensure_ascii=False)

    @staticmethod
    def _reply_error(req_id: Any, code: int, message: str) -> str:
        return json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )

    def serve_stdio(self, stdin: Any = None, stdout: Any = None) -> None:
        """按行读 stdin、按行写 stdout。供调试用的同步循环。"""
        inp = stdin if stdin is not None else sys.stdin
        out = stdout if stdout is not None else sys.stdout
        for line in inp:
            line = line.strip()
            if not line:
                continue
            reply = self.handle(line)
            if reply is None:
                continue
            out.write(reply + "\n")
            out.flush()


def build_mcp_server(
    provider: Callable[[], Dict[str, Any]],
    server_info: Optional[Dict[str, str]] = None,
) -> MCPToolServer:
    return MCPToolServer(provider, server_info)
