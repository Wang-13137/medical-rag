#!/usr/bin/env python
"""MCP stdio 服务入口：把 medical-rag 的检索能力暴露给任何支持 MCP 的客户端。

    python scripts/run_mcp_server.py                     # 常驻 stdio
    python scripts/run_mcp_server.py --keyword           # 额外启用 BM25 工具
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/run_mcp_server.py

在支持 MCP 的客户端里把命令配成上面这一行即可，不需要起 HTTP 服务。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.generate import EchoGenerator, load_generator  # noqa: E402
from medical_rag.mcp_server import MCPToolServer  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402
from run_agent import make_keyword_fn  # noqa: E402


def build_provider(use_keyword: bool = False):
    def provider() -> dict:
        cfg = load_config()
        candidate_k = int(cfg.get("rerank.candidate_k", 20))
        mode, retriever = make_retriever(cfg, True, candidate_k, verbose=True)
        try:
            generator = load_generator(cfg=cfg, verbose=True)
        except Exception:
            generator = EchoGenerator()
        keyword_fn = None
        if use_keyword:
            index = getattr(retriever, "index", None) or getattr(
                getattr(retriever, "first_stage", None), "index", None
            )
            if index is not None:
                keyword_fn = make_keyword_fn(index, cfg.get("retrieval.hybrid.cache_path"))
        return {
            "cfg": cfg,
            "mode": mode,
            "retriever": retriever,
            "generator": generator,
            "candidate_k": candidate_k,
            "keyword_fn": keyword_fn,
        }

    return provider


def main() -> int:
    ap = argparse.ArgumentParser(description="启动 MCP 工具服务（stdio 传输）")
    ap.add_argument("--keyword", action="store_true", help="启用 BM25 关键词检索工具")
    ap.add_argument("--once", help="只处理一行 JSON 请求后退出，便于调试")
    args = ap.parse_args()

    server = MCPToolServer(build_provider(args.keyword))
    if args.once:
        reply = server.handle(args.once)
        if reply:
            print(reply)
        return 0
    server.serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
