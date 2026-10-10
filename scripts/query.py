# -*- coding: utf-8 -*-
"""命令行检索测试：加载已构建的索引，输出 Top-K 片段。

用法（在仓库根目录执行）：
    python scripts/query.py --q "2型糖尿病血糖控制目标"
    python scripts/query.py --q "他汀类药物的禁忌" --smoke --top-k 3
    python scripts/query.py --index-path data/index/main.index --meta-path data/index/chunks_main.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from medical_rag.retrieve import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="向量检索基线查询")
    parser.add_argument("--q", "--query", dest="query", required=True, help="查询文本")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"), help="配置文件路径")
    parser.add_argument("--index-path", default=None, help="覆盖配置里的 paths.index_path")
    parser.add_argument("--meta-path", default=None, help="覆盖配置里的 paths.meta_path")
    parser.add_argument("--smoke", action="store_true", help="使用配置里的 smoke 索引路径（默认即此路径）")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数，默认取配置 retrieval.top_k")
    parser.add_argument("--full", action="store_true", help="打印完整文本而不是截断预览")
    parser.add_argument("--quiet", action="store_true", help="不打印模型加载信息")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    overrides = {}
    if args.index_path:
        overrides["paths.index_path"] = args.index_path
    if args.meta_path:
        overrides["paths.meta_path"] = args.meta_path

    try:
        retriever = Retriever.from_config_paths(args.config, overrides, verbose=not args.quiet)
    except FileNotFoundError as exc:
        print(f"[query] {exc}")
        print("[query] 请先运行 scripts/build_index.py 构建索引")
        return 1

    hits = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"[query] query：{args.query}")
    print(f"[query] 命中 {len(hits)} 条（索引共 {len(retriever.index)} 条，score 为余弦相似度）")
    print("-" * 78)
    width = 10_000 if args.full else 200
    print(retriever.format_hits(hits, text_width=width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
