# -*- coding: utf-8 -*-
"""一键构建索引：解析 → 切片 → 向量化 → 写 FAISS 索引。

用法（在仓库根目录执行）：
    python scripts/build_index.py                       # 用 data/raw 建索引
    python scripts/build_index.py --smoke               # 用 data/smoke 的样例数据建索引
    python scripts/build_index.py --data-dir D:/docs --index-path data/index/main.index
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

import numpy as np

from medical_rag.chunking import chunk_documents_from_config
from medical_rag.config import load_config
from medical_rag.embedding import BGEEmbedder
from medical_rag.ingest import load_documents
from medical_rag.index import FaissIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 FAISS 向量索引")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "config.yaml"), help="配置文件路径")
    parser.add_argument("--data-dir", default=None, help="覆盖配置里的 paths.data_dir")
    parser.add_argument("--index-path", default=None, help="覆盖配置里的 paths.index_path")
    parser.add_argument("--meta-path", default=None, help="覆盖配置里的 paths.meta_path")
    parser.add_argument("--smoke", action="store_true", help="使用 data/smoke 样例数据")
    parser.add_argument("--quiet", action="store_true", help="不打印模型与入库过程信息")
    return parser.parse_args()


def resolve_path(value: Path | str, is_index: bool = False) -> Path:
    """相对路径统一按仓库根目录解析；索引类路径写到 data/index 下。"""
    path = Path(value)
    return path if path.is_absolute() else (ROOT_DIR / path)


def main() -> int:
    args = parse_args()
    verbose = not args.quiet

    overrides = {}
    if args.index_path:
        overrides["paths.index_path"] = args.index_path
    if args.meta_path:
        overrides["paths.meta_path"] = args.meta_path
    if args.data_dir:
        overrides["paths.data_dir"] = args.data_dir

    cfg = load_config(args.config, overrides)
    # --smoke 未显式给出数据目录时，切到配置里的样例数据目录
    if args.smoke and not args.data_dir:
        overrides["paths.data_dir"] = cfg.get("paths.smoke_data_dir", "data/smoke")
        cfg = load_config(args.config, overrides)

    data_dir = resolve_path(cfg.require("paths.data_dir"))
    index_path = resolve_path(cfg.require("paths.index_path"))
    meta_path = resolve_path(cfg.require("paths.meta_path"))

    print(f"[build] 数据目录：{data_dir}")
    if not data_dir.exists() or not any(data_dir.rglob("*")):
        print(f"[build] 数据目录为空，请先把 PDF / docx / txt / md 放进 {data_dir}")
        return 1

    # 第一步：解析
    documents = load_documents(data_dir)
    if not documents:
        print("[build] 没有解析出任何文档，任务终止")
        return 1
    total_chars = sum(len(doc) for doc in documents)
    print(f"[build] 解析完成：{len(documents)} 份文档，共 {total_chars} 字符")

    # 第二步：切片
    chunks = chunk_documents_from_config(documents, cfg)
    if not chunks:
        print("[build] 切片结果为空，任务终止")
        return 1
    avg_len = sum(len(chunk.text) for chunk in chunks) / len(chunks)
    print(
        f"[build] 切片完成：{len(chunks)} 片，平均 {avg_len:.0f} 字符"
        f"（chunk_size={cfg.get('chunking.chunk_size')}, overlap={cfg.get('chunking.chunk_overlap')}）"
    )

    # 第三步：向量化
    embedder = BGEEmbedder.from_config(cfg, verbose=verbose)
    embeddings = embedder.encode_documents([chunk.text for chunk in chunks])
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"[build] 向量化完成：shape={embeddings.shape}, dtype={embeddings.dtype}")

    if cfg.get("embedding.normalize", True):
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"[build] 向量 L2 范数检查：min={norms.min():.6f}, max={norms.max():.6f}")

    # 第四步：写索引
    meta = {
        "model": embedder.metadata(),
        "chunking": {
            "chunk_size": cfg.get("chunking.chunk_size"),
            "chunk_overlap": cfg.get("chunking.chunk_overlap"),
            "prepend_heading_path": cfg.get("chunking.prepend_heading_path"),
        },
        "source_dir": str(data_dir),
        "num_documents": len(documents),
    }
    index = FaissIndex.build(embeddings, chunks, meta=meta)
    index_file, meta_file = index.save(index_path, meta_path)

    size_mb = index_file.stat().st_size / 1024 / 1024 if index_file.exists() else 0.0
    print(f"[build] 索引已写入：{index_file}（{size_mb:.2f} MB，{len(index)} 条）")
    print(f"[build] 元数据已写入：{meta_file}")
    print(f"[build] 完成。查询请运行：python scripts/query.py --q \"2型糖尿病血糖控制目标\" --index-path {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
