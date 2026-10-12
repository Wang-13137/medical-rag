# -*- coding: utf-8 -*-
"""语料清洗：去掉 HTML 来源带进来的出版商网页样板（导航栏、备案号、版权与 AI 问答挂件）。

为什么需要这一步：
    上一轮混合检索上线后，几条 query 的 Top-3 被「高级检索 / 上一篇 下一篇 / 暂无评论」
    这类网页框架文字占掉了——文档数据结构otyping 时的 DOM 噪声没清干净。这些行本身在语料里
    重复出现，向量和 BM25 都会把它们当成正常内容。

清洗规则（两条，都可解释、不依赖黑箱模型）：
    1. 跨文档重复行：同一行在 >= min_df 份不同文档里都出现，判为样板；
       依据是正版医学内容不会在十几份不同指南里逐字重复，只有网站框架会。
    2. 备案 / 版权 / 挂件正则：ICP 备案、网信算备、版权所有、隐私政策、SparkDesk 等。

用法：
    python scripts/clean_corpus.py                       # data/raw -> data/clean
    python scripts/clean_corpus.py --min-df 3 --dry-run  # 只看会被删掉什么，不写文件
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# 备案号、版权、站点挂件等，命中整行即删
NOISE_PATTERNS = [
    r"^[|\s]*(联系我们|友情链接|帮助中心|网站地图)",
    r"ICP备|网信算备|出版物经营许可证|互联网药品信息服务资格证|京公网安备",
    r"版权所有|隐私政策|隐私条款|用户协议|法律声明|技术支持[:：]",
    r"REQUEST\s+PERMISSIONS|Cookie|万方数据|Google Scholar|百度学术",
    r"还没有聊天记录|历史对话|你好，我可以帮助|回答内容由人工智能生成",
    r"关注微信号|投稿|订阅|下载APP|纯净模式|沉浸模式|帮助中心",
    r"^(首页|期刊首页|过刊列表|高级检索|稿件发表|引用本文|评价本文|生成快照|文献快照|展开阅读)$",
    r"^(上一篇|下一篇|时间排序|暂无评论|评论|本文提要|本文全部|机器翻译|智问智答|文档即答)",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS))


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def build_boilerplate(files: list[Path], min_df: int) -> set[str]:
    """统计跨文档重复行：同一行在 >= min_df 份文档里出现即判为样板。"""
    df = Counter()
    for path in files:
        seen = {ln.strip() for ln in read_lines(path) if len(ln.strip()) >= 3}
        df.update(seen)
    return {line for line, count in df.items() if count >= min_df}


def clean_text(text: str, boilerplate: set[str]) -> tuple[str, int, int]:
    """返回 (清洗后文本, 删除行数, 原行数)。"""
    out = []
    dropped = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (stripped in boilerplate or NOISE_RE.search(stripped)):
            dropped += 1
            continue
        out.append(line)
    total = len(text.splitlines())
    return "\n".join(out), dropped, total


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗语料中的网页样板噪声")
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "data" / "clean"))
    parser.add_argument("--min-df", type=int, default=3, help="跨文档重复行的最小文档数")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(p for p in raw_dir.glob("*.txt"))
    if not files:
        print(f"[clean] {raw_dir} 下没有 txt，退出")
        return 1

    boilerplate = build_boilerplate(files, args.min_df)
    print(f"[clean] 语料 {len(files)} 份；跨文档重复行（df>={args.min_df}）黑名单 {len(boilerplate)} 条")

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_raw = total_clean = total_dropped = 0
    rows = []
    for path in files:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned, dropped, lines = clean_text(raw_text, boilerplate)
        total_raw += len(raw_text)
        total_clean += len(cleaned)
        total_dropped += dropped
        rows.append((path.name, lines, dropped, dropped / max(lines, 1), len(raw_text), len(cleaned)))
        if not args.dry_run:
            (out_dir / path.name).write_text(cleaned, encoding="utf-8")

    print(f"[clean] {'（dry-run，未写文件）' if args.dry_run else '输出目录：' + str(out_dir)}")
    print(f"[clean] {'文件名':<42} {'原行':>6} {'删行':>6} {'删行占比':>8} {'原字符':>9} {'剩字符':>9}")
    for name, lines, dropped, ratio, before, after in rows:
        print(f"       {name:<42} {lines:>6} {dropped:>6} {ratio:>7.1%} {before:>9} {after:>9}")
    print(f"[clean] 合计：删除 {total_dropped} 行，字符 {total_raw} -> {total_clean} "
          f"（砍掉 {1 - total_clean / max(total_raw, 1):.2%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
