"""从真实指南语料中抽取「命题素材」，供人工编写评测题时核对答案出处。

只做抽取与统计，不产出任何答案。原则：写进评测集的每个关键串，
都必须能在指定来源文档里搜到原文，宁可少出题也不编答案。

用法:
    python scripts/mine_eval_material.py --per-doc 40 --out reports/eval_material.md
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DIR = ROOT / "data" / "clean"

# 强推荐句式：指南里真正给出意见的句子，才配拿来出题
RECOMMEND_RE = re.compile(
    r"(推荐|建议|首选|一线|应避免|不建议|禁用|慎用|可选择|"
    r"适应证|禁忌证|诊断标准|控制目标|指征)"
)
# 数值 / 剂量 / 阈值：数字型题目的来源
NUMBER_RE = re.compile(
    r"(\d+(\.\d+)?\s*(mg|g|μg|ug|ml|mL|U|IU|mmol/L|mol/L|mmHg|cm|mm|%|"
    r"分钟|小时|天|周|月|岁|次/日|次/d|kg))"
)
# 推荐强度 / 证据等级（GRADE）
GRADE_RE = re.compile(r"(强推荐|弱推荐|[\(（]?[ⅠⅠ1II][类级][\)）]?|[ABC][级类]证据)")

HEADING_RE = re.compile(r"^\s*(第[一二三四五六七八九十]+章|[一二三四五六七八九十]+[、．.]|\d+(\.\d+)*[、．.\s])")


def iter_sentences(text: str):
    """按中文标点切句，过滤过短碎片。"""
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in "。；!?\n":
            s = "".join(buf).strip()
            if len(s) >= 12:
                yield s
            buf = []
    tail = "".join(buf).strip()
    if len(tail) >= 12:
        yield tail


def mine_document(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]
    headings = [ln for ln in lines if ln and HEADING_RE.match(ln) and len(ln) <= 40]

    hot: list[tuple[str, float]] = []
    for s in iter_sentences(text):
        s = re.sub(r"\s+", " ", s)
        score = 0.0
        if RECOMMEND_RE.search(s):
            score += 1.0
        if NUMBER_RE.search(s):
            score += 1.5  # 数字型答案最有评测价值
        if GRADE_RE.search(s):
            score += 0.5
        if score > 0 and 20 <= len(s) <= 220:
            hot.append((s, score))

    hot.sort(key=lambda x: -x[1])
    return {
        "name": path.name,
        "chars": len(text),
        "headings": headings,
        "hot": hot,
        "num_sent": sum(1 for s, _ in hot if NUMBER_RE.search(s)),
        "rec_sent": sum(1 for s, _ in hot if RECOMMEND_RE.search(s)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="抽取评测命题素材")
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--per-doc", type=int, default=40, help="每份文档保留的高价值句子数")
    ap.add_argument("--out", default="reports/eval_material.md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.txt"))
    docs = [mine_document(p) for p in files]

    total_chars = sum(d["chars"] for d in docs)
    lines = [
        "# 评测命题素材（自动抽取，仅供人工出题核对）",
        "",
        f"- 语料目录：`{data_dir}`",
        f"- 文档数：{len(docs)}，总字符数：{total_chars:,}",
        "",
        "| 文档 | 字符数 | 含推荐句 | 含数值句 |",
        "| --- | --- | --- | --- |",
    ]
    for d in docs:
        lines.append(f"| `{d['name']}` | {d['chars']:,} | {d['rec_sent']} | {d['num_sent']} |")

    counter = Counter()
    lines.append("")
    for d in docs:
        lines.append("")
        lines.append(f"## `{d['name']}`")
        seen = set()
        picked = []
        for s, _ in d["hot"]:
            key = s[:30]
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= args.per_doc:
                break
        # 按原文出现顺序重排，便于定位上下文
        order = {s: i for i, s in enumerate(picked)}
        picked.sort(key=lambda s: order[s])
        for s in picked:
            counter[d["name"]] += 1
            lines.append(f"- {s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[mine] 文档 {len(docs)} 份，素材写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
