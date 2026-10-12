# -*- coding: utf-8 -*-
"""把 data/source_original 下两份痛风指南源文件转成 data/raw 的纯文本。

- hyperuricemia_gout_2019：中华内分泌代谢杂志 HTML 全文 → txt
- hyperuricemia_gout_2024：两个 PDF（第一部分 / 第二部分）合并 → txt

用法：python scripts/convert_gout_sources.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

SRC_DIR = ROOT_DIR / "data" / "source_original" / "clinical_guidelines"
RAW_DIR = ROOT_DIR / "data" / "raw"

# 导航、版权、页眉页脚等噪声行特征
_NOISE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"(javascript|var\s+\w+\s*=|\{\s*\}|\)\s*;\s*$)"),
    re.compile(r"^(首页|登录|注册|投稿|订阅|关于我们|联系我们|帮助|反馈)$"),
    re.compile(r"^(上一篇|下一篇|返回顶部|打印|收藏|分享|引用本文)$"),
    re.compile(r"^http(s)?://"),
    re.compile(r"^\s*(Copyright|版权所有|All rights reserved)", re.I),
    re.compile(r"^[\s\-—_=·.。|]+$"),
]

# 正文中需要保留的段落最小长度（剔除纯标签按钮等碎片）
_MIN_PARA_CHARS = 2


# 期刊 HTML 里正文结束后的元信息与互动区，命中即截断
_TAIL_CUTOFF = re.compile(
    r"^(历史\s*$|Published\s|Cite as\s|Contributor Information|共有\s*\d*\s*条评论|"
    r"施引文献|相关视频|相关资源|Abstract$|PDF全文|全文下载)"
)

# PDF 里第三方转载留下的水印，形如 fmx_T3RoZXJNaXJyb3Jz
_WATERMARK = re.compile(r"fmx_[A-Za-z0-9+/=]+")

# 双栏期刊 PDF 提取产生的页眉行（刊名 + 年卷期页码）
_RUNNING_HEADER = re.compile(r"ＣｈｉｎＪＥｎｄｏｃｒｉｎｏｌＭｅｔａｂ|中华内分泌代谢杂志\s*年\s*月第")


def _strip_pdf_noise(text: str) -> str:
    """去掉 PDF 提取产生的水印、页眉行与私有区字形。"""
    # 私有区字符（BMP PUA 与 Supplementary PUA-A/B），pdfplumber 会把装饰字形解成这些码位
    text = "".join(
        ch for ch in text
        if not (0xE000 <= ord(ch) <= 0xF8FF or 0xF0000 <= ord(ch) <= 0x10FFFD)
    )
    text = _WATERMARK.sub("", text)
    kept = [line for line in text.splitlines() if line.strip() and not _RUNNING_HEADER.search(line)]
    text = "\n".join(line.strip() for line in kept)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _cut_html_tail(text: str) -> str:
    """正文之后的出版信息、评论区等不属于指南内容，一律截断。"""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _TAIL_CUTOFF.match(line.strip()):
            return "\n".join(lines[:index]).strip()
    return text


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _MIN_PARA_CHARS:
        return True
    return any(pattern.search(stripped) for pattern in _NOISE_PATTERNS)


def html_to_text(html_path: Path) -> str:
    """把期刊 HTML 全文转成保留段落结构的纯文本。"""
    from bs4 import BeautifulSoup

    raw = html_path.read_text("utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # 去掉不承载正文的节点
    for tag in soup.find_all(["script", "style", "noscript", "head", "nav", "footer", "iframe", "svg"]):
        tag.decompose()

    # 优先取正文容器：在候选容器里选文本最长的一个，而不是第一个 class 命中的，
    # 否则容易选中 article_head_title 这类只有标题的空壳 div。
    body = soup.body or soup
    candidates = soup.find_all(["div", "article"], class_=re.compile(r"(article|content|main|text)", re.I))
    candidates += soup.find_all("article")
    if candidates:
        # 只有当最佳候选确实承载了大部分正文时才采用，否则沿用 body
        best = max(candidates, key=lambda node: len(node.get_text(strip=True)))
        if len(best.get_text(strip=True)) >= 0.5 * len(body.get_text(strip=True)):
            body = best

    blocks: list[str] = []
    for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th", "br", "div"]):
        if element.name == "br":
            continue
        # div 只在没有更细粒度子标签时取值，避免重复输出嵌套内容
        if element.name == "div" and element.find(["p", "li", "div", "table"]):
            continue
        text = element.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if not text or _is_noise(text):
            continue
        if blocks and blocks[-1] == text:
            continue
        blocks.append(text)

    if not blocks:  # 兜底：整篇 get_text
        fallback = body.get_text("\n", strip=True)
        blocks = [line.strip() for line in fallback.splitlines() if line.strip() and not _is_noise(line)]

    return "\n\n".join(blocks)


def pdf_to_text(pdf_paths: list[Path], titles: list[str]) -> str:
    """按页提取 PDF 文本，双栏排版下 pdfplumber 已按阅读序输出。"""
    import pdfplumber

    sections: list[str] = []
    for pdf_path, title in zip(pdf_paths, titles):
        page_texts: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = re.sub(r"[ \t]+", " ", text)
                page_texts.append(text.strip())
        body = "\n".join(t for t in page_texts if t)
        sections.append(f"{title}\n\n{body}")
    return "\n\n".join(sections)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 2019 版：HTML
    html_path = SRC_DIR / "hyperuricemia_gout_2019" / "中国高尿酸血症与痛风诊疗指南(2019).html"
    if not html_path.exists():
        print(f"[convert] 找不到源文件：{html_path}")
        return 1
    text_2019 = _cut_html_tail(html_to_text(html_path))
    out_2019 = RAW_DIR / "hyperuricemia_gout_2019.txt"
    out_2019.write_text(text_2019, encoding="utf-8")
    print(f"[convert] 2019 HTML -> {out_2019.name}：{len(text_2019)} 字符，{text_2019.count(chr(10)) + 1} 行")

    # 2024 版：两个 PDF
    pdf_dir = SRC_DIR / "hyperuricemia_gout_2024"
    part1 = pdf_dir / "中国高尿酸血症与痛风诊疗指南（2024） 第一部分：对一般痛风患者的建议.pdf"
    part2 = pdf_dir / "中国高尿酸血症与痛风诊疗指南(2024)　第二部分：常见合并症患者的治疗建议.pdf"
    missing = [p.name for p in (part1, part2) if not p.exists()]
    if missing:
        print(f"[convert] 找不到源文件：{missing}")
        return 1
    text_2024 = _strip_pdf_noise(pdf_to_text(
        [part1, part2],
        [
            "中国高尿酸血症与痛风诊疗指南（2024）第一部分：对一般痛风患者的建议",
            "中国高尿酸血症与痛风诊疗指南（2024）第二部分：常见合并症患者的治疗建议",
        ],
    ))
    out_2024 = RAW_DIR / "hyperuricemia_gout_2024.txt"
    out_2024.write_text(text_2024, encoding="utf-8")
    print(f"[convert] 2024 PDF  -> {out_2024.name}：{len(text_2024)} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
