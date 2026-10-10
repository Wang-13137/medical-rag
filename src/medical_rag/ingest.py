# -*- coding: utf-8 -*-
"""文档解析层：把 PDF / docx / txt / md 统一成带页码信息的行序列。

产出的 Document 只负责"忠实还原文本的行结构与页码"，不做切分，
结构性判断交给 chunking.py，避免解析层与策略层耦合。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}

# 中文文本常见编码，逐个尝试，全失败则抛出
_CANDIDATE_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5")


@dataclass
class LineBlock:
    """一行文本及其所在页码。纯文本与 Markdown 文件没有页码概念，记为 None。"""

    text: str
    page: Optional[int] = None


@dataclass
class Document:
    """一份被解析后的文档。"""

    doc_id: str          # 去掉扩展名的文件名，用作稳定标识
    doc_name: str        # 带扩展名的文件名
    source_path: str     # 磁盘绝对路径，用于溯源
    file_type: str       # pdf / docx / txt / md
    lines: List[LineBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def __len__(self) -> int:
        return len(self.text)


def _read_text_file(path: Path) -> str:
    """带编码回退地读取纯文本文件。"""
    raw = path.read_bytes()
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # 最后兜底：忽略无法解码的字节，保证不因为单个坏字符中断整批入库
    return raw.decode("utf-8", errors="ignore")


def _split_paragraph_to_lines(text: str) -> List[str]:
    """把段落文本拆行：先按换行，再按句末标点软换行便于后续均匀切窗。

    软换行只起排版作用，不影响语义，切窗后仍是完整句子。
    """
    lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lines.extend(_soft_wrap(stripped))
    return lines


_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")


def _soft_wrap(line: str, width: int = 120) -> List[str]:
    """长行按句末标点软换行，短行原样保留；标题类短行不被拆。"""
    if len(line) <= width:
        return [line]
    pieces = _SENTENCE_END.split(line)
    out: List[str] = []
    buffer = ""
    for piece in pieces:
        if not piece:
            continue
        buffer += piece
        if len(buffer) >= width * 0.6:
            out.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        out.append(buffer.strip())
    return out or [line]


def parse_txt(path: Path) -> Document:
    """解析 txt / md：整篇视为单"页"，页码为 None。"""
    text = _read_text_file(path)
    file_type = "md" if path.suffix.lower() in {".md", ".markdown"} else "txt"
    return Document(
        doc_id=path.stem,
        doc_name=path.name,
        source_path=str(path.resolve()),
        file_type=file_type,
        lines=[LineBlock(text=line, page=None) for line in _split_paragraph_to_lines(text)],
    )


def parse_docx(path: Path) -> Document:
    """解析 docx：按段落顺序读取，表格行也计入文本。"""
    import docx  # 局部导入，未安装 docx 依赖时不影响其它格式解析

    document = docx.Document(str(path))
    blocks: List[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            joined = " | ".join(cell for cell in cells if cell)
            if joined:
                blocks.append(joined)

    lines: List[LineBlock] = []
    for block in blocks:
        # 段落内部若含多个标题/正文混合，交给 _split_paragraph_to_lines 再拆一次
        for piece in _split_paragraph_to_lines(block):
            lines.append(LineBlock(text=piece, page=None))
    return Document(
        doc_id=path.stem,
        doc_name=path.name,
        source_path=str(path.resolve()),
        file_type="docx",
        lines=lines,
    )


def parse_pdf(path: Path) -> Document:
    """解析 PDF：pdfplumber 逐页取文本，保留页码用于溯源。"""
    import pdfplumber  # 局部导入，同上

    lines: List[LineBlock] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            for piece in _split_paragraph_to_lines(page_text):
                lines.append(LineBlock(text=piece, page=page_no))
    return Document(
        doc_id=path.stem,
        doc_name=path.name,
        source_path=str(path.resolve()),
        file_type="pdf",
        lines=lines,
    )


_PARSERS = {
    ".txt": parse_txt,
    ".md": parse_txt,
    ".markdown": parse_txt,
    ".docx": parse_docx,
    ".pdf": parse_pdf,
}


def parse_file(path: Path) -> Document:
    """按后缀分派解析器。"""
    suffix = path.suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(f"不支持的文件类型：{path}（受支持：{sorted(SUPPORTED_SUFFIXES)}）")
    return _PARSERS[suffix](path)


def iter_source_files(root: Path) -> Iterable[Path]:
    """递归遍历目录下的受支持文档，按路径排序保证结果可复现。"""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_documents(source: Path | str, suffixes: Sequence[str] = tuple(SUPPORTED_SUFFIXES)) -> List[Document]:
    """加载单文件或整个目录，返回 Document 列表；空目录返回空列表而不报错。"""
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"数据路径不存在：{root}")

    files = [root] if root.is_file() else list(iter_source_files(root))
    files = [f for f in files if f.suffix.lower() in set(suffixes)]

    documents: List[Document] = []
    for file_path in files:
        try:
            documents.append(parse_file(file_path))
        except Exception as exc:  # 单份文档的解析失败不应中断整批入库
            print(f"[ingest] 跳过 {file_path.name}：{type(exc).__name__}: {exc}")
    return documents
