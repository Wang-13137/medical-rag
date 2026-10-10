# -*- coding: utf-8 -*-
"""切片层：标题层级优先，过长的正文段落再走滑窗。

设计要点
--------
1. 先用规则识别小标题行（Markdown #、中文章节编号、数字编号等），
   把文档切成若干 (标题路径, 正文) 的 section。
2. section 正文长度不超过 chunk_size 时整块作为一个切片，
   超过时按字符滑窗，相邻切片保留 chunk_overlap 个字符重叠。
3. 每个切片携带 doc_name / source_path / page / heading_path / 字符区间，
   这些字段是后续引用溯源的唯一依据，不允许丢失。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .ingest import Document

# ---------------------------------------------------------------- 标题识别规则

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_CN_CHAPTER = re.compile(r"^第\s*[零一二三四五六七八九十百千0-9]+\s*[章节篇编部]\s*[:：、.\s]*\S{0,50}$")
_CN_NUMBER = re.compile(r"^[（(]\s*[零一二三四五六七八九十0-9]+\s*[)）]\s*\S{1,50}$")
_CN_DOTTED = re.compile(r"^[一二三四五六七八九十]{1,3}\s*[、.．]\s*\S{1,50}$")
_NUM_DOTTED = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}[、.．]?\s+\S{1,50}$")
_NUM_PAREN = re.compile(r"^[（(]\s*\d{1,3}\s*[)）]\s*\S{1,50}$")

# 标题行通常以这些标点收尾才说明它是正文而不是标题
_SENTENCE_ENDINGS = ("。", "；", "，", ",", "；", "：", "?", "？", "!", "！")


def detect_heading(line: str, max_chars: int = 60) -> Optional[tuple[int, str]]:
    """判断一行是否为小标题。

    返回 (层级, 标题正文)，层级从 1 开始；不是标题则返回 None。
    heading_max_chars 之外的长行一律视为正文而非标题，避免误伤长段落。
    """
    stripped = line.strip().strip("：:")
    if not stripped or len(stripped) > max_chars:
        return None

    match = _MD_HEADING.match(stripped)
    if match:
        return len(match.group(1)), match.group(2).strip()

    # 其余规则要求该行不以句末标点结尾，否则更像一段正文
    if stripped.endswith(_SENTENCE_ENDINGS):
        return None

    for level, pattern in (
        (1, _CN_CHAPTER),
        (2, _CN_DOTTED),
        (2, _NUM_DOTTED),
        (3, _CN_NUMBER),
        (3, _NUM_PAREN),
    ):
        if pattern.match(stripped):
            # 去掉编号外壳，只留可读标题："第三章 检验科" → "检验科"、"2.1 目标" → "目标"
            title = re.sub(
                r"^第\s*[零一二三四五六七八九十百千0-9]+\s*[章节篇编部]\s*[:：、.．\s]*", "", stripped
            )
            title = re.sub(r"^[（(]?\s*[零一二三四五六七八九十百千0-9]+\s*[)）]?\s*[、.．]?\s*", "", title)
            title = re.sub(r"^\d{1,2}(?:\.\d{1,2}){0,3}[、.．]?\s*", "", title)
            return level, title.strip().strip("：:") or stripped
    return None


@dataclass
class Chunk:
    """一个切片：正文 + 溯源元数据。"""

    chunk_id: str
    text: str
    doc_id: str
    doc_name: str
    source_path: str
    file_type: str
    # 页码；txt / md 无页码概念，为 None
    page: Optional[int]
    heading_path: List[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    # 在本文档所有切片中的序号，从 0 开始
    chunk_index: int = 0

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    def citation(self) -> str:
        """面向用户的溯源串，形如「文件.md P3 · 标题 > 子标题」。"""
        location = f"P{self.page}" if self.page is not None else self.file_type
        title = f" · {self.heading}" if self.heading else ""
        return f"{self.doc_name} {location}{title}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class Section:
    """按标题层级切出来的中间产物。"""

    heading_path: List[str]
    texts: List[str] = field(default_factory=list)
    page: Optional[int] = None

    @property
    def body(self) -> str:
        return "\n".join(self.texts).strip()


def build_sections(document: Document, heading_max_chars: int = 60) -> List[Section]:
    """把文档按标题层级组织成 section 列表。

    标题本身会作为包含它的后续全部切片的 heading_path 前缀，
    这样即使正文被切碎，切片仍知道自己属于哪一小节。
    """
    heading_stack: List[tuple[int, str]] = []
    # 当前尚未归属任何标题的正文归入一个空路径 section
    sections: List[Section] = []
    current = Section(heading_path=[t for _, t in heading_stack])

    def flush() -> None:
        nonlocal current
        if current.body:
            sections.append(current)
        current = Section(heading_path=[title for _, title in heading_stack])

    for line_block in document.lines:
        line = line_block.text.strip()
        if not line:
            continue
        detected = detect_heading(line, max_chars=heading_max_chars)
        if detected is not None:
            level, title = detected
            # 弹掉层级 >= 当前层级的旧标题，维护祖先链
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            flush()
            continue
        if current.page is None:
            current.page = line_block.page
        current.texts.append(line)

    flush()
    return sections


def _sliding_windows(text: str, chunk_size: int, overlap: int, min_tail_chars: int) -> List[tuple[int, int]]:
    """生成字符级滑窗区间列表 [(start, end)]。

    步长 = chunk_size - overlap，因此相邻窗口天然重叠 overlap 个字符。
    最后一窗若余料短于 min_tail_chars 会被上一窗吸收，避免出现碎片切片。
    """
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"chunk_overlap({overlap}) 必须小于 chunk_size({chunk_size})")

    windows: List[tuple[int, int]] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_size, total)
        remaining = total - end
        # 余料太少则直接吃进当前窗口并结束
        if 0 < remaining < min_tail_chars:
            end = total
            windows.append((start, end))
            break
        windows.append((start, end))
        # 窗口触及末尾即停止，避免产出空切片
        if end >= total:
            break
        start += step
    return windows


def _make_chunk(
    document: Document,
    section: Section,
    body: str,
    char_start: int,
    char_end: int,
    chunk_index: int,
    prepend_heading_path: bool,
) -> Chunk:
    heading_path = list(section.heading_path)
    text = body
    if prepend_heading_path and heading_path:
        text = f"{' > '.join(heading_path)}\n{body}"
    raw = f"{document.doc_id}|{char_start}|{char_end}|{body[:64]}"
    return Chunk(
        chunk_id=hashlib.md5(raw.encode("utf-8")).hexdigest()[:16],
        text=text.strip(),
        doc_id=document.doc_id,
        doc_name=document.doc_name,
        source_path=document.source_path,
        file_type=document.file_type,
        page=section.page,
        heading_path=heading_path,
        char_start=char_start,
        char_end=char_end,
        chunk_index=chunk_index,
    )


def chunk_document(
    document: Document,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    min_tail_chars: int = 30,
    heading_max_chars: int = 60,
    prepend_heading_path: bool = True,
) -> List[Chunk]:
    """单文档切片：先标题再滑窗。"""
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")

    chunks: List[Chunk] = []
    for section in build_sections(document, heading_max_chars=heading_max_chars):
        body = section.body
        if not body:
            continue
        if len(body) <= chunk_size:
            chunks.append(
                _make_chunk(document, section, body, 0, len(body), len(chunks), prepend_heading_path)
            )
            continue
        for char_start, char_end in _sliding_windows(body, chunk_size, chunk_overlap, min_tail_chars):
            chunks.append(
                _make_chunk(
                    document,
                    section,
                    body[char_start:char_end],
                    char_start,
                    char_end,
                    len(chunks),
                    prepend_heading_path,
                )
            )
    return chunks


def chunk_documents(documents: Sequence[Document], **kwargs) -> List[Chunk]:
    """批量切片，chunk_index 按文档重新从 0 计数。"""
    all_chunks: List[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, **kwargs))
    return all_chunks


def chunk_documents_from_config(documents: Sequence[Document], cfg) -> List[Chunk]:
    """从配置对象取参数，避免脚本层重复拼参数。"""
    return chunk_documents(
        documents,
        chunk_size=int(cfg.get("chunking.chunk_size", 400)),
        chunk_overlap=int(cfg.get("chunking.chunk_overlap", 80)),
        min_tail_chars=int(cfg.get("chunking.min_tail_chars", 30)),
        heading_max_chars=int(cfg.get("chunking.heading_max_chars", 60)),
        prepend_heading_path=bool(cfg.get("chunking.prepend_heading_path", True)),
    )
