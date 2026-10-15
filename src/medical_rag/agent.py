"""ReAct 风格的检索智能体（W5）。

设计取舍：
1. 主循环手写，不引入 LangGraph / CrewAI —— 这样每一步的工具调用、预算消耗、
   解析失败都能被单元测试精确覆盖，也便于回答「ReAct 到底多了什么开销」。
2. 工具是检索系统已有的能力（向量检索 / BM25 / 精排 / 蕴含校验），
   Agent 只负责决定「什么时候用哪一个」，不重复实现检索逻辑。
3. 模型只输出 Thought / Action / Action Input 三段纯文本，不用 JSON mode，
   这样换任何聊天模型都能跑；解析失败时把格式要求再喂回去，而不是直接崩。
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from .generate import parse_citations
from .verify import verify_answer

REFUSAL_TEXT = "根据提供的指南片段无法回答该问题。"

ACTION_RE = re.compile(r"行动\s*[:：]\s*(.+)")
INPUT_RE = re.compile(r"行动输入\s*[:：]\s*(.+)", re.S)
# 思考只取本行，不能吃掉后面的「行动」段
THOUGHT_RE = re.compile(r"思考\s*[:：]\s*(.+)")


# --------------------------------------------------------------------- 数据结构


@dataclass
class ToolResult:
    """工具返回的观察结果。"""

    text: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    args_hint: str
    func: Callable[..., ToolResult]


@dataclass
class Evidence:
    """工作集里的一条片段。"""

    eid: int
    text: str
    title: str
    score: float = 0.0
    origin: str = "vector_search"
    selected: bool = False

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class Step:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    """一次 Agent 运行的完整轨迹，可直接落 JSONL 供评测复用。"""

    query: str
    answer: str = ""
    refused: bool = False
    steps: List[Step] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)
    tool_calls: Dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""
    budget_chars: int = 0
    used_chars: int = 0
    latency_sec: float = 0.0
    llm_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            # 与 run_generation.py 对齐：引用编号从答案里解析，供引用精确率统计
            "citations": parse_citations(self.answer, max_index=len(self.contexts)),
            "refused": self.refused,
            "steps": [s.to_dict() for s in self.steps],
            "contexts": self.contexts,
            "titles": self.titles,
            "notes": self.notes,
            "tool_calls": self.tool_calls,
            "stop_reason": self.stop_reason,
            "num_steps": len(self.steps),
            "llm_calls": self.llm_calls,
            "tool_calls_total": sum(self.tool_calls.values()),
            "budget_chars": self.budget_chars,
            "used_chars": self.used_chars,
            "latency_sec": round(self.latency_sec, 3),
        }


# --------------------------------------------------------------------- 解析


def parse_action(text: str) -> tuple[str, str]:
    """从模型输出里抽出 (工具名, 输入)。解析不出工具名则返回 ("", "")."""
    m = ACTION_RE.search(text or "")
    if not m:
        return "", ""
    name = m.group(1).strip().strip("`").strip()
    # "行动：vector_search(query=...)" 这种一次写全的写法也要认
    if "(" in name:
        head, _, rest = name.partition("(")
        return head.strip(), rest.rstrip(")").strip()
    im = INPUT_RE.search(text or "")
    return name, (im.group(1).strip() if im else "")


def parse_thought(text: str) -> str:
    m = THOUGHT_RE.search(text or "")
    return m.group(1).strip() if m else ""


def _align_kwargs(func: Callable[..., ToolResult], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """模型经常把「行动输入」写成裸文本（不带 answer=/ids=）。

    此时 parse_kwargs 只能给出 {"query": ...}，直接调用会 TypeError。
    这里做一次兜底：参数名对不上时，把唯一的值映射到工具的第一个形参。
    """
    if not kwargs:
        return kwargs
    params = [p for p in inspect.signature(func).parameters]
    if all(k in params for k in kwargs):
        return kwargs
    value = kwargs.get("query", next(iter(kwargs.values())))
    return {params[0]: value}


def parse_kwargs(raw: str) -> Dict[str, Any]:
    """把 'query=高血压, top_k=8' 或 '高血压' 或 JSON 都解析成参数字典。"""
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"query": str(data)}
        except json.JSONDecodeError:
            pass
    # key=value 形式：只在「下一个 key=」处切分，值内部的逗号要保留（如 ids=1,2,3）
    marks = list(re.finditer(r"([A-Za-z_]\w*)\s*=", raw))
    if marks:
        out: Dict[str, Any] = {}
        for i, m in enumerate(marks):
            start = m.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            value = raw[start:end].strip().strip("'\"").strip("，,").strip()
            out[m.group(1)] = int(value) if value.lstrip("-").isdigit() else value
        return out
    return {"query": raw.strip("'\"")}


# --------------------------------------------------------------------- 提示词

SYSTEM_TEMPLATE = """你是中文临床指南问答助手，只能依据工具返回的指南片段作答。

可用工具：
{tool_lines}

严格按下面的格式回复，一次只输出一个行动：
思考：<你要做什么、为什么>
行动：<工具名>
行动输入：<参数>

规则：
- 先用检索工具找片段，再用 read 把要引用的片段读全，最后用 finish 给答案。
- 答案里每个事实后紧跟[n]引用编号，n 必须是 context 里的编号。
- 参数必须写成 key=value 形式：finish/verify 用 answer=...，read 用 ids=1,3。
- 片段里没有依据时，直接 finish 输出「{refusal}」，不要编造。
- verify 工具会逐句检查依据；它报出无依据的句子时必须改写或拒答。
"""


def build_system_prompt(tools: Sequence[Tool], refusal_text: str = REFUSAL_TEXT) -> str:
    lines = [f"- {t.name}（{t.args_hint}）：{t.description}" for t in tools]
    return SYSTEM_TEMPLATE.format(tool_lines="\n".join(lines), refusal=refusal_text)


# --------------------------------------------------------------------- Agent


class ReActAgent:
    """手写 ReAct 主循环。chat_fn 只需满足 (messages) -> str。"""

    def __init__(
        self,
        chat_fn: Callable[[List[Dict[str, str]]], str],
        tools: Sequence[Tool],
        max_steps: int = 6,
        budget_chars: int = 2400,
        max_parse_retries: int = 2,
        refusal_text: str = REFUSAL_TEXT,
        min_read: int = 0,
    ) -> None:
        self.chat_fn = chat_fn
        self.tools = {t.name: t for t in tools}
        self.tool_order = [t.name for t in tools]
        self.max_steps = max_steps
        self.budget_chars = budget_chars
        self.max_parse_retries = max_parse_retries
        self.refusal_text = refusal_text
        # 欠召回护栏：finish 之前至少要读过 min_read 条（拒答不受此限）
        self.min_read = min_read
        self.evidence: List[Evidence] = []
        self.notes: Dict[str, str] = {}

    # -- 工作集 ---------------------------------------------------------

    def _add_evidence(self, items: Sequence[Dict[str, Any]], origin: str) -> List[int]:
        seen = {e.text: e.eid for e in self.evidence}
        ids: List[int] = []
        for it in items:
            text = (it.get("text") or "").strip()
            if not text or text in seen:
                continue
            eid = len(self.evidence) + 1
            self.evidence.append(
                Evidence(
                    eid=eid,
                    text=text,
                    title=it.get("title") or "",
                    score=float(it.get("score") or 0.0),
                    origin=origin,
                )
            )
            seen[text] = eid
            ids.append(eid)
        return ids

    def _contexts(self) -> List[str]:
        """按预算顺序取已读片段的正文，超预算即停。"""
        texts: List[str] = []
        used = 0
        for e in self.evidence:
            if not e.selected:
                continue
            if used + e.length > self.budget_chars and texts:
                break
            texts.append(e.text)
            used += e.length
        return texts

    def _titles(self) -> List[str]:
        return [e.title for e in self.evidence if e.selected][: len(self._contexts())]

    # -- 主循环 ---------------------------------------------------------

    def run(self, query: str) -> AgentRun:
        started = time.time()
        self.evidence = []
        run = AgentRun(query=query, budget_chars=self.budget_chars)
        system = build_system_prompt(list(self.tools.values()), self.refusal_text)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{query}"},
        ]
        parse_fails = 0

        for _ in range(self.max_steps):
            raw = self.chat_fn(messages)
            run.llm_calls += 1
            step = Step(thought=parse_thought(raw))
            name, raw_input = parse_action(raw)

            if not name:
                parse_fails += 1
                step.error = "未解析出行动"
                step.observation = "格式不对，请严格按「行动：<工具名> / 行动输入：<参数>」输出。"
                run.steps.append(step)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": step.observation})
                if parse_fails > self.max_parse_retries:
                    run.stop_reason = "parse_failed"
                    break
                continue

            step.action = name
            step.action_input = raw_input
            run.tool_calls[name] = run.tool_calls.get(name, 0) + 1

            if name not in self.tools:
                step.error = "未知工具"
                step.observation = f"没有这个工具，可用：{', '.join(self.tool_order)}"
                run.steps.append(step)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": step.observation})
                continue

            kwargs = parse_kwargs(raw_input)
            kwargs = _align_kwargs(self.tools[name].func, kwargs)
            try:
                result = self.tools[name].func(**kwargs)
            except TypeError as exc:
                result = ToolResult(text=f"参数不对：{exc}", payload={"error": str(exc)})
            step.observation = result.text
            run.steps.append(step)

            if name == "finish":
                answer = (result.payload or {}).get("answer", "")
                gate = self._read_gate(answer)
                if gate:
                    # 还没读够：把答案退回，让模型补读，不算完成
                    step.observation = gate
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"观察：{gate}"})
                    continue
                run.answer = answer
                run.refused = self.refusal_text.replace(" ", "") in answer.replace(" ", "")
                run.stop_reason = "finish"
                break

            # 非终止工具：把观察回喂给模型，并附上当前 context 编号
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"观察：{result.text}\n\n当前 context（引用编号即此顺序）：\n"
                    + self._context_block(),
                }
            )
        else:
            run.stop_reason = "max_steps"

        run.contexts = self._contexts()
        run.titles = self._titles()
        run.notes = dict(self.notes)
        run.used_chars = sum(len(c) for c in run.contexts)
        if run.stop_reason == "max_steps" and not run.answer:
            run.answer = self.refusal_text
            run.refused = True
        run.latency_sec = time.time() - started
        return run

    def _read_gate(self, answer: str) -> str:
        """finish 前的欠召回护栏。返回空串表示放行。"""
        if self.min_read <= 0:
            return ""
        compact = (answer or "").replace(" ", "")
        if self.refusal_text.replace(" ", "") in compact:
            return ""  # 拒答是合理终局，不强求继续读
        selected = [e for e in self.evidence if e.selected]
        if len(selected) >= self.min_read:
            return ""
        unread = [e.eid for e in self.evidence if not e.selected]
        if not unread:
            return ""
        return (
            f"你只读了 {len(selected)} 条片段，未达下限 {self.min_read} 条，"
            f"请用 read 补读（未读编号：{unread[:8]}）后再 finish。"
        )

    def _context_block(self) -> str:
        ctx = self._contexts()
        if not ctx:
            return "（空：还没有 read 过任何片段）"
        return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(ctx, start=1))


# --------------------------------------------------------------------- 工具构造


def build_tools(
    retriever: Any,
    keyword_search: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    reranker: Any = None,
    budget_chars: int = 2400,
    preview_chars: int = 120,
) -> tuple[List[Tool], Dict[str, Any]]:
    """把检索系统已有的能力包装成 Agent 工具。

    retriever 需要实现 retrieve(query, top_k) -> List[SearchHit]。
    """
    state = {"evidence_owner": None}  # run 时由 agent 注入

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

    def _call_vector(query: str = "", top_k: int = 8) -> ToolResult:
        if not query:
            return ToolResult("缺少 query 参数")
        hits = retriever.retrieve(query, top_k=int(top_k))
        ids = state["evidence_owner"]._add_evidence(_hits_to_items(hits), "vector_search")
        preview = _preview(state["evidence_owner"].evidence, ids, preview_chars)
        return ToolResult(f"命中 {len(ids)} 条，编号 {ids}：\n{preview}", {"ids": ids})

    def _call_keyword(query: str = "", top_k: int = 8) -> ToolResult:
        if keyword_search is None:
            return ToolResult("BM25 未启用（配置 retrieval.use_hybrid=false）")
        if not query:
            return ToolResult("缺少 query 参数")
        items = keyword_search(query, int(top_k))
        ids = state["evidence_owner"]._add_evidence(items, "keyword_search")
        preview = _preview(state["evidence_owner"].evidence, ids, preview_chars)
        return ToolResult(f"命中 {len(ids)} 条，编号 {ids}：\n{preview}", {"ids": ids})

    def _call_rerank(query: str = "", top_k: int = 8) -> ToolResult:
        if reranker is None:
            return ToolResult("精排模型未加载")
        owner = state["evidence_owner"]
        pool = [e.text for e in owner.evidence]
        if not pool:
            return ToolResult("工作集是空的，先检索")
        scores = [float(s) for s in reranker.score(query or pool[0], pool)]
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        for e, i in zip(owner.evidence, range(len(pool))):
            e.score = scores[i]
        owner.evidence = [owner.evidence[i] for i in order]
        top = [e.eid for e in owner.evidence[: int(top_k)]]
        return ToolResult(f"已按相关性重排，前 {len(top)} 条：{top}", {"ids": top})

    def _call_read(ids: str = "") -> ToolResult:
        owner = state["evidence_owner"]
        picked: List[int] = []
        for tok in re.split(r"[，,、\s]+", str(ids)):
            tok = tok.strip("[]#")
            if tok.lstrip("-").isdigit():
                picked.append(int(tok))
        if not picked:
            return ToolResult("请给要读的编号，例如 ids=1,3,5")
        hit = []
        for eid in picked:
            for e in owner.evidence:
                if e.eid == eid:
                    e.selected = True
                    hit.append(eid)
                    break
        return ToolResult(f"已读入 {hit}（context 现有 {len(owner._contexts())} 条）", {"ids": hit})

    def _call_remember(key: str = "", value: str = "") -> ToolResult:
        if not key:
            return ToolResult("缺少 key")
        state["evidence_owner"].notes[key] = value
        return ToolResult(f"已记住 {key}")

    def _call_verify(answer: str = "") -> ToolResult:
        owner = state["evidence_owner"]
        ctx = owner._contexts()
        if not answer:
            return ToolResult("缺少待检查的答案")
        result = verify_answer(answer, ctx, mode="flag")
        bad = result.unsupported_sentences
        if not bad:
            return ToolResult("全部句子都有依据（可用 finish 输出）", {"unsupported": []})
        return ToolResult(
            "以下句子在被引用片段里找不到依据，请改写或拒答：" + "；".join(bad[:5]),
            {"unsupported": bad},
        )

    def _call_finish(answer: str = "") -> ToolResult:
        return ToolResult("完成", {"answer": answer})

    tools = [
        Tool("vector_search", "语义检索指南片段，返回编号与摘要", "query, top_k=8", _call_vector),
        Tool("keyword_search", "BM25 关键词检索，适合找专有名词与剂量数字", "query, top_k=8", _call_keyword),
        Tool("rerank", "对已召回的工作集按相关性重排", "query, top_k=8", _call_rerank),
        Tool("read", "把指定编号的片段读进 context（必须做，否则无法引用）", "ids=1,3,5", _call_read),
        Tool("remember", "记一条中间结论，key=value", "key, value", _call_remember),
        Tool("verify", "检查草稿答案的每一句是否有依据", "answer", _call_verify),
        Tool("finish", "输出最终答案；无依据时输出拒答句", "answer", _call_finish),
    ]
    return tools, state


def _preview(evidence: Sequence[Evidence], ids: Sequence[int], preview_chars: int) -> str:
    lines = []
    for eid in ids:
        for e in evidence:
            if e.eid == eid:
                head = f"#{e.eid} {e.title}" if e.title else f"#{e.eid}"
                lines.append(f"{head}：{e.text[:preview_chars]}")
                break
    return "\n".join(lines)


def make_agent(
    chat_fn: Callable[[List[Dict[str, str]]], str],
    retriever: Any,
    keyword_search: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    reranker: Any = None,
    max_steps: int = 6,
    budget_chars: int = 2400,
    max_parse_retries: int = 2,
    min_read: int = 0,
) -> tuple[ReActAgent, Dict[str, Any]]:
    """构造 Agent 并把工作集注入工具闭包。"""
    tools, state = build_tools(
        retriever,
        keyword_search=keyword_search,
        reranker=reranker,
        budget_chars=budget_chars,
    )
    agent = ReActAgent(
        chat_fn,
        tools,
        max_steps=max_steps,
        budget_chars=budget_chars,
        max_parse_retries=max_parse_retries,
        min_read=min_read,
    )
    state["evidence_owner"] = agent
    return agent, state
