#!/usr/bin/env python
"""Gradio 演示界面：把「检索 -> 精排 -> 带引用答案 -> 校验闸门」这条链可视化。

四个面板的设计意图：
1. 答案：保留 [n] 引用编号，和原文面板对照看。
2. 检索明细：把「召回名次」和「精排后名次」并排，解释精排到底改了什么。
3. 闸门判决：逐句显示有没有依据、字面覆盖率、哪些数字没溯源到——把 W4 的负结果摊开给人看。
4. Agent 轨迹：列出 ReAct 每一步的工具、参数、观察，解释它为什么省上下文却召回更低。

用法：
    python scripts/gradio_demo.py                 # 本地 http://127.0.0.1:7860
    python scripts/gradio_demo.py --share         # 生成公网临时链接
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.config import load_config  # noqa: E402
from medical_rag.generate import DeepSeekGenerator, EchoGenerator, load_generator  # noqa: E402
from medical_rag.verify import verify_answer  # noqa: E402
from eval_benchmark import make_retriever  # noqa: E402
from run_generation import build_contexts  # noqa: E402

MODE_PIPELINE_STRICT = "流水线（受限提示词）"
MODE_PIPELINE_NAIVE = "流水线（无约束对照）"
MODE_AGENT = "ReAct 智能体"
MODE_AGENT_GUARD = "ReAct 智能体 + 欠召回护栏"

_RESOURCES: Dict[str, Any] = {}


# ------------------------------------------------------------------ 资源


def load_resources(offline: bool) -> Dict[str, Any]:
    """索引与模型只加载一次：精排模型 2.2GB，不能每次提问都付一遍。"""
    key = "offline" if offline else "api"
    if key in _RESOURCES:
        return _RESOURCES[key]

    cfg = load_config()
    candidate_k = int(cfg.get("rerank.candidate_k", 20))
    mode, retriever = make_retriever(cfg, True, candidate_k, verbose=True)

    generator: Any
    note = ""
    if offline:
        generator = EchoGenerator()
    else:
        try:
            generator = load_generator(cfg=cfg, verbose=True)
        except Exception as exc:  # 缺 Key 不能让界面起不来
            generator = EchoGenerator()
            note = f"模型加载失败，已退回离线替身：{exc}"

    pack = {
        "cfg": cfg,
        "mode": mode,
        "retriever": retriever,
        "generator": generator,
        "candidate_k": candidate_k,
        "note": note,
    }
    _RESOURCES[key] = pack
    return pack


# ------------------------------------------------------------------ 渲染


def render_sources(contexts: List[str], titles: List[str]) -> str:
    if not contexts:
        return "_没有进入上下文的片段。_"
    lines = []
    for i, (title, text) in enumerate(zip(titles, contexts), start=1):
        head = title or "（无溯源标题）"
        body = text.strip().replace("\n", " ")
        if len(body) > 320:
            body = body[:320] + "…"
        lines.append(f"**[{i}]** {head}\n\n> {body}\n")
    return "\n".join(lines)


def render_retrieval_table(
    hits: List[Any],
    top_n: int,
    before_rank: Optional[Dict[int, int]] = None,
) -> List[List[Any]]:
    """名次 / 分数 / 精排前名次 / 文档 / 章节。

    精排前的名次由 `RerankRetriever.explain()` 提供（按 id(chunk) 索引），
    拿不到就标 —，不做假数据。
    """
    rows: List[List[Any]] = []
    for rank, h in enumerate(hits[: max(top_n, 8)], start=1):
        chunk = h.chunk
        before = (before_rank or {}).get(id(chunk))
        if before is None:
            before = getattr(h, "first_stage_rank", None)
        head = " > ".join(chunk.heading_path or [])[:28]
        rows.append(
            [
                rank,
                round(float(getattr(h, "score", 0.0) or 0.0), 4),
                before if before is not None else "—",
                chunk.doc_name[:28],
                head or "—",
            ]
        )
    return rows


def render_verdicts(answer: str, contexts: List[str]) -> List[List[Any]]:
    if not answer.strip():
        return []
    result = verify_answer(answer, contexts, mode="flag")
    rows = []
    for v in result.verdicts:
        rows.append(
            [
                v.text[:40],
                "有依据" if v.supported else "**无依据**",
                round(v.coverage, 3),
                "、".join(v.unsupported_numbers) or "—",
                "引用偏弱" if v.weak_citation else "—",
            ]
        )
    return rows


def render_agent_trace(steps: List[Any]) -> str:
    lines = []
    for i, s in enumerate(steps, start=1):
        obs = (s.observation or "").replace("\n", " ")[:120]
        lines.append(
            f"**第 {i} 步 · `{s.action or '未解析'}`**\n\n"
            f"- 输入：{(s.action_input or '—')[:80]}\n"
            f"- 观察：{obs or '—'}\n"
        )
    return "\n".join(lines) or "_没有轨迹。_"


# ------------------------------------------------------------------ 主流程


def answer_once(
    question: str,
    mode: str,
    top_n: int,
    budget: int,
    min_read: int,
    offline: bool,
) -> Tuple[str, str, List[List[Any]], List[List[Any]], str, str]:
    """返回六个输出：答案、引用原文、检索明细、逐句判决、Agent 轨迹、成本。"""
    question = (question or "").strip()
    if not question:
        return "请输入问题。", "", [], [], "", ""

    is_agent = mode in (MODE_AGENT, MODE_AGENT_GUARD)
    try:
        pack = load_resources(offline)
    except Exception as exc:
        return f"资源加载失败：{exc}", "", [], [], "", ""

    retriever, generator = pack["retriever"], pack["generator"]
    contexts: List[str] = []
    titles: List[str] = []
    answer = ""
    hits: List[Any] = []
    trace = "_流水线模式没有工具轨迹。_"
    cost_lines: List[str] = []

    try:
        if is_agent:
            from medical_rag.agent import make_agent
            from run_agent import offline_policy

            use_api = (not offline) and isinstance(generator, DeepSeekGenerator)
            chat_fn = generator.chat if use_api else offline_policy
            guard = min_read if mode == MODE_AGENT_GUARD else 0
            agent, _ = make_agent(
                chat_fn,
                retriever,
                reranker=getattr(retriever, "reranker", None),
                max_steps=8,
                budget_chars=int(budget),
                min_read=int(guard),
            )
            run = agent.run(question)
            contexts, titles, answer = run.contexts, run.titles, run.answer
            trace = render_agent_trace(run.steps)
            cost_lines = [
                f"- 步数：{len(run.steps)}（结束原因 {run.stop_reason}）",
                f"- 工具调用：{sum(run.tool_calls.values())} 次 {dict(run.tool_calls)}",
                f"- 模型调用：{run.llm_calls} 次",
                f"- 上下文：{run.used_chars} 字 / {len(contexts)} 片（预算 {budget}）",
                f"- 延迟：{run.latency_sec:.1f} s",
            ]
            if not use_api:
                cost_lines.append("- 生成：离线替身（未调用 API，答案不代表模型能力）")
        else:
            prompt_mode = "naive" if mode == MODE_PIPELINE_NAIVE else "strict"
            hits = retriever.retrieve(question, top_k=max(pack["candidate_k"], int(top_n)))
            before_rank: Optional[Dict[int, int]] = None
            try:
                before_rank = {
                    id(row_chunk): row["first_stage_rank"]
                    for row in retriever.explain(question, top_k=len(hits))
                    for row_chunk in [h.chunk for h in hits if h.chunk.text == row["text"]]
                }
            except Exception:
                before_rank = None
            contexts, titles, used = build_contexts(hits, int(budget), int(top_n))
            res = generator.answer(question, contexts, titles, prompt_mode)
            answer = res.answer
            cost_lines = [
                f"- 检索：{pack['mode']}",
                f"- 生成：{type(generator).__name__} / 提示词 {prompt_mode}",
                f"- 上下文：{used} 字 / {len(contexts)} 片（预算 {budget}）",
                f"- 引用编号：{res.citations}",
                f"- 延迟：{res.latency_sec:.1f} s",
            ]
    except Exception as exc:
        return f"运行出错：{exc}", "", [], [], "", ""

    if pack["note"]:
        cost_lines.append(f"- 提示：{pack['note']}")

    return (
        answer,
        render_sources(contexts, titles),
        render_retrieval_table(hits, int(top_n), before_rank) if hits else [],
        render_verdicts(answer, contexts),
        trace,
        "\n".join(cost_lines),
    )


# ------------------------------------------------------------------ 界面


def load_examples(path: str = "data/eval/benchmark_v1.json") -> List[List[str]]:
    """样例里特意混入应拒答题，用来演示拒答行为。"""
    p = ROOT / path
    if not p.exists():
        return []
    items = json.load(open(p, encoding="utf-8"))["items"]
    answerable = [x for x in items if x.get("type") != "unanswerable"][:3]
    unanswerable = [x for x in items if x.get("type") == "unanswerable"][:2]
    return [[x["question"]] for x in answerable + unanswerable]


def build_demo():
    import gradio as gr

    with gr.Blocks(title="中文临床指南 RAG 演示") as demo:
        gr.Markdown(
            "# 中文临床指南检索增强问答 · 演示\n"
            "左侧提问，右侧回答四个问题：答案从哪来、谁排到前面、每句话有没有依据、Agent 走了哪几步。"
        )
        with gr.Row():
            with gr.Column(scale=4):
                question = gr.Textbox(
                    label="问题", lines=3,
                    placeholder="例如：妊娠合并急性阑尾炎首选手术还是保守治疗？",
                )
                mode = gr.Radio(
                    [MODE_PIPELINE_STRICT, MODE_PIPELINE_NAIVE, MODE_AGENT, MODE_AGENT_GUARD],
                    value=MODE_PIPELINE_STRICT,
                    label="运行方式",
                )
                with gr.Row():
                    top_n = gr.Slider(2, 12, value=8, step=1, label="上下文片段数上限")
                    budget = gr.Slider(800, 4000, value=2400, step=200, label="上下文字符预算")
                min_read = gr.Slider(0, 8, value=5, step=1, label="Agent 最少读入片段数（护栏）")
                offline = gr.Checkbox(value=False, label="不调用 API（离线替身，仅验证链路）")
                submit = gr.Button("提问", variant="primary")
                gr.Examples(load_examples(), inputs=[question], label="样例问题（含 2 条应拒答题）")

            with gr.Column(scale=6):
                answer_md = gr.Markdown()
                cost_md = gr.Markdown()
                with gr.Tabs():
                    with gr.TabItem("引用原文"):
                        sources_md = gr.Markdown()
                    with gr.TabItem("逐句依据校验"):
                        verdict_tbl = gr.Dataframe(
                            headers=["句子", "判定", "字面覆盖率", "未溯源数字", "引用"],
                            label="W4 蕴含校验闸门",
                        )
                    with gr.TabItem("检索明细"):
                        retrieval_tbl = gr.Dataframe(
                            headers=["精排后名次", "分数", "精排前名次", "文档", "章节"],
                            label="流水线模式下才显示",
                        )
                    with gr.TabItem("Agent 轨迹"):
                        trace_md = gr.Markdown()

        outputs = [answer_md, sources_md, retrieval_tbl, verdict_tbl, trace_md, cost_md]
        inputs = [question, mode, top_n, budget, min_read, offline]
        submit.click(answer_once, inputs=inputs, outputs=outputs)
        question.submit(answer_once, inputs=inputs, outputs=outputs)
    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="Gradio 演示")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
