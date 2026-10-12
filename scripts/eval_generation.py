#!/usr/bin/env python
"""对已生成的答案做评测：正确性、引用可溯源性、语境支持度（幻觉）、拒答权衡。

设计取向：
1. 正确性（gold_key_recall）沿用 benchmark 的关键串口径，和检索端指标同量纲可比。
2. 幻觉用两条互补的路子，不用单一口径下结论：
   - 规则版：答案里的每个数字，必须能在送进去的上下文里找到（NFKC + 去空白后子串匹配）。
     零成本、完全可复现，但只能覆盖数值，漏掉「叙述性编造」。
   - 裁判版：让大模型判「有无上下文无支撑的陈述」，覆盖面广，但要声明它是近似。
3. 引用精确率单独看：引用编号指向的片段，是否真的包含答案关键串。这决定了
   「带引用」是真溯源还是装饰。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
CITE_RE = re.compile(r"\[\d{1,3}\]")

JUDGE_SYSTEM = (
    "你是医学问答系统的评审员，任务是判断机器生成的答案是否存在「上下文没有提供依据」的陈述。"
    "严格要求：只依据给定的上下文片段判断，不要使用你自己的医学知识来补 gap。"
)

JUDGE_TEMPLATE = """上下文片段：
{context}

问题：{question}

机器答案：
{answer}

请判断：机器答案中是否存在上下文片段里找不到依据的事实陈述？

判定标准：
- 复述、概括上下文内容，即使不完整，也不算无依据。
- 「少说了」不算无依据；只有「多说了上下文没写的内容」才算。
- 数值（剂量、阈值、时间窗、百分比）与上下文不一致，或与上下文都找不到来源，算无依据。

只输出如下 JSON，不要输出其他内容：
{{"unsupported": true或false, "spans": ["无依据的原句片段，没有则空数组"], "reason": "一句话说明"}}"""


def normalize(text: str) -> str:
    """NFKC + 去空白 + 小写：PDF 抽取会塞空格、网页版用全角字符。"""
    s = unicodedata.normalize("NFKC", text or "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def extract_numbers(text: str) -> List[str]:
    """抽答案里的事实性数字，剔除引用编号和列表序号这类结构性编号。"""
    t = CITE_RE.sub(" ", text or "")
    t = re.sub(r"(?m)^\s*\d{1,2}[.、)]\s*", " ", t)          # 行首列表序号
    t = re.sub(r"[（(]\s*\d{1,2}\s*[)）]", " ", t)           # （1）这类括号序号
    return NUM_RE.findall(t)


def key_hits(text: str, keys: Sequence[str]) -> List[str]:
    n = normalize(text)
    return [k for k in keys if normalize(k) in n]


def citation_stats(record: Dict[str, Any]) -> Dict[str, Any]:
    """引用的行为是否真的指向支撑答案的片段。"""
    cites: List[int] = record.get("citations") or []
    contexts: List[str] = record.get("contexts") or []
    raw_keys: Sequence[str] = record.get("answer_keys") or []

    valid = [i for i in cites if 1 <= i <= len(contexts)]
    if not valid:
        return {
            "citation_rate": 0.0,
            "citation_precision": 0.0,
            "citation_supported": 0,
            "citation_total": len(cites),
            "citation_dangling": len(cites),
        }
    if not raw_keys:
        # 应拒答题没有关键串，只统计引用是否指向真实片段
        return {
            "citation_rate": 1.0,
            "citation_precision": 0.0,
            "citation_supported": 0,
            "citation_total": len(valid),
            "citation_dangling": len(cites) - len(valid),
        }
    supported = 0
    for idx in valid:
        ctx = contexts[idx - 1]
        if key_hits(ctx, raw_keys):
            supported += 1
    return {
        "citation_rate": 1.0,
        "citation_precision": supported / len(valid),
        "citation_supported": supported,
        "citation_total": len(valid),
        "citation_dangling": len(cites) - len(valid),
    }


def judge_one(client: Any, record: Dict[str, Any], model: str) -> Dict[str, Any]:
    """调用裁判模型判定有无依据。返回原始判定结果，失败时标记 error。"""
    context = "\n\n".join(
        f"[{i}] {c}" for i, c in enumerate(record.get("contexts") or [], start=1)
    )
    prompt = JUDGE_TEMPLATE.format(
        context=context[:6000], question=record["question"], answer=record.get("answer", "")
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "stream": False,
    }
    try:
        import httpx

        resp = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}
        content = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return {"error": f"无法解析：{content[:120]}"}
        data = json.loads(m.group(0))
        return {
            "unsupported": bool(data.get("unsupported")),
            "spans": data.get("spans") or [],
            "reason": data.get("reason", ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def evaluate_records(records: List[Dict[str, Any]], use_judge: bool, judge_model: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        answer = rec.get("answer", "")
        keys = rec.get("answer_keys") or []
        keys_norm = [normalize(k) for k in keys]
        ans_norm = normalize(answer)
        hit = [1 for k in keys_norm if k in ans_norm]

        ctx_all = normalize("\n".join(rec.get("contexts") or []))
        nums = extract_numbers(answer)
        unsupported = [n for n in nums if normalize(n) not in ctx_all]

        row: Dict[str, Any] = {
            "id": rec["id"],
            "type": rec["type"],
            "question": rec["question"],
            "refused": bool(rec.get("refused")),
            "gold_key_recall": round(len(hit) / len(keys_norm), 4) if keys_norm else 0.0,
            "full_hit": 1.0 if keys_norm and len(hit) == len(keys_norm) else 0.0,
            "missing_keys": [k for k, h in zip(keys, hit) if not h],
            "answer_numbers": len(nums),
            "unsupported_numbers": len(unsupported),
            "unsupported_number_rate": round(len(unsupported) / len(nums), 4) if nums else 0.0,
            "unsupported_samples": unsupported[:6],
            "retrieval_recall": rec.get("retrieval_recall"),
        }
        row.update(citation_stats(rec))
        if use_judge:
            row["judge"] = judge_one(None, rec, judge_model)
            row["judge_unsupported"] = (
                1.0 if row["judge"].get("unsupported") else 0.0
            )
        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]

    def mean(xs: List[Dict[str, Any]], field: str) -> float:
        vals = [x[field] for x in xs if isinstance(x.get(field), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    out: Dict[str, Any] = {
        "label": label,
        "n_total": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        # 正确性
        "gold_key_recall": mean(ans, "gold_key_recall"),
        "full_hit_rate": mean(ans, "full_hit"),
        # 引用
        "citation_rate": mean(ans, "citation_rate"),
        "citation_precision": mean(ans, "citation_precision"),
        # 规则版幻觉
        "answer_with_unsupported_number_rate": round(
            sum(1 for r in ans if r["unsupported_numbers"] > 0) / len(ans), 4
        ) if ans else 0.0,
        "unsupported_number_rate": mean(ans, "unsupported_number_rate"),
        # 拒答权衡
        "false_refusal_rate": round(sum(1 for r in ans if r["refused"]) / len(ans), 4) if ans else 0.0,
        "refusal_rate_unanswerable": round(sum(1 for r in una if r["refused"]) / len(una), 4) if una else 0.0,
    }
    if ans and isinstance(ans[0].get("judge_unsupported"), float):
        out["judge_unsupported_rate_answerable"] = mean(ans, "judge_unsupported")
    if una and isinstance(una[0].get("judge_unsupported"), float):
        out["judge_unsupported_rate_unanswerable"] = mean(una, "judge_unsupported")
    out["per_query"] = rows
    return out


def attach_retrieval_recall(records: List[Dict[str, Any]], path: Optional[str]) -> None:
    """把检索端同题的答案关键串召回率挂上来，用来算「召回对了但生成漏了」的漏斗损耗。"""
    if not path or not Path(path).exists():
        return
    data = json.load(open(path, encoding="utf-8"))
    lookup = {}
    for q in data.get("answerable_per_query", []):
        lookup[q["id"]] = q.get("answer_recall")
    for q in data.get("unanswerable_per_query", []):
        lookup[q["id"]] = None
    for rec in records:
        rec["retrieval_recall"] = lookup.get(rec["id"])


def load_records(path: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成端评测")
    ap.add_argument("inputs", nargs="+", help="一个或多个 generation_*.jsonl")
    ap.add_argument("--retrieval-report", default="reports/benchmark_eval_rerank.json",
                    help="同口径的检索端报告，用于关联漏斗损耗")
    ap.add_argument("--judge", action="store_true", help="启用裁判模型判定（会产生 API 调用）")
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--out", default="reports/generation_eval.json")
    args = ap.parse_args()

    reports = []
    for path in args.inputs:
        records = load_records(path)
        attach_retrieval_recall(records, args.retrieval_report)
        label = Path(path).stem
        rows = evaluate_records(records, args.judge, args.judge_model)
        reports.append(summarize(rows, label))
        s = reports[-1]
        print(f"[eval] === {label} ===")
        print(f"[eval] 可回答 {s['n_answerable']} 条 / 应拒答 {s['n_unanswerable']} 条")
        print(f"[eval] 关键串召回 {s['gold_key_recall']:.4f}  全中率 {s['full_hit_rate']:.4f}")
        print(f"[eval] 引用率 {s['citation_rate']:.4f}  引用精确率 {s['citation_precision']:.4f}")
        print(f"[eval] 答案含无依据数字比例 {s['answer_with_unsupported_number_rate']:.4f}  "
              f"无依据数字占比 {s['unsupported_number_rate']:.4f}")
        print(f"[eval] 可答题误拒率 {s['false_refusal_rate']:.4f}  "
              f"拒答题正确拒答率 {s['refusal_rate_unanswerable']:.4f}")
        if "judge_unsupported_rate_answerable" in s:
            print(f"[eval] 裁判判定无依据比例（可答题）{s['judge_unsupported_rate_answerable']:.4f}  "
                  f"（拒答题）{s['judge_unsupported_rate_unanswerable']:.4f}")
        print()

    # 漏斗损耗：检索召回了关键串，生成却没写出来
    for s in reports:
        pairs = [
            (r["retrieval_recall"], r["gold_key_recall"])
            for r in s["per_query"]
            if r["type"] != "unanswerable" and isinstance(r.get("retrieval_recall"), float)
        ]
        if pairs:
            lost = sum(1 for a, b in pairs if a >= 0.99 and b < 0.99)
            s["retrieval_recalled_but_generation_missed"] = round(lost / len(pairs), 4)
            print(f"[eval] {s['label']}：检索已召回但生成未写出的题目占比 "
                  f"{lost}/{len(pairs)} = {lost / len(pairs):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 报告写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
