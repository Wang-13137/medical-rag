#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把逐句蕴含校验当成硬闸门，在已生成的答案上做离线回测。

回答两个问题：
1. 闸门会改动多少条答案？（行动分布）
2. 改动之后，正确性掉了多少、幻觉降了多少？（这是权衡的核心）

指标口径与 eval_generation.py 保持一致，便于横向比较。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.verify import (  # noqa: E402
    DEFAULT_COVERAGE_THRESHOLD,
    extract_numbers,
    normalize,
    verify_record,
)


def key_hits(text: str, keys: Sequence[str]) -> List[str]:
    pool = normalize(text)
    return [k for k in keys if normalize(k) in pool]


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def unsupported_number_rate(answer: str, contexts: Sequence[str]) -> float:
    """答案里的数字有多少比例在上下文里找不到。"""
    numbers = extract_numbers(answer)
    if not numbers:
        return 0.0
    pool = normalize("\n".join(contexts))
    missing = sum(1 for num in numbers if normalize(num) not in pool)
    return missing / len(numbers)


def summarize(records: Sequence[Dict], verified: Sequence[Dict]) -> Dict[str, Any]:
    answerable = [
        (rec, ver)
        for rec, ver in zip(records, verified)
        if rec.get("type") != "unanswerable" and (rec.get("answer_keys") or [])
    ]
    unanswerable = [
        (rec, ver) for rec, ver in zip(records, verified) if rec.get("type") == "unanswerable"
    ]

    actions: Dict[str, int] = {}
    for ver in verified:
        actions[ver["action"]] = actions.get(ver["action"], 0) + 1

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    before_recall: List[float] = []
    after_recall: List[float] = []
    before_bad: List[float] = []
    after_bad: List[float] = []
    lost_items: List[Dict[str, Any]] = []

    for rec, ver in answerable:
        keys = rec["answer_keys"]
        original = rec.get("answer") or ""
        revised = ver["revised_answer"] or ""
        contexts = rec.get("contexts") or []

        b = len(key_hits(original, keys)) / len(keys)
        a = len(key_hits(revised, keys)) / len(keys)
        before_recall.append(b)
        after_recall.append(a)
        before_bad.append(unsupported_number_rate(original, contexts))
        after_bad.append(unsupported_number_rate(revised, contexts))
        if a < b:
            lost_items.append(
                {
                    "id": ver.get("qid"),
                    "before": round(b, 3),
                    "after": round(a, 3),
                    "action": ver["action"],
                    "dropped": ver["unsupported_sentences"][:2],
                }
            )

    # 拒答题：闸门会不会把正确的拒答改坏
    refusal_broken = sum(
        1
        for rec, ver in unanswerable
        if not ver["refused"] and "无法回答" not in (ver["revised_answer"] or "")
    )

    return {
        "counts": {
            "total": len(records),
            "answerable_scored": len(answerable),
            "unanswerable": len(unanswerable),
        },
        "actions": actions,
        "changed_rate": round(
            sum(v for k, v in actions.items() if k != "pass") / max(1, len(records)), 4
        ),
        "answerable": {
            "key_recall_before": round(mean(before_recall), 4),
            "key_recall_after": round(mean(after_recall), 4),
            "key_recall_delta": round(mean(after_recall) - mean(before_recall), 4),
            "unsupported_number_rate_before": round(mean(before_bad), 4),
            "unsupported_number_rate_after": round(mean(after_bad), 4),
            "items_losing_recall": len(lost_items),
        },
        "unanswerable": {
            "refusal_broken": refusal_broken,
        },
        "lost_detail": lost_items[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="在已生成答案上回测硬闸门")
    ap.add_argument("--generated", default="data/generated/generation_strict.jsonl")
    ap.add_argument("--mode", default="prune", choices=["flag", "prune", "refuse"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    ap.add_argument("--require-cited-support", action="store_true", help="连引用不准也算不支持")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = ROOT / args.generated
    if not path.exists():
        print(f"[verify] 找不到生成结果：{path}")
        return 1
    records = load_records(path)

    verified = [
        verify_record(
            rec,
            coverage_threshold=args.threshold,
            mode=args.mode,
            require_cited_support=args.require_cited_support,
        )
        for rec in records
    ]
    report = summarize(records, verified)
    report.update(
        {
            "generated": str(args.generated),
            "mode": args.mode,
            "threshold": args.threshold,
            "require_cited_support": args.require_cited_support,
        }
    )

    print(f"[verify] 闸门模式 {args.mode}，覆盖率阈值 {args.threshold}")
    print(
        f"[verify] 行动分布：pass {report['actions'].get('pass', 0)} / "
        f"flagged {report['actions'].get('flagged', 0)} / "
        f"pruned {report['actions'].get('pruned', 0)} / "
        f"refused {report['actions'].get('refused', 0)}"
        f"（改动比例 {report['changed_rate']:.3f}）"
    )
    ans = report["answerable"]
    print(
        f"[verify] 可答题关键串召回：{ans['key_recall_before']:.4f} 降到 {ans['key_recall_after']:.4f}"
        f"（{ans['key_recall_delta']:+.4f}），受影响 {ans['items_losing_recall']} 条"
    )
    print(
        f"[verify] 无依据数字占比：{ans['unsupported_number_rate_before']:.4f} 降到 "
        f"{ans['unsupported_number_rate_after']:.4f}"
    )
    print(f"[verify] 拒答题被改坏：{report['unanswerable']['refusal_broken']} 条")
    if report["lost_detail"]:
        print("[verify] 召回受损明细（前 5 条）：")
        for item in report["lost_detail"][:5]:
            print(f"    {item['id']} {item['before']:.2f} 降到 {item['after']:.2f} [{item['action']}]")

    out = Path(args.out) if args.out else ROOT / f"reports/verifier_apply_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verify] 报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
