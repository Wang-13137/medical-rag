#!/usr/bin/env python
"""生成策略之间的配对显著性检验。

两份 generation_*.jsonl 必须在同一批评测题上跑出来（题号一致），逐题配对比较。
对连续指标用配对 bootstrap（重抽样 10000 次），对二分类指标用 McNemar 精确检验。
不假设分布，也不只看均值差——100 条样本很容易出现「看起来提升了其实是噪声」。

用法：
    python scripts/compare_generation.py generation_strict_v2.jsonl generation_strict.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.stats import binomtest  # noqa: E402

from eval_generation import evaluate_records, load_records  # noqa: E402

CONTINUOUS = ["gold_key_recall", "citation_precision", "unsupported_number_rate"]
BINARY = ["full_hit", "refused"]


def pair_by_id(a: List[Dict], b: List[Dict]) -> List[Tuple[Dict, Dict]]:
    idx_b = {r["id"]: r for r in b}
    return [(r, idx_b[r["id"]]) for r in a if r["id"] in idx_b]


def bootstrap_diff(xs: Sequence[float], ys: Sequence[float], n_boot: int, seed: int):
    """paired bootstrap：返回差值均值与 95% 置信区间。"""
    rng = random.Random(seed)
    n = len(xs)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(xs[i] - ys[i] for i in idx) / n)
    diffs.sort()
    mean_diff = sum(xs[i] - ys[i] for i in range(n)) / n
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    return mean_diff, lo, hi


def mcnemar(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """配对二分类检验：只统计「一成一败」的不一致对。"""
    b = sum(1 for x, y in zip(xs, ys) if x > y)   # 前者胜
    c = sum(1 for x, y in zip(xs, ys) if x < y)   # 后者胜
    total = b + c
    if total == 0:
        return {"b": b, "c": c, "p": 1.0}
    res = binomtest(b, total, 0.5)
    return {"b": b, "c": c, "p": float(res.pvalue)}


def main() -> int:
    ap = argparse.ArgumentParser(description="生成策略配对比较")
    ap.add_argument("left", help="被比较的策略 A（通常是改进后的）")
    ap.add_argument("right", help="对照策略 B")
    ap.add_argument("--left-label", default="")
    ap.add_argument("--right-label", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default="reports/generation_compare.json")
    args = ap.parse_args()

    rows_a = evaluate_records(load_records(args.left), use_judge=False, judge_model="")
    rows_b = evaluate_records(load_records(args.right), use_judge=False, judge_model="")
    pairs = pair_by_id(rows_a, rows_b)
    la = args.left_label or Path(args.left).stem
    lb = args.right_label or Path(args.right).stem
    print(f"[cmp] {la} vs {lb}：配对题目 {len(pairs)} 条")
    if not pairs:
        print("[cmp] 两份结果没有共同题号，无法配对")
        return 1

    scope = {
        "全部题": pairs,
        "仅可答题": [(x, y) for x, y in pairs if x["type"] != "unanswerable"],
        "仅应拒答题": [(x, y) for x, y in pairs if x["type"] == "unanswerable"],
    }

    report: Dict[str, object] = {
        "left": la,
        "right": lb,
        "n_pairs": len(pairs),
        "scopes": {},
    }
    for name, ps in scope.items():
        if not ps:
            continue
        print(f"\n[cmp] --- {name}（n={len(ps)}）---")
        entry: Dict[str, object] = {}
        for field in CONTINUOUS:
            xs = [x[field] for x, _ in ps]
            ys = [y[field] for _, y in ps]
            mean_diff, lo, hi = bootstrap_diff(xs, ys, args.n_boot, args.seed)
            flag = "显著" if (lo > 0 or hi < 0) else "不显著"
            print(
                f"[cmp] {field:<24} A={sum(xs)/len(xs):.4f} B={sum(ys)/len(ys):.4f} "
                f"差值={mean_diff:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}] {flag}"
            )
            entry[field] = {
                "left_mean": round(sum(xs) / len(xs), 4),
                "right_mean": round(sum(ys) / len(ys), 4),
                "diff": round(mean_diff, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0 or hi < 0),
            }
        for field in BINARY:
            xs = [x[field] for x, _ in ps]
            ys = [y[field] for _, y in ps]
            res = mcnemar(xs, ys)
            flag = "显著" if res["p"] < 0.05 else "不显著"
            print(
                f"[cmp] {field:<24} A={sum(xs)/len(xs):.4f} B={sum(ys)/len(ys):.4f} "
                f"McNemar b={res['b']} c={res['c']} p={res['p']:.4f} {flag}"
            )
            entry[field] = {
                "left_mean": round(sum(xs) / len(xs), 4),
                "right_mean": round(sum(ys) / len(ys), 4),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
            }
        report["scopes"][name] = entry  # type: ignore[index]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[cmp] 结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
