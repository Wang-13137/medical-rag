"""把人工标注的 queries_v1.yaml 构建成正式评测集，并做硬性校验。

这个脚本存在的唯一理由：**不允许任何一个凭空写出来的答案混进评测集。**
每条可回答题的 answer_keys 必须能在 gold_doc 原文里逐字搜到；
每条应拒答题的 probe_terms 必须在全部语料里零出现。

用法:
    python scripts/build_benchmark.py                    # 校验并产出 benchmark_v1.json
    python scripts/build_benchmark.py --strict           # 有任何一条校验失败就以非零码退出
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SOURCE = ROOT / "data" / "eval" / "queries_v1.yaml"
DEFAULT_CORPUS = ROOT / "data" / "clean"
DEFAULT_OUT = ROOT / "data" / "eval" / "benchmark_v1.json"

ANSWERABLE = {"fact", "numeric", "disambiguation"}


def normalize(text: str) -> str:
    """统一文本形态：NFKC 折叠全角字符，再去掉所有空白。

    语料里同一句话在不同文档中的写法差别很大：PDF 抽出的文档会在字符间
    塞满空格，网页抓取的文档带着全角标点。去掉空白 + NFKC 之后两侧才能
    做可靠的子串匹配。
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("＜", "<").replace("＞", ">")
    folded = folded.replace("－", "-").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", "", folded)


def load_corpus(corpus_dir: Path) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        texts[path.name] = {"raw": raw, "norm": normalize(raw)}
    return texts  # type: ignore[return-value]


def validate_answers(items: List[Dict[str, Any]], corpus: Dict[str, dict]) -> Dict[str, Any]:
    """逐条校验关键串是否真实存在于指定文档。"""
    ok_items: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []

    for item in items:
        if item["type"] not in ANSWERABLE:
            continue
        gold_doc = item.get("gold_doc")
        missing: List[str] = []
        if not gold_doc or gold_doc not in corpus:
            failures.append({"id": item["id"], "reason": f"gold_doc 不存在：{gold_doc}"})
            continue
        norm = corpus[gold_doc]["norm"]
        for key in item.get("answer_keys") or []:
            if normalize(key) not in norm:
                missing.append(key)

        # 关键串在几份文档里同时出现：串过于通用会削弱评测区分力
        spread = []
        for key in item.get("answer_keys") or []:
            nk = normalize(key)
            docs = [name for name, payload in corpus.items() if nk in payload["norm"]]
            if len(docs) > 3:
                spread.append({"key": key, "docs": len(docs)})

        if missing:
            failures.append({"id": item["id"], "reason": "关键串未在原文中找到", "missing": missing})
            continue
        if spread:
            ambiguous.append({"id": item["id"], "spread": spread})
        ok_items.append(item)

    return {"ok": ok_items, "failures": failures, "ambiguous": ambiguous}


def validate_refusals(items: List[Dict[str, Any]], corpus: Dict[str, dict]) -> Dict[str, Any]:
    """应拒答题：主题词必须在整个语料里零出现，否则说明题库无法声称「无依据」。"""
    results = []
    contaminated = []
    for item in items:
        if item.get("type") != "unanswerable":
            continue
        hits = {}
        for term in item.get("probe_terms") or []:
            nterm = normalize(term)
            total = 0
            per_doc = {}
            for name, payload in corpus.items():
                c = payload["norm"].count(nterm)
                if c:
                    per_doc[name] = c
                    total += c
            hits[term] = {"total": total, "docs": per_doc}
            if total > 0:
                contaminated.append(
                    {"id": item["id"], "term": term, "total": total, "docs": per_doc}
                )
        results.append({"id": item["id"], "probe_hits": hits})
    return {"results": results, "contaminated": contaminated}


def main() -> int:
    ap = argparse.ArgumentParser(description="构建并校验评测集")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default="reports/benchmark_validation.json")
    ap.add_argument("--strict", action="store_true", help="存在校验失败时以非零码退出")
    args = ap.parse_args()

    source_path = Path(args.source)
    if source_path.name == "_part2.yaml":
        print("[bench] 请勿直接构建分片文件")
        return 1

    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = payload["items"]
    meta: Dict[str, Any] = payload.get("metadata", {})

    corpus = load_corpus(Path(args.corpus_dir))
    ans = validate_answers(items, corpus)
    ref = validate_refusals(items, corpus)

    built_at = datetime.now().isoformat(timespec="seconds")

    answerable_ok = ans["ok"]
    refusals = [it for it in items if it.get("type") == "unanswerable"]

    # 题型分布
    type_counter = Counter(it["type"] for it in answerable_ok)
    doc_counter = Counter(it["gold_doc"] for it in answerable_ok)

    benchmark = {
        "metadata": {
            "version": str(meta.get("version", "v1")),
            "created": str(meta.get("created", datetime.now().strftime("%Y-%m-%d"))),
            "built_at": built_at,
            "corpus_dir": str(Path(args.corpus_dir).relative_to(ROOT)),
            "corpus_docs": len(corpus),
            "total_items": len(answerable_ok) + len(refusals),
            "answerable": len(answerable_ok),
            "unanswerable": len(refusals),
            "type_distribution": dict(type_counter),
            "gold_doc_distribution": dict(doc_counter),
            "validation": {
                "answer_keys_checked": sum(
                    len(it.get("answer_keys") or []) for it in answerable_ok
                ),
                "failed_items": len(ans["failures"]),
                "contaminated_refusals": len(ref["contaminated"]),
            },
        },
        "items": answerable_ok + refusals,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "built_at": benchmark["metadata"]["built_at"],
        "source": str(source_path.relative_to(ROOT)),
        "answer_key_failures": ans["failures"],
        "ambiguous_keys": ans["ambiguous"],
        "refusal_contamination": ref["contaminated"],
        "refusal_probe_detail": ref["results"],
        "type_distribution": dict(type_counter),
        "gold_doc_distribution": dict(doc_counter),
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[bench] 可回答题通过校验 {len(answerable_ok)} 条，应拒答题 {len(refusals)} 条")
    print(f"[bench] 题型分布：{dict(type_counter)}")
    print(f"[bench] 关键串总数 {benchmark['metadata']['validation']['answer_keys_checked']}")
    print(f"[bench] 校验失败 {len(ans['failures'])} 条 → {rep}")
    for f in ans["failures"]:
        detail = f.get("missing") or []
        print(f"    ✗ {f['id']}: {f['reason']} {detail}")
    if ref["contaminated"]:
        print(f"[bench] 应拒答题污染 {len(ref['contaminated'])} 处：")
        for c in ref["contaminated"]:
            print(f"    ✗ {c['id']} 主题词「{c['term']}」在语料出现 {c['total']} 次：{list(c['docs'])}")
    if ans["ambiguous"]:
        print(f"[bench] 通用关键串提示 {len(ans['ambiguous'])} 条（出现在 >3 份文档）：")
        for a in ans["ambiguous"][:20]:
            print(f"    ! {a['id']} {[s['key'] for s in a['spread']]}")

    if args.strict and (ans["failures"] or ref["contaminated"]):
        print("[bench] strict 模式下存在校验失败，benchmark 未通过")
        return 1
    print(f"[bench] 已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
