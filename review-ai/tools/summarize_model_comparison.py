# -*- coding: utf-8 -*-
"""Summary table for docs/model_comparison_plan.md (Parts A and B).

For every encoder (3 seeds, run by tools/run_model_comparison.sh) and every LLM setting
(tools/llm_absa_eval.py), report valid and test means, then a seed-averaged paired
bootstrap against three references on the test set:
    XLM-R E1 (re-run, 3 seeds)        the thesis baseline
    DeBERTa E1 (re-run, 3 seeds)      the strongest encoder before this comparison
    DeBERTa + E7/E8 (3 seeds)         the best configuration of the thesis

Output: results/model_comparison_summary.json and a printed table.

    .venv/bin/python tools/summarize_model_comparison.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROBS = RES / "test_probs"
PY = sys.executable

ENCODERS = ["xlmr_base", "deberta_ja", "mbert", "luke_ja", "modernbert_ja_130m", "modernbert_ja_130m_lr1e-5",
            "modernbert_ja_310m", "bert_large_ja", "xlmr_large"]
# fine-tuned groups that are not plain E1 runs: name -> run prefix (seeds appended)
GROUPS = {"modernbert_310m_E7E8 (A')": "Aprime_E8_modernbert310m_seed"}
LLMS = ["LLM_qwen25_7b_zero", "LLM_qwen25_7b_few", "LLM_swallow8b_zero", "LLM_swallow8b_few",
        "LLM_llmjp31_13b_zero", "LLM_llmjp31_13b_few", "LLM_gptoss20b_zero", "LLM_gptoss20b_few"]
SEEDS = (42, 43, 44)
REFS = {
    "modernbert310m_E1": [f"CMP_E1_modernbert_ja_310m_seed{s}" for s in SEEDS],
    "xlmr_E1": [f"CMP_E1_xlmr_base_seed{s}" for s in SEEDS],
    "deberta_E1": [f"CMP_E1_deberta_ja_seed{s}" for s in SEEDS],
    "deberta_E7E8": [f"E8_deberta_recent2M_seed{s}" for s in SEEDS],
}
KEYS = [("agg_negative_f1", "neg"), ("agg_positive_f1", "pos"), ("micro_f1", "micro")]


def load(name: str) -> dict | None:
    p = RES / f"{name}_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def mean_sd(vals: list[float]) -> tuple[float, float]:
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


def bootstrap(a: list[str], b: list[str], tag: str) -> dict | None:
    pa = [str(PROBS / f"{n}.npz") for n in a]
    pb = [str(PROBS / f"{n}.npz") for n in b]
    if not all(Path(p).exists() for p in pa + pb):
        return None
    out = RES / "model_comparison" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        subprocess.run([PY, str(ROOT / "tools/multiseed_f1_bootstrap.py"), "--probs_a", *pa, "--probs_b", *pb,
                        "--label_a", tag.split("__vs__")[1], "--label_b", tag.split("__vs__")[0],
                        "--out", str(out)], check=True, capture_output=True)
    m = json.loads(out.read_text(encoding="utf-8"))["metrics"]
    return {k: {"delta": m[k]["delta_point"], "ci": m[k]["delta_ci"], "sig": m[k]["significant_at_95"]}
            for k in ("agg_neg", "implicit_neg", "micro")}


def detail(runs: list[str]) -> dict | None:
    """Seed-mean per-aspect negative F1 and explicit/implicit negative F1 from saved predictions."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from tools.multiseed_gap_bootstrap import build_source_masks, load_preds
    paths = [PROBS / f"{n}.npz" for n in runs]
    if not all(p.exists() for p in paths):
        return None
    preds = [load_preds(str(p)) for p in paths]
    cols, y = preds[0]["label_cols"], preds[0]["y_true"].astype(int)
    explicit, implicit = build_source_masks(pd.read_csv(ROOT / "data/absa_explicit_implicit_tags.csv"),
                                            preds[0]["text_ids"], cols)
    neg = np.array(["ネガティブ" in c for c in cols])

    def f1(mask, p):
        yy, pp = y[mask], p[mask]
        tp = int(((yy == 1) & (pp == 1)).sum()); d = 2 * tp + int(((yy == 0) & (pp == 1)).sum()) + int(((yy == 1) & (pp == 0)).sum())
        return 2 * tp / d if d else 0.0
    out = {"explicit_neg": float(np.mean([f1(explicit & neg, r["pred"]) for r in preds])),
           "implicit_neg": float(np.mean([f1(implicit & neg, r["pred"]) for r in preds]))}
    for j, c in enumerate(cols):
        if neg[j]:
            m = np.zeros_like(y, dtype=bool); m[:, j] = True
            out[c.split("_")[0]] = float(np.mean([f1(m, r["pred"]) for r in preds]))
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = {}
    for key in ENCODERS:
        names = [f"CMP_E1_{key}_seed{s}" for s in SEEDS]
        ms = [load(n) for n in names]
        if not all(ms):
            print(f"[skip] {key}: {sum(m is None for m in ms)} run(s) missing")
            continue
        row = {"runs": names, "model": ms[0]["args"]["model_name"]}
        for k, short in KEYS:
            row[f"valid_{short}"] = mean_sd([m[f"valid_{k}"] for m in ms])
            row[f"test_{short}"] = mean_sd([m[f"eval_{k}"] for m in ms])
        rows[key] = row
    for gname, prefix in GROUPS.items():
        names = [f"{prefix}{s}" for s in SEEDS]
        ms = [load(n) for n in names]
        if not all(ms):
            print(f"[skip] {gname}: {sum(m is None for m in ms)} run(s) missing")
            continue
        row = {"runs": names, "model": gname}
        for k, short in KEYS:
            row[f"valid_{short}"] = mean_sd([m[f"valid_{k}"] for m in ms])
            row[f"test_{short}"] = mean_sd([m[f"eval_{k}"] for m in ms])
        rows[gname] = row
    for name in LLMS:
        m = load(name)
        if m is None:
            print(f"[skip] {name}: missing")
            continue
        rows[name] = {"runs": [name], "model": m["model"], "mode": m["mode"], "parse_failures": m["parse_failures"],
                      **{f"test_{short}": (m[k], 0.0) for k, short in KEYS}}
        if "lenient" in m and (PROBS / f"{name}_lenient.npz").exists():
            rows[f"{name}_lenient"] = {"runs": [f"{name}_lenient"], "model": m["model"], "mode": m["mode"],
                                       **{f"test_{short}": (m["lenient"][k], 0.0) for k, short in KEYS}}

    for key, row in rows.items():
        row["bootstrap"] = {}
        for ref, ref_runs in REFS.items():
            if row["runs"] == ref_runs:
                continue
            res = bootstrap(ref_runs, row["runs"], f"{key}__vs__{ref}")
            if res:
                row["bootstrap"][ref] = res

    for key, row in rows.items():
        row["detail"] = detail(row["runs"])
    (RES / "model_comparison_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'model':22s} {'valid neg':>14s} {'test neg':>14s} {'test pos':>14s} {'test micro':>14s}   "
          f"Δneg vs " + " / ".join(REFS))
    for key, row in rows.items():
        v = row.get("valid_neg")
        fmt = lambda t: f"{t[0]:.4f}±{t[1]:.3f}"
        b = row["bootstrap"]
        deltas = " / ".join(
            (f"{100 * b[r]['agg_neg']['delta']:+.2f}{'*' if b[r]['agg_neg']['sig'] else ''}" if r in b else "—")
            for r in REFS)
        print(f"{key:22s} {fmt(v) if v else '—':>14s} {fmt(row['test_neg']):>14s} {fmt(row['test_pos']):>14s} "
              f"{fmt(row['test_micro']):>14s}   {deltas}")
    print("\n* = 95% CI excludes 0. Written results/model_comparison_summary.json")


if __name__ == "__main__":
    main()
