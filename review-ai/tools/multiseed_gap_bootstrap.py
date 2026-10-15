# -*- coding: utf-8 -*-
"""Multi-seed, seed-averaged paired bootstrap on the explicit/implicit negative-F1 gap.

Motivation: single-seed paired bootstrap of the gap was underpowered because the
E1 baseline's gap itself varies a lot across seeds (sd ~2.7pp). Here each
bootstrap resample draws ONE set of sentence indices, applies it to EVERY seed of
BOTH models, computes each model's gap averaged over its seeds, and takes the
difference. Sentence-sampling noise cancels (paired), seed noise is averaged.

Inputs are .npz files produced by tools/save_test_probs.py.

Example:
    .venv/bin/python tools/multiseed_gap_bootstrap.py \
        --probs_a results/test_probs/E1_xlmr_seed42.npz results/test_probs/E1_xlmr_seed43.npz \
        --probs_b results/test_probs/deberta_seed42.npz results/test_probs/deberta_seed43.npz \
        --label_a E1_xlmr --label_b deberta_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Seed-averaged paired bootstrap on explicit/implicit gap")
    ap.add_argument("--probs_a", nargs="+", required=True, help="npz files for model A seeds (baseline)")
    ap.add_argument("--probs_b", nargs="+", required=True, help="npz files for model B seeds (candidate)")
    ap.add_argument("--label_a", default="E1_xlmr")
    ap.add_argument("--label_b", required=True)
    ap.add_argument("--tags_file", default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def build_source_masks(tags_df: pd.DataFrame, text_ids: list[str], label_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    test_tags = tags_df[tags_df["split"] == "test"].copy()
    col_to_idx = {col: i for i, col in enumerate(label_cols)}
    explicit_mask = np.zeros((len(text_ids), len(label_cols)), dtype=bool)
    implicit_mask = np.zeros((len(text_ids), len(label_cols)), dtype=bool)
    id_to_row = {tid: i for i, tid in enumerate(text_ids)}
    for _, row in test_tags.iterrows():
        text_id = str(row["text_id"])
        if text_id not in id_to_row:
            continue
        label_col = str(row["label_column"])
        if label_col not in col_to_idx:
            continue
        r = id_to_row[text_id]
        c = col_to_idx[label_col]
        if row["source"] == "explicit":
            explicit_mask[r, c] = True
        elif row["source"] == "implicit":
            implicit_mask[r, c] = True
    return explicit_mask, implicit_mask


def load_preds(npz_path: str) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    probs = z["probs"]
    thresholds = z["thresholds"]
    return {
        "pred": (probs >= thresholds).astype(np.int8),
        "y_true": z["y_true"],
        "text_ids": [str(t) for t in z["text_ids"]],
        "label_cols": [str(c) for c in z["label_cols"]],
        "path": npz_path,
    }


def neg_gap(y_true: np.ndarray, y_pred: np.ndarray, exp_sub: np.ndarray, imp_sub: np.ndarray) -> float:
    f1_e = float(f1_score(y_true[exp_sub], y_pred[exp_sub], average="binary", zero_division=0)) if exp_sub.any() else float("nan")
    f1_i = float(f1_score(y_true[imp_sub], y_pred[imp_sub], average="binary", zero_division=0)) if imp_sub.any() else float("nan")
    return f1_e - f1_i


def ci(vals: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def main() -> None:
    args = parse_args()
    runs_a = [load_preds(p) for p in args.probs_a]
    runs_b = [load_preds(p) for p in args.probs_b]

    ref = runs_a[0]
    label_cols = ref["label_cols"]
    text_ids = ref["text_ids"]
    y_true = ref["y_true"].astype(int)
    for r in runs_a + runs_b:
        assert r["label_cols"] == label_cols, f"label mismatch: {r['path']}"
        assert r["text_ids"] == text_ids, f"test-set mismatch: {r['path']}"

    neg_idx = [i for i, c in enumerate(label_cols) if "ネガティブ" in c]
    tags_df = pd.read_csv(args.tags_file)
    explicit_mask, implicit_mask = build_source_masks(tags_df, text_ids, label_cols)
    exp_sub = np.zeros_like(explicit_mask)
    imp_sub = np.zeros_like(implicit_mask)
    for i in neg_idx:
        exp_sub[:, i] = explicit_mask[:, i]
        imp_sub[:, i] = implicit_mask[:, i]
    print(f"[INFO] sentences={len(text_ids):,}  explicit neg pairs={exp_sub.sum()}  implicit neg pairs={imp_sub.sum()}")
    print(f"[INFO] model A ({args.label_a}): {len(runs_a)} seeds | model B ({args.label_b}): {len(runs_b)} seeds")

    gaps_a = [neg_gap(y_true, r["pred"], exp_sub, imp_sub) for r in runs_a]
    gaps_b = [neg_gap(y_true, r["pred"], exp_sub, imp_sub) for r in runs_b]
    print("\n[POINT ESTIMATES per seed]")
    for r, g in zip(runs_a, gaps_a):
        print(f"  {args.label_a}  {Path(r['path']).stem:28s} gap={g:+.4f}")
    for r, g in zip(runs_b, gaps_b):
        print(f"  {args.label_b}  {Path(r['path']).stem:28s} gap={g:+.4f}")
    mean_a, mean_b = float(np.mean(gaps_a)), float(np.mean(gaps_b))
    print(f"  mean gap: {args.label_a}={mean_a:+.4f} (sd {np.std(gaps_a, ddof=1):.4f})  "
          f"{args.label_b}={mean_b:+.4f} (sd {np.std(gaps_b, ddof=1):.4f})  shrinkage={mean_a - mean_b:+.4f}")

    preds_a = [r["pred"] for r in runs_a]
    preds_b = [r["pred"] for r in runs_b]
    rng = np.random.default_rng(args.seed)
    n = len(y_true)
    diffs, ga_means, gb_means = [], [], []
    print(f"\n[INFO] seed-averaged paired bootstrap n_boot={args.n_boot} ...")
    for it in range(args.n_boot):
        idx = rng.integers(0, n, n)  # same sentences for every seed of BOTH models
        yt = y_true[idx]
        es = exp_sub[idx]
        _is = imp_sub[idx]
        ga = float(np.mean([neg_gap(yt, p[idx], es, _is) for p in preds_a]))
        gb = float(np.mean([neg_gap(yt, p[idx], es, _is) for p in preds_b]))
        ga_means.append(ga)
        gb_means.append(gb)
        diffs.append(ga - gb)
        if (it + 1) % 500 == 0:
            print(f"       {it + 1}/{args.n_boot}")

    d = np.array(diffs)
    d_lo, d_hi = ci(d)
    ga_lo, ga_hi = ci(np.array(ga_means))
    gb_lo, gb_hi = ci(np.array(gb_means))
    p_gt0 = float((d > 0).mean())
    sig = d_lo > 0

    report = {
        "label_a": args.label_a, "label_b": args.label_b,
        "runs_a": [r["path"] for r in runs_a], "runs_b": [r["path"] for r in runs_b],
        "n_test_sentences": n, "n_boot": args.n_boot,
        "per_seed_gaps_a": gaps_a, "per_seed_gaps_b": gaps_b,
        "mean_gap_a": mean_a, "mean_gap_b": mean_b,
        "shrinkage_point": mean_a - mean_b,
        "bootstrap": {
            "gap_a_ci": [ga_lo, ga_hi],
            "gap_b_ci": [gb_lo, gb_hi],
            "shrinkage_ci": [d_lo, d_hi],
            "P_shrinkage_gt_0": p_gt0,
            "significant_at_95": bool(sig),
        },
    }
    print("\n[RESULT] seed-averaged paired bootstrap 95% CI")
    print(f"  gap_{args.label_a}  {mean_a:+.4f}  CI [{ga_lo:+.4f}, {ga_hi:+.4f}]")
    print(f"  gap_{args.label_b}  {mean_b:+.4f}  CI [{gb_lo:+.4f}, {gb_hi:+.4f}]")
    print(f"  shrinkage (A-B)  {mean_a - mean_b:+.4f}  CI [{d_lo:+.4f}, {d_hi:+.4f}]  "
          f"P(shrinkage>0)={p_gt0:.4f}  -> {'SIGNIFICANT (CI excludes 0)' if sig else 'not significant at 95%'}")

    out = Path(args.out) if args.out else PROJECT_ROOT / f"results/gap_multiseed_bootstrap_{args.label_a}_vs_{args.label_b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
