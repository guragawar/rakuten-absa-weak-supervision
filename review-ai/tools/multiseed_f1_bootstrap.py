# -*- coding: utf-8 -*-
"""Seed-averaged paired bootstrap of F1 differences between two models (B - A).

Generalises tools/multiseed_gap_bootstrap.py from the explicit/implicit gap to the
headline metrics. Each bootstrap resample draws ONE set of test-sentence indices,
applies it to every seed of both models, averages each model's metric over its
seeds, and takes the difference. Sentence-sampling noise cancels (paired); seed
noise is averaged.

Metrics (all binary F1 on pooled label cells, i.e. the training script's
`eval_agg_*` definition):
    agg_neg       all ネガティブ cells
    agg_pos       all ポジティブ cells
    micro         all 14 labels
    implicit_neg  ネガティブ cells tagged "implicit" in the tags file (gold-positive
                  cells only carry a tag; see multiseed_gap_bootstrap.py)

Inputs are .npz files from tools/save_test_probs.py or train_absa_baseline.py.

Example (DeBERTa E1 vs DeBERTa + E7/E8, 3 seeds each):
    .venv/bin/python tools/multiseed_f1_bootstrap.py \
        --probs_a results/test_probs/deberta_v3_seed4{2,3,4}.npz \
        --probs_b results/test_probs/E8_deberta_recent2M_seed4{2,3,4}.npz \
        --label_a deberta_E1 --label_b deberta_E8 --out results/repro_stacking.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.multiseed_gap_bootstrap import build_source_masks, ci, load_preds  # noqa: E402

METRICS = ("agg_neg", "agg_pos", "micro", "implicit_neg")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Seed-averaged paired bootstrap of F1 differences")
    ap.add_argument("--probs_a", nargs="+", required=True, help="npz files, model A (reference)")
    ap.add_argument("--probs_b", nargs="+", required=True, help="npz files, model B (candidate)")
    ap.add_argument("--label_a", required=True)
    ap.add_argument("--label_b", required=True)
    ap.add_argument("--tags_file", default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labels", nargs="*", default=[],
                    help="extra per-label metrics, e.g. 朝食_ネガティブ 夕食_ネガティブ")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def pooled_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & (p == 1)))
    denom = 2 * tp + int(np.sum((y == 0) & (p == 1))) + int(np.sum((y == 1) & (p == 0)))
    return 2 * tp / denom if denom else 0.0


def all_metrics(y: np.ndarray, p: np.ndarray, cells: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: pooled_f1(y[c], p[c]) for m, c in cells.items()}


def main() -> None:
    args = parse_args()
    runs_a = [load_preds(p) for p in args.probs_a]
    runs_b = [load_preds(p) for p in args.probs_b]
    ref = runs_a[0]
    label_cols, text_ids = ref["label_cols"], ref["text_ids"]
    y = ref["y_true"].astype(int)
    for r in runs_a + runs_b:
        assert r["label_cols"] == label_cols, f"label mismatch: {r['path']}"
        assert r["text_ids"] == text_ids, f"test-set mismatch: {r['path']}"
        assert np.array_equal(r["y_true"].astype(int), y), f"y_true mismatch: {r['path']}"

    shape = y.shape
    neg_cols = np.array(["ネガティブ" in c for c in label_cols])
    pos_cols = np.array(["ポジティブ" in c for c in label_cols])
    _, implicit_mask = build_source_masks(pd.read_csv(args.tags_file), text_ids, label_cols)
    cells = {
        "agg_neg": np.broadcast_to(neg_cols, shape),
        "agg_pos": np.broadcast_to(pos_cols, shape),
        "micro": np.ones(shape, dtype=bool),
        "implicit_neg": implicit_mask & neg_cols,
    }
    for lab in args.labels:
        assert lab in label_cols, f"unknown label {lab}"
        cells[f"label:{lab}"] = np.broadcast_to(np.array([c == lab for c in label_cols]), shape)
    metrics = list(cells)

    def seed_mean(runs, idx=None):
        vals = [all_metrics(y if idx is None else y[idx], r["pred"] if idx is None else r["pred"][idx],
                            cells if idx is None else {k: v[idx] for k, v in cells.items()}) for r in runs]
        return {m: float(np.mean([v[m] for v in vals])) for m in metrics}, vals

    point_a, per_a = seed_mean(runs_a)
    point_b, per_b = seed_mean(runs_b)

    rng = np.random.default_rng(args.seed)
    n = len(y)
    diffs = {m: [] for m in metrics}
    for it in range(args.n_boot):
        idx = rng.integers(0, n, n)
        ma, _ = seed_mean(runs_a, idx)
        mb, _ = seed_mean(runs_b, idx)
        for m in metrics:
            diffs[m].append(mb[m] - ma[m])
        if (it + 1) % 500 == 0:
            print(f"  {it + 1}/{args.n_boot}")

    report = {
        "label_a": args.label_a, "label_b": args.label_b,
        "runs_a": [r["path"] for r in runs_a], "runs_b": [r["path"] for r in runs_b],
        "n_test_sentences": n, "n_boot": args.n_boot, "bootstrap_seed": args.seed,
        "n_implicit_neg_cells": int(cells["implicit_neg"].sum()),
        "method": "seed-averaged paired bootstrap; pooled binary F1 per metric",
        "metrics": {},
    }
    print(f"\n{args.label_b} - {args.label_a}  ({len(runs_a)} vs {len(runs_b)} seeds)")
    for m in metrics:
        d = np.array(diffs[m])
        lo, hi = ci(d)
        report["metrics"][m] = {
            "a_per_seed": [v[m] for v in per_a], "b_per_seed": [v[m] for v in per_b],
            "a_mean": point_a[m], "b_mean": point_b[m],
            "delta_point": point_b[m] - point_a[m],
            "delta_ci": [lo, hi], "P_delta_gt_0": float((d > 0).mean()),
            "significant_at_95": bool(lo > 0 or hi < 0),
        }
        print(f"  {m:13s} A={point_a[m]:.4f} B={point_b[m]:.4f} Δ={100 * (point_b[m] - point_a[m]):+.2f}pp "
              f"CI[{100 * lo:+.2f}, {100 * hi:+.2f}] P(Δ>0)={(d > 0).mean():.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
