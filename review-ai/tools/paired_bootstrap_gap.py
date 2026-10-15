# -*- coding: utf-8 -*-
"""Paired bootstrap significance test on the explicit/implicit negative-F1 GAP.

Answers "does swapping the backbone significantly shrink the explicit-vs-implicit
negative F1 gap compared to the E1 (XLM-R) baseline, or is it within test-sampling
noise?".

gap(model) = agg_negative_f1(explicit active pairs) - agg_negative_f1(implicit active pairs)

Paired bootstrap: each resample draws the same sentence indices for BOTH models
(same mechanism as tools/paired_bootstrap_e1_e8.py), so shared test-sampling
variance cancels and the CI on gap_A - gap_B is tight.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/paired_bootstrap_gap.py \
        --checkpoint_a checkpoints/absa_asl_v2_baseline/checkpoint-3984 \
        --checkpoint_b checkpoints/absa_tohoku_v3_baseline/checkpoint-3984 \
        --label_b tohoku_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_absa_baseline import load_absa_csv, load_label_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired bootstrap on explicit/implicit negative-F1 gap")
    ap.add_argument("--checkpoint_a", default="checkpoints/absa_asl_v2_baseline/checkpoint-3984",
                    help="Baseline (E1, XLM-R)")
    ap.add_argument("--checkpoint_b", required=True, help="Candidate backbone checkpoint")
    ap.add_argument("--label_a", default="E1_xlmr")
    ap.add_argument("--label_b", required=True)
    ap.add_argument("--test_file", default=str(PROJECT_ROOT / "data/absa_test.csv"))
    ap.add_argument("--tags_file", default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"))
    ap.add_argument("--labels_file", default=str(PROJECT_ROOT / "data/absa_label_columns.json"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def load_thresholds(ckpt_dir: str, label_cols: list[str], default: float = 0.5) -> np.ndarray:
    for cand in (Path(ckpt_dir) / "thresholds.json", Path(ckpt_dir).parent / "thresholds.json"):
        if cand.exists():
            payload = json.loads(cand.read_text(encoding="utf-8"))
            return np.array([float(payload.get(c, default)) for c in label_cols], dtype=np.float32)
    print(f"[WARN] no thresholds.json near {ckpt_dir}; using {default}")
    return np.full(len(label_cols), default, dtype=np.float32)


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


@torch.no_grad()
def predict_probs(ckpt: str, texts: list[str], max_len: int, bs: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def neg_gap(y_true: np.ndarray, y_pred: np.ndarray, explicit_mask: np.ndarray,
            implicit_mask: np.ndarray, neg_idx: list[int]) -> tuple[float, float, float]:
    exp_sub = np.zeros_like(explicit_mask)
    imp_sub = np.zeros_like(implicit_mask)
    for i in neg_idx:
        exp_sub[:, i] = explicit_mask[:, i]
        imp_sub[:, i] = implicit_mask[:, i]
    f1_e = float(f1_score(y_true[exp_sub], y_pred[exp_sub], average="binary", zero_division=0)) if exp_sub.any() else float("nan")
    f1_i = float(f1_score(y_true[imp_sub], y_pred[imp_sub], average="binary", zero_division=0)) if imp_sub.any() else float("nan")
    return f1_e, f1_i, f1_e - f1_i


def ci(vals: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def main() -> None:
    args = parse_args()
    label_cols = load_label_columns(args.labels_file)
    df = load_absa_csv(args.test_file, label_cols)
    texts = df["text"].astype(str).tolist()
    text_ids = df["text_id"].astype(str).tolist()
    y_true = df[label_cols].values.astype(int)
    neg_idx = [i for i, c in enumerate(label_cols) if "ネガティブ" in c]

    tags_df = pd.read_csv(args.tags_file)
    explicit_mask, implicit_mask = build_source_masks(tags_df, text_ids, label_cols)
    print(f"[INFO] test sentences={len(texts):,}  explicit_pairs={explicit_mask.sum()}  implicit_pairs={implicit_mask.sum()}")

    print(f"[INFO] predicting A ({args.label_a}): {args.checkpoint_a}")
    pa = predict_probs(args.checkpoint_a, texts, args.max_seq_length, args.batch_size)
    ta = load_thresholds(args.checkpoint_a, label_cols)
    pred_a = (pa >= ta).astype(int)

    print(f"[INFO] predicting B ({args.label_b}): {args.checkpoint_b}")
    pb = predict_probs(args.checkpoint_b, texts, args.max_seq_length, args.batch_size)
    tb = load_thresholds(args.checkpoint_b, label_cols)
    pred_b = (pb >= tb).astype(int)

    fe_a, fi_a, gap_a = neg_gap(y_true, pred_a, explicit_mask, implicit_mask, neg_idx)
    fe_b, fi_b, gap_b = neg_gap(y_true, pred_b, explicit_mask, implicit_mask, neg_idx)
    print("\n[POINT ESTIMATES]")
    print(f"  {args.label_a}: explicit={fe_a:.4f} implicit={fi_a:.4f} gap={gap_a:+.4f}")
    print(f"  {args.label_b}: explicit={fe_b:.4f} implicit={fi_b:.4f} gap={gap_b:+.4f}")
    print(f"  gap shrinkage (A - B) = {gap_a - gap_b:+.4f}  (positive = B closes the gap more than A)")

    rng = np.random.default_rng(args.seed)
    n = len(y_true)
    boot = {"gap_a": [], "gap_b": [], "diff": []}
    print(f"\n[INFO] paired bootstrap n_boot={args.n_boot} ...")
    for it in range(args.n_boot):
        idx = rng.integers(0, n, n)  # same indices for BOTH models (paired)
        yt = y_true[idx]
        em = explicit_mask[idx]
        im = implicit_mask[idx]
        _, _, ga = neg_gap(yt, pred_a[idx], em, im, neg_idx)
        _, _, gb = neg_gap(yt, pred_b[idx], em, im, neg_idx)
        boot["gap_a"].append(ga)
        boot["gap_b"].append(gb)
        boot["diff"].append(ga - gb)
        if (it + 1) % 500 == 0:
            print(f"       {it + 1}/{args.n_boot}")

    ga_arr = np.array(boot["gap_a"])
    gb_arr = np.array(boot["gap_b"])
    d_arr = np.array(boot["diff"])
    ga_lo, ga_hi = ci(ga_arr)
    gb_lo, gb_hi = ci(gb_arr)
    d_lo, d_hi = ci(d_arr)
    p_gt0 = float((d_arr > 0).mean())
    sig = d_lo > 0

    report = {
        "checkpoint_a": args.checkpoint_a, "label_a": args.label_a,
        "checkpoint_b": args.checkpoint_b, "label_b": args.label_b,
        "n_test_sentences": n, "n_boot": args.n_boot,
        "point": {
            "explicit_f1_a": fe_a, "implicit_f1_a": fi_a, "gap_a": gap_a,
            "explicit_f1_b": fe_b, "implicit_f1_b": fi_b, "gap_b": gap_b,
            "gap_shrinkage_a_minus_b": gap_a - gap_b,
        },
        "bootstrap": {
            "gap_a_ci": [ga_lo, ga_hi],
            "gap_b_ci": [gb_lo, gb_hi],
            "shrinkage_diff_ci": [d_lo, d_hi],
            "P_shrinkage_gt_0": p_gt0,
            "significant_at_95": bool(sig),
        },
    }
    print("\n[RESULT] paired bootstrap 95% CI")
    print(f"  gap_{args.label_a}  {gap_a:+.4f}  CI [{ga_lo:+.4f}, {ga_hi:+.4f}]")
    print(f"  gap_{args.label_b}  {gap_b:+.4f}  CI [{gb_lo:+.4f}, {gb_hi:+.4f}]")
    print(f"  shrinkage (A-B)  {gap_a - gap_b:+.4f}  CI [{d_lo:+.4f}, {d_hi:+.4f}]  "
          f"P(shrinkage>0)={p_gt0:.4f}  -> {'SIGNIFICANT (CI excludes 0)' if sig else 'not significant at 95%'}")

    out = Path(args.out) if args.out else PROJECT_ROOT / f"results/gap_bootstrap_E1_vs_{args.label_b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
