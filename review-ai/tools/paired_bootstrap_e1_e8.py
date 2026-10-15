# -*- coding: utf-8 -*-
"""Paired bootstrap significance test: E1 (baseline) vs E8 (weak-pretrained).

Answers "is the +1.9pp negative-F1 gain real, or within test-sampling noise?".

Uses a PAIRED bootstrap: each resample draws the same sentence indices for both
models, so the shared test-sampling variance cancels and the CI of the DIFFERENCE
is tight. If the Δ CI excludes 0, the gain is significant at that level.

Metric definition mirrors train_absa_baseline.py exactly:
    agg_negative_f1 = f1_score(y_true[:, neg_idx].flatten(),
                               y_pred[:, neg_idx].flatten(), average="binary")

Example:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/paired_bootstrap_e1_e8.py \
        --checkpoint_a checkpoints/absa_asl_v2_baseline/checkpoint-3984 \
        --checkpoint_b checkpoints/absa_weak_finetune_v1/checkpoint-3984
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_absa_baseline import load_absa_csv, load_label_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired bootstrap E1 vs E8")
    ap.add_argument("--checkpoint_a", default="checkpoints/absa_asl_v2_baseline/checkpoint-3984",
                    help="Baseline (E1)")
    ap.add_argument("--checkpoint_b", default="checkpoints/absa_weak_finetune_v1/checkpoint-3984",
                    help="Weak-pretrained (E8)")
    ap.add_argument("--test_file", default=str(PROJECT_ROOT / "data/absa_test.csv"))
    ap.add_argument("--labels_file", default=str(PROJECT_ROOT / "data/absa_label_columns.json"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/A1_paired_bootstrap_E1_vs_E8.json"))
    return ap.parse_args()


def load_thresholds(ckpt_dir: str, label_cols: list[str], default: float = 0.5) -> np.ndarray:
    for cand in (Path(ckpt_dir) / "thresholds.json", Path(ckpt_dir).parent / "thresholds.json"):
        if cand.exists():
            payload = json.loads(cand.read_text(encoding="utf-8"))
            print(f"[INFO] thresholds from {cand}")
            return np.array([float(payload.get(c, default)) for c in label_cols], dtype=np.float32)
    print(f"[WARN] no thresholds.json near {ckpt_dir}; using {default}")
    return np.full(len(label_cols), default, dtype=np.float32)


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


def agg_f1(y_true: np.ndarray, y_pred: np.ndarray, idx: list[int]) -> float:
    return float(f1_score(y_true[:, idx].ravel(), y_pred[:, idx].ravel(),
                          average="binary", zero_division=0))


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true.ravel(), y_pred.ravel(), average="binary", zero_division=0))


def ci(vals: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def main() -> None:
    args = parse_args()
    label_cols = load_label_columns(args.labels_file)
    df = load_absa_csv(args.test_file, label_cols)
    texts = df["text"].astype(str).tolist()
    y_true = df[label_cols].values.astype(int)
    neg_idx = [i for i, c in enumerate(label_cols) if "ネガティブ" in c]
    pos_idx = [i for i, c in enumerate(label_cols) if "ポジティブ" in c]
    print(f"[INFO] test sentences={len(texts):,} labels={len(label_cols)} neg_labels={len(neg_idx)}")

    print(f"[INFO] predicting A (E1): {args.checkpoint_a}")
    pa = predict_probs(args.checkpoint_a, texts, args.max_seq_length, args.batch_size)
    ta = load_thresholds(args.checkpoint_a, label_cols)
    pred_a = (pa >= ta).astype(int)

    print(f"[INFO] predicting B (E8): {args.checkpoint_b}")
    pb = predict_probs(args.checkpoint_b, texts, args.max_seq_length, args.batch_size)
    tb = load_thresholds(args.checkpoint_b, label_cols)
    pred_b = (pb >= tb).astype(int)

    metrics = {
        "agg_negative_f1": lambda yt, ya, yb: (agg_f1(yt, ya, neg_idx), agg_f1(yt, yb, neg_idx)),
        "agg_positive_f1": lambda yt, ya, yb: (agg_f1(yt, ya, pos_idx), agg_f1(yt, yb, pos_idx)),
        "micro_f1": lambda yt, ya, yb: (micro_f1(yt, ya), micro_f1(yt, yb)),
    }

    point = {k: fn(y_true, pred_a, pred_b) for k, fn in metrics.items()}
    print("\n[POINT ESTIMATES]")
    for k, (a, b) in point.items():
        print(f"  {k:18} E1={a:.4f}  E8={b:.4f}  delta={b - a:+.4f}")

    rng = np.random.default_rng(args.seed)
    n = len(y_true)
    boot = {k: {"a": [], "b": [], "d": []} for k in metrics}
    print(f"\n[INFO] paired bootstrap n_boot={args.n_boot} ...")
    for it in range(args.n_boot):
        idx = rng.integers(0, n, n)  # same indices for BOTH models (paired)
        yt, ya, yb = y_true[idx], pred_a[idx], pred_b[idx]
        for k, fn in metrics.items():
            a, b = fn(yt, ya, yb)
            boot[k]["a"].append(a)
            boot[k]["b"].append(b)
            boot[k]["d"].append(b - a)
        if (it + 1) % 500 == 0:
            print(f"       {it + 1}/{args.n_boot}")

    report = {"checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
              "n_test": n, "n_boot": args.n_boot, "metrics": {}}
    print("\n[RESULT] paired bootstrap 95% CI")
    for k in metrics:
        a = np.array(boot[k]["a"]); b = np.array(boot[k]["b"]); d = np.array(boot[k]["d"])
        a_lo, a_hi = ci(a); b_lo, b_hi = ci(b); d_lo, d_hi = ci(d)
        p_gt0 = float((d > 0).mean())
        sig = d_lo > 0
        report["metrics"][k] = {
            "E1_point": point[k][0], "E8_point": point[k][1], "delta_point": point[k][1] - point[k][0],
            "E1_ci": [a_lo, a_hi], "E8_ci": [b_lo, b_hi],
            "delta_ci": [d_lo, d_hi], "P_delta_gt_0": p_gt0,
            "significant_at_95": bool(sig),
        }
        print(f"\n  {k}")
        print(f"    E1     {point[k][0]:.4f}  CI [{a_lo:.4f}, {a_hi:.4f}]")
        print(f"    E8     {point[k][1]:.4f}  CI [{b_lo:.4f}, {b_hi:.4f}]")
        print(f"    delta  {point[k][1]-point[k][0]:+.4f}  CI [{d_lo:+.4f}, {d_hi:+.4f}]  "
              f"P(delta>0)={p_gt0:.4f}  -> {'SIGNIFICANT (CI excludes 0)' if sig else 'not significant at 95%'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
