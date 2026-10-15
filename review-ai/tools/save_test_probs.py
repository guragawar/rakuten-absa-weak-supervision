# -*- coding: utf-8 -*-
"""Predict test-set probabilities for a checkpoint and save them as .npz.

Purpose: multi-seed significance testing without keeping 3-6GB checkpoints.
The saved file (~600KB) contains everything the gap/F1 bootstrap needs:
probs, per-label tuned thresholds, and label columns.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/save_test_probs.py \
        --checkpoint_dir checkpoints/tmp_e1_s43/checkpoint-3984 \
        --out results/test_probs/E1_xlmr_seed43.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_absa_baseline import load_absa_csv, load_label_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Save test-set sigmoid probs for later bootstrap analysis")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--test_file", default=str(PROJECT_ROOT / "data/absa_test.csv"))
    ap.add_argument("--labels_file", default=str(PROJECT_ROOT / "data/absa_label_columns.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    return ap.parse_args()


def load_thresholds(ckpt_dir: str, label_cols: list[str], default: float = 0.5) -> np.ndarray:
    for cand in (Path(ckpt_dir) / "thresholds.json", Path(ckpt_dir).parent / "thresholds.json"):
        if cand.exists():
            payload = json.loads(cand.read_text(encoding="utf-8"))
            return np.array([float(payload.get(c, default)) for c in label_cols], dtype=np.float32)
    print(f"[WARN] no thresholds.json near {ckpt_dir}; using {default}")
    return np.full(len(label_cols), default, dtype=np.float32)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    label_cols = load_label_columns(args.labels_file)
    df = load_absa_csv(args.test_file, label_cols)
    texts = df["text"].astype(str).tolist()
    text_ids = df["text_id"].astype(str).tolist()
    y_true = df[label_cols].values.astype(np.int8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir).to(device).eval()
    out = []
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_seq_length,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    probs = np.concatenate(out, axis=0).astype(np.float32)
    thresholds = load_thresholds(args.checkpoint_dir, label_cols)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        probs=probs,
        thresholds=thresholds,
        y_true=y_true,
        text_ids=np.array(text_ids),
        label_cols=np.array(label_cols),
        checkpoint_dir=np.array(args.checkpoint_dir),
    )
    print(f"[OK] saved probs {probs.shape} -> {out_path}")


if __name__ == "__main__":
    main()
