# -*- coding: utf-8 -*-
"""Evaluate ABSA checkpoint on explicit vs implicit test subsets (E2).

Examples:
    python tools/evaluate_by_source.py \
        --checkpoint_dir checkpoints/absa_asl_v2_baseline/checkpoint-XXXX \
        --experiment_name E2_explicit_vs_implicit_diagnosis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_absa_baseline import load_absa_csv, load_label_columns


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate explicit vs implicit ABSA subsets")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--test_file", default=str(PROJECT_ROOT / "data/absa_test.csv"))
    ap.add_argument(
        "--tags_file",
        default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"),
    )
    ap.add_argument("--labels_file", default=str(PROJECT_ROOT / "data/absa_label_columns.json"))
    ap.add_argument("--experiment_name", default="E2_explicit_vs_implicit_diagnosis")
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--bootstrap_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_thresholds(checkpoint_dir: Path, label_cols: list[str], default: float = 0.5) -> np.ndarray:
    parent = checkpoint_dir.parent
    thresholds_path = parent / "thresholds.json"
    if thresholds_path.exists():
        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        return np.array([float(payload.get(col, default)) for col in label_cols], dtype=np.float32)
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


def compute_sentence_subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sentence_mask: np.ndarray,
    label_cols: list[str],
) -> dict:
    if not sentence_mask.any():
        return {"n_sentences": 0}

    yt = y_true[sentence_mask]
    yp = y_pred[sentence_mask]
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    sample_p, sample_r, sample_f1, _ = precision_recall_fscore_support(
        yt, yp, average="samples", zero_division=0
    )

    out = {
        "n_sentences": int(sentence_mask.sum()),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "sample_f1": float(sample_f1),
        "per_label": {},
    }

    pos_idx = [i for i, c in enumerate(label_cols) if "ポジティブ" in c]
    neg_idx = [i for i, c in enumerate(label_cols) if "ネガティブ" in c]
    for name, idxs in [("agg_positive", pos_idx), ("agg_negative", neg_idx)]:
        yts = yt[:, idxs].reshape(-1)
        yps = yp[:, idxs].reshape(-1)
        p, r, f1, _ = precision_recall_fscore_support(yts, yps, average="binary", zero_division=0)
        out[f"{name}_precision"] = float(p)
        out[f"{name}_recall"] = float(r)
        out[f"{name}_f1"] = float(f1)

    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        yt, yp, average=None, zero_division=0
    )
    for i, col in enumerate(label_cols):
        out["per_label"][col] = {
            "support": float(support[i]),
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
        }
    return out


def compute_masked_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    label_cols: list[str],
) -> dict:
    if not mask.any():
        return {"n_pairs": 0}

    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )

    out = {
        "n_pairs": int(mask.sum()),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_label": {},
    }

    pos_idx = [i for i, c in enumerate(label_cols) if "ポジティブ" in c]
    neg_idx = [i for i, c in enumerate(label_cols) if "ネガティブ" in c]

    for name, idxs in [("agg_positive", pos_idx), ("agg_negative", neg_idx)]:
        submask = np.zeros_like(mask)
        for i in idxs:
            submask[:, i] = mask[:, i]
        if submask.any():
            yts = y_true[submask].astype(int)
            yps = y_pred[submask].astype(int)
            p, r, f1, _ = precision_recall_fscore_support(yts, yps, average="binary", zero_division=0)
            out[f"{name}_precision"] = float(p)
            out[f"{name}_recall"] = float(r)
            out[f"{name}_f1"] = float(f1)

    for i, col in enumerate(label_cols):
        col_mask = mask[:, i]
        if not col_mask.any():
            continue
        yts = y_true[col_mask, i].astype(int)
        yps = y_pred[col_mask, i].astype(int)
        p, r, f1, _ = precision_recall_fscore_support(yts, yps, average="binary", zero_division=0)
        out["per_label"][col] = {
            "support": int(col_mask.sum()),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
    return out


def bootstrap_sentence_micro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sentence_mask: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    sent_idx = np.where(sentence_mask)[0]
    if len(sent_idx) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        pick = rng.choice(sent_idx, size=len(sent_idx), replace=True)
        yt = y_true[pick]
        yp = y_pred[pick]
        f1 = precision_recall_fscore_support(yt, yp, average="samples", zero_division=0)[2]
        scores.append(f1)
    scores_arr = np.asarray(scores, dtype=float)
    return {
        "metric": "sample_f1",
        "mean": float(scores_arr.mean()),
        "ci_low": float(np.percentile(scores_arr, 2.5)),
        "ci_high": float(np.percentile(scores_arr, 97.5)),
    }


def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    if not mask.any():
        return {"mean": None, "ci_low": None, "ci_high": None}
    pairs_true = y_true[mask].astype(int)
    pairs_pred = y_pred[mask].astype(int)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(pairs_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1 = precision_recall_fscore_support(
            pairs_true[idx], pairs_pred[idx], average="micro", zero_division=0
        )[2]
        scores.append(f1)
    scores = np.asarray(scores, dtype=float)
    return {
        "mean": float(scores.mean()),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }


def plot_gap(metrics: dict, output_path: Path) -> None:
    subsets = ["explicit", "implicit", "overall"]
    f1_vals = [
        metrics["subsets"]["explicit_sentences"]["sample_f1"],
        metrics["subsets"]["implicit_sentences"]["sample_f1"],
        metrics["subsets"]["overall"]["sample_f1"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(subsets, f1_vals, color=["#4C78A8", "#E45756", "#72B7B2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sample F1")
    ax.set_title("E2: Explicit vs Implicit Test Subset Gap")
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    label_cols = load_label_columns(args.labels_file)
    df_test = load_absa_csv(args.test_file, label_cols)
    tags_df = pd.read_csv(args.tags_file)

    text_ids = df_test["text_id"].astype(str).tolist()
    explicit_mask, implicit_mask = build_source_masks(tags_df, text_ids, label_cols)
    active_mask = (df_test[label_cols].values > 0)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)

    ds = Dataset.from_dict(
        {
            "text": df_test["text"].astype(str).tolist(),
            "labels": df_test[label_cols].astype("float32").values.tolist(),
        }
    )

    def tokenize(batch):
        out = tokenizer(batch["text"], max_length=args.max_seq_length, truncation=True)
        out["labels"] = batch["labels"]
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/absa_eval_by_source",
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to="none",
            bf16=bf16,
        ),
        processing_class=tokenizer,
        data_collator=collator,
    )

    pred = trainer.predict(ds)
    probs = 1.0 / (1.0 + np.exp(-pred.predictions))
    thresholds = load_thresholds(Path(args.checkpoint_dir), label_cols)
    y_pred = (probs >= thresholds).astype(int)
    y_true = pred.label_ids.astype(int)

    overall_mask = active_mask
    explicit_sentence_mask = explicit_mask.any(axis=1)
    implicit_sentence_mask = implicit_mask.any(axis=1)
    negative_implicit_sentence_mask = implicit_mask[:, [i for i, c in enumerate(label_cols) if "ネガティブ" in c]].any(axis=1)

    subset_metrics = {
        "explicit_sentences": compute_sentence_subset_metrics(
            y_true, y_pred, explicit_sentence_mask, label_cols
        ),
        "implicit_sentences": compute_sentence_subset_metrics(
            y_true, y_pred, implicit_sentence_mask, label_cols
        ),
        "negative_implicit_sentences": compute_sentence_subset_metrics(
            y_true, y_pred, negative_implicit_sentence_mask, label_cols
        ),
        "overall": compute_sentence_subset_metrics(y_true, y_pred, np.ones(len(y_true), dtype=bool), label_cols),
        "explicit_active_pairs": compute_masked_metrics(y_true, y_pred, explicit_mask, label_cols),
        "implicit_active_pairs": compute_masked_metrics(y_true, y_pred, implicit_mask, label_cols),
    }

    exp_f1 = subset_metrics["explicit_sentences"]["sample_f1"]
    imp_f1 = subset_metrics["implicit_sentences"]["sample_f1"]
    gap = exp_f1 - imp_f1

    bootstrap = {
        "explicit_sentence_micro_f1": bootstrap_sentence_micro_f1(
            y_true, y_pred, explicit_sentence_mask, n_boot=args.bootstrap_samples, seed=args.seed
        ),
        "implicit_sentence_micro_f1": bootstrap_sentence_micro_f1(
            y_true, y_pred, implicit_sentence_mask, n_boot=args.bootstrap_samples, seed=args.seed
        ),
    }

    payload = {
        "experiment_name": args.experiment_name,
        "checkpoint_dir": args.checkpoint_dir,
        "test_file": args.test_file,
        "tags_file": args.tags_file,
        "thresholds": {col: float(thresholds[i]) for i, col in enumerate(label_cols)},
        "subsets": subset_metrics,
        "gap": {
            "sample_f1_explicit_sentences_minus_implicit_sentences": float(gap),
            "active_pair_negative_f1_explicit_minus_implicit": float(
                subset_metrics["explicit_active_pairs"].get("agg_negative_f1", 0.0)
                - subset_metrics["implicit_active_pairs"].get("agg_negative_f1", 0.0)
            ),
            "implicit_lower_than_explicit": bool(gap > 0),
            "relative_drop_pct": float((gap / exp_f1) * 100) if exp_f1 > 0 else None,
        },
        "bootstrap_ci": bootstrap,
        "counts": {
            "explicit_sentences": int(explicit_sentence_mask.sum()),
            "implicit_sentences": int(implicit_sentence_mask.sum()),
            "negative_implicit_sentences": int(negative_implicit_sentence_mask.sum()),
            "explicit_active_pairs": int(explicit_mask.sum()),
            "implicit_active_pairs": int(implicit_mask.sum()),
            "overall_active_pairs": int(overall_mask.sum()),
        },
    }

    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_json = results_dir / f"{args.experiment_name}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_path = figures_dir / "E2_gap.png"
    plot_gap(payload, fig_path)

    print("[INFO] E2 explicit vs implicit diagnosis")
    print(
        f"       explicit sentences sample F1: {exp_f1:.4f}  "
        f"(n={subset_metrics['explicit_sentences']['n_sentences']:,})"
    )
    print(
        f"       implicit sentences sample F1: {imp_f1:.4f}  "
        f"(n={subset_metrics['implicit_sentences']['n_sentences']:,})"
    )
    print(f"       sentence-level sample F1 gap: {gap:.4f}")
    pair_neg_gap = (
        subset_metrics["explicit_active_pairs"].get("agg_negative_f1", 0.0)
        - subset_metrics["implicit_active_pairs"].get("agg_negative_f1", 0.0)
    )
    print(f"       active-pair negative F1 gap: {pair_neg_gap:.4f}")
    print(f"       bootstrap implicit sentence CI: {bootstrap.get('implicit_sentence_micro_f1')}")
    print(f"[OK] JSON -> {out_json}")
    print(f"[OK] Figure -> {fig_path}")


if __name__ == "__main__":
    main()
