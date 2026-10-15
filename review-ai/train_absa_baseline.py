# -*- coding: utf-8 -*-
"""
Train an XLM-R baseline for hotel aspect-based sentiment analysis (ABSA).

Task:
    Multi-label sentence classification over 14 labels:
    7 aspects x {ポジティブ, ネガティブ}

Expected input CSV format:
    text_id, review_id, sentence_id, text, <14 label columns>, num_labels

Use tools/prepare_absa_dataset.py to create the split files first.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_LABELS_FILE = Path("data/absa_label_columns.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_label_columns(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["label_columns"])


def load_absa_csv(path: str | Path, label_cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    missing = ["text", *label_cols]
    missing = [col for col in missing if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df["text"] = df["text"].astype(str)
    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).clip(0, 1)
    return df


def compute_pos_weight(df: pd.DataFrame, label_cols: list[str], max_weight: float = 999.0) -> np.ndarray:
    """BCE pos_weight = min(neg/pos, max_weight) per label."""
    n = len(df)
    weights = []
    for col in label_cols:
        pos = float(df[col].sum())
        neg = float(n - pos)
        if pos <= 0:
            weights.append(1.0)
        else:
            weights.append(min(max(neg / pos, 1.0), max_weight))
    return np.asarray(weights, dtype=np.float32)


def focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal-BCE loss for multi-label classification.
    Reduces gradient from easy (high-confidence) predictions.
    """
    probs = torch.sigmoid(logits)
    # p_t: probability of the true class
    p_t = probs * labels + (1 - probs) * (1 - labels)
    focal_weight = (1 - p_t) ** gamma
    bce = F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
        reduction="none",
    )
    return (focal_weight * bce).mean()


def asymmetric_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Asymmetric Loss (ASL) for multi-label classification.
    Ref: Ben-Baruch et al. 2021 (https://arxiv.org/abs/2009.14119).
    Applies stronger down-weighting on easy negatives (gamma_neg > gamma_pos).
    """
    probs = torch.sigmoid(logits)
    # Shift negative probs up by clip to remove very-easy negatives
    probs_neg = (probs + clip).clamp(max=1.0)

    los_pos = labels       * torch.log(probs.clamp(min=eps))
    los_neg = (1 - labels) * torch.log((1 - probs_neg).clamp(min=eps))

    if gamma_pos > 0:
        los_pos = los_pos * (1 - probs) ** gamma_pos
    if gamma_neg > 0:
        los_neg = los_neg * probs_neg ** gamma_neg

    return -(los_pos + los_neg).mean()


class MultilabelTrainer(Trainer):
    def __init__(
        self,
        *args,
        pos_weight: Optional[np.ndarray] = None,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_type == "focal":
            loss = focal_bce_loss(logits, labels, gamma=self.focal_gamma, pos_weight=self.pos_weight)
        elif self.loss_type == "asl":
            loss = asymmetric_loss(
                logits, labels,
                gamma_pos=self.asl_gamma_pos,
                gamma_neg=self.asl_gamma_neg,
                clip=self.asl_clip,
            )
        else:  # bce (default)
            loss_fct = BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            )
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def create_dataset(df: pd.DataFrame, label_cols: list[str]) -> Dataset:
    labels = df[label_cols].astype("float32").values.tolist()
    data = {
        "text": df["text"].astype(str).tolist(),
        "labels": labels,
    }
    return Dataset.from_dict(data)


def summarize_dataset(name: str, df: pd.DataFrame, label_cols: list[str]) -> None:
    labels_per_row = df[label_cols].sum(axis=1)
    print(f"[INFO] {name}: {len(df)} rows, {df['review_id'].nunique() if 'review_id' in df else 'N/A'} reviews")
    print(f"       annotated rows: {(labels_per_row > 0).sum()}, multi-label rows: {(labels_per_row > 1).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ABSA multi-label baseline with XLM-R")
    ap.add_argument("--train_file", default="data/absa_train.csv")
    ap.add_argument("--validation_file", default="data/absa_valid.csv")
    ap.add_argument("--test_file", default="data/absa_test.csv")
    ap.add_argument("--labels_file", default=str(DEFAULT_LABELS_FILE))
    ap.add_argument("--output_dir", default="checkpoints/absa_xlmr_baseline")
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--pretrained_encoder", default="",
                    help="Dir of a weak-supervision pretrained encoder (E7). If set, the encoder is "
                         "initialized from here instead of --model_name; the 14-label head stays randomly "
                         "initialized. Everything else is identical to the E1 baseline (E8 = A1 comparison).")
    ap.add_argument("--encoder_lr", type=float, default=0.0,
                    help="A2b: layered LR. If >0, encoder uses this LR while the head uses "
                         "--learning_rate. 0 = uniform LR for all params (A2a, the default/E8).")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="A2c: freeze the encoder, train the classification head only (expected lower bound).")
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--no_pos_weight", action="store_true", help="Disable BCE pos_weight class balancing")
    ap.add_argument("--max_pos_weight", type=float, default=999.0,
                    help="Cap pos_weight per label (avoids extreme imbalance amplification)")
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--experiment_name", default=None,
                    help="Name prefix for results files (default: basename of --output_dir)")
    # Loss type
    ap.add_argument("--loss_type", choices=["bce", "focal", "asl"], default="bce",
                    help="Loss function: bce (default), focal, or asl (Asymmetric Loss)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma (higher = more focus on hard examples)")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0,
                    help="ASL gamma for positive samples")
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0,
                    help="ASL gamma for negative samples (hard down-weighting of easy negatives)")
    ap.add_argument("--asl_clip", type=float, default=0.05,
                    help="ASL probability margin shift for negative samples")
    # Resampling
    ap.add_argument("--resample_neg_factor", type=float, default=1.0,
                    help="Oversample sentences containing ネガティブ labels by this factor (1.0 = off)")
    # Threshold tuning
    ap.add_argument("--tune_threshold", action="store_true",
                    help="After training, find per-label optimal threshold on validation set")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    label_cols = load_label_columns(args.labels_file)
    label_to_id = {label: i for i, label in enumerate(label_cols)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("=" * 60)
    print("ABSA XLM-R baseline")
    print("=" * 60)
    print(f"[INFO] Labels ({len(label_cols)}): {label_cols}")
    print(f"[INFO] Model: {args.model_name}")
    print(f"[INFO] Output: {args.output_dir}")

    df_train = load_absa_csv(args.train_file, label_cols)
    df_valid = load_absa_csv(args.validation_file, label_cols)
    df_test = load_absa_csv(args.test_file, label_cols)

    # Optional: oversample sentences containing negative-aspect labels
    if args.resample_neg_factor > 1.0:
        neg_cols = [c for c in label_cols if "ネガティブ" in c]
        has_neg = df_train[neg_cols].sum(axis=1) > 0
        df_neg_rows = df_train[has_neg]
        extra_n = int(len(df_neg_rows) * (args.resample_neg_factor - 1))
        df_extra = df_neg_rows.sample(n=extra_n, replace=True, random_state=args.seed)
        df_train = pd.concat([df_train, df_extra]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Resampled: {len(df_train)} rows (+{extra_n} neg rows, factor={args.resample_neg_factor})")

    summarize_dataset("train", df_train, label_cols)
    summarize_dataset("valid", df_valid, label_cols)
    summarize_dataset("test", df_test, label_cols)

    experiment_config = {
        "task": "hotel_absa_multilabel",
        "model_name": args.model_name,
        "num_labels": len(label_cols),
        "label_columns": label_cols,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "threshold": args.threshold,
        "train_file": args.train_file,
        "validation_file": args.validation_file,
        "test_file": args.test_file,
        "use_pos_weight": not args.no_pos_weight,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
        "asl_gamma_pos": args.asl_gamma_pos if args.loss_type == "asl" else None,
        "asl_gamma_neg": args.asl_gamma_neg if args.loss_type == "asl" else None,
        "asl_clip": args.asl_clip if args.loss_type == "asl" else None,
        "max_pos_weight": args.max_pos_weight,
        "resample_neg_factor": args.resample_neg_factor,
        "tune_threshold": args.tune_threshold,
        "pretrained_encoder": args.pretrained_encoder or None,
    }
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)

    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = compute_pos_weight(df_train, label_cols, max_weight=args.max_pos_weight)
        pos_weight_dict = {label: float(pos_weight[i]) for i, label in enumerate(label_cols)}
        with open(os.path.join(args.output_dir, "pos_weight.json"), "w", encoding="utf-8") as f:
            json.dump(pos_weight_dict, f, ensure_ascii=False, indent=2)
        print(f"[INFO] pos_weight (max={args.max_pos_weight}): { {k: round(v,2) for k,v in pos_weight_dict.items()} }")
        print(f"[INFO] pos_weight saved: {os.path.join(args.output_dir, 'pos_weight.json')}")

    model_src = args.pretrained_encoder or args.model_name
    if args.pretrained_encoder:
        print(f"[INFO] encoder initialized from pretrained (E7): {model_src}")
    tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=True)

    ds_train = create_dataset(df_train, label_cols)
    ds_valid = create_dataset(df_valid, label_cols)
    ds_test = create_dataset(df_test, label_cols)

    def tokenize(batch):
        out = tokenizer(batch["text"], max_length=args.max_seq_length, truncation=True)
        out["labels"] = batch["labels"]
        return out

    print("[INFO] Tokenizing...")
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])
    ds_valid = ds_valid.map(tokenize, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_src,
        num_labels=len(label_cols),
        id2label=id_to_label,
        label2id=label_to_id,
        problem_type="multi_label_classification",
    )

    # --- A2 ablation: fine-tuning strategy -------------------------------
    prefix = model.base_model_prefix + "."  # "roberta." for XLM-R
    if args.freeze_encoder:
        frozen = 0
        for n, p in model.named_parameters():
            if n.startswith(prefix):
                p.requires_grad = False
                frozen += p.numel()
        print(f"[INFO] A2c: encoder FROZEN ({frozen/1e6:.1f}M params), head-only training")

    optimizers = (None, None)
    if args.encoder_lr and args.encoder_lr > 0:
        from torch.optim import AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        enc = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        head = [(n, p) for n, p in model.named_parameters() if not n.startswith(prefix)]

        def _groups(params, lr):
            return [
                {"params": [p for n, p in params if not any(x in n for x in no_decay)],
                 "lr": lr, "weight_decay": args.weight_decay},
                {"params": [p for n, p in params if any(x in n for x in no_decay)],
                 "lr": lr, "weight_decay": 0.0},
            ]

        optimizers = (AdamW(_groups(enc, args.encoder_lr) + _groups(head, args.learning_rate)), None)
        print(f"[INFO] A2b: layered LR -> encoder={args.encoder_lr} head={args.learning_rate}")

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    print(f"[INFO] bf16: {bf16}")

    common_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        seed=args.seed,
        bf16=bf16,
        tf32=True,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        log_level="info",
        max_grad_norm=1.0,
        logging_steps=50,
        logging_first_step=True,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
    )
    try:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)
    except TypeError:
        training_args = TrainingArguments(eval_strategy="epoch", **common_args)

    # Per-label thresholds; updated in-place by tune_threshold step
    _thresholds = np.full(len(label_cols), args.threshold, dtype=np.float32)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= _thresholds).astype(int)
        labels = labels.astype(int)

        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        sample_p, sample_r, sample_f1, _ = precision_recall_fscore_support(
            labels, preds, average="samples", zero_division=0
        )
        exact_match = accuracy_score(labels, preds)

        out = {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "sample_precision": sample_p,
            "sample_recall": sample_r,
            "sample_f1": sample_f1,
            "exact_match_accuracy": exact_match,
        }

        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        for i, label in enumerate(label_cols):
            safe = label.replace("・", "_").replace("ー", "_").replace("ポジティブ", "positive").replace("ネガティブ", "negative")
            out[f"{safe}_precision"] = per_p[i]
            out[f"{safe}_recall"] = per_r[i]
            out[f"{safe}_f1"] = per_f1[i]
            out[f"{safe}_support"] = float(support[i])

        # Aggregated polarity F1: all ポジティブ labels pooled vs all ネガティブ labels pooled
        pos_idx = [i for i, c in enumerate(label_cols) if "ポジティブ" in c]
        neg_idx = [i for i, c in enumerate(label_cols) if "ネガティブ" in c]
        if pos_idx:
            pp, pr, pf, _ = precision_recall_fscore_support(
                labels[:, pos_idx].flatten(), preds[:, pos_idx].flatten(),
                average="binary", zero_division=0,
            )
            out["agg_positive_precision"] = pp
            out["agg_positive_recall"] = pr
            out["agg_positive_f1"] = pf
        if neg_idx:
            np_, nr, nf, _ = precision_recall_fscore_support(
                labels[:, neg_idx].flatten(), preds[:, neg_idx].flatten(),
                average="binary", zero_division=0,
            )
            out["agg_negative_precision"] = np_
            out["agg_negative_recall"] = nr
            out["agg_negative_f1"] = nf

        return out

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
        optimizers=optimizers,
    )

    print(f"[INFO] Loss: {args.loss_type}"
          + (f"  gamma={args.focal_gamma}" if args.loss_type == "focal" else "")
          + (f"  gamma_pos={args.asl_gamma_pos} gamma_neg={args.asl_gamma_neg}" if args.loss_type == "asl" else ""))
    print("[INFO] Start training...")
    trainer.train()
    print("[INFO] Training complete.")

    # Per-label threshold tuning on validation set
    if args.tune_threshold:
        print("[INFO] Tuning per-label thresholds on validation set...")
        pred_out = trainer.predict(ds_valid)
        val_probs = 1.0 / (1.0 + np.exp(-pred_out.predictions))
        val_labels = pred_out.label_ids.astype(int)
        thresholds_tuned = {}
        for i, col in enumerate(label_cols):
            best_f1, best_t = 0.0, float(_thresholds[i])
            for t in np.arange(0.10, 0.91, 0.05):
                p_i = (val_probs[:, i] >= t).astype(int)
                f1 = precision_recall_fscore_support(
                    val_labels[:, i], p_i, average="binary", zero_division=0
                )[2]
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            _thresholds[i] = best_t
            thresholds_tuned[col] = best_t
            print(f"  {col}: t*={best_t:.2f}  val_F1={best_f1:.4f}")
        with open(os.path.join(args.output_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(thresholds_tuned, f, ensure_ascii=False, indent=2)
        print(f"[OK] Tuned thresholds saved to {args.output_dir}/thresholds.json")

    # Validation metrics with the final thresholds: model/config comparisons are made on these,
    # never on test. (Thresholds were tuned on valid, so these are optimistic but comparable.)
    valid_metrics = trainer.evaluate(ds_valid, metric_key_prefix="valid")

    print("[INFO] Evaluating on test set...")
    test_metrics = trainer.evaluate(ds_test)
    run_name = args.experiment_name or os.path.basename(args.output_dir.rstrip("/"))
    results_dir = os.path.join(_SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Test probabilities for later paired bootstrap, so the checkpoint can be deleted
    # (same format as tools/save_test_probs.py)
    test_pred = trainer.predict(ds_test)
    probs_dir = os.path.join(results_dir, "test_probs")
    os.makedirs(probs_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(probs_dir, f"{run_name}.npz"),
        probs=(1.0 / (1.0 + np.exp(-test_pred.predictions))).astype(np.float32),
        thresholds=_thresholds.copy(),
        y_true=test_pred.label_ids.astype(np.int8),
        text_ids=np.array(df_test["text_id"].astype(str).tolist()) if "text_id" in df_test else np.array([]),
        label_cols=np.array(label_cols),
        checkpoint_dir=np.array(str(trainer.state.best_model_checkpoint)),
    )
    print(f"[OK] Test probs: {probs_dir}/{run_name}.npz")

    metrics_payload = {
        **test_metrics,
        **valid_metrics,
        # threshold_initial: 命令行默认值；thresholds_per_label: 实际用于 test 评估的阈值
        # 若 tune_threshold=True，两者会不同，以 thresholds_per_label 为准
        "threshold_initial": args.threshold,
        "threshold_tuned": args.tune_threshold,
        "thresholds_per_label": {col: float(_thresholds[i]) for i, col in enumerate(label_cols)},
        "loss_type": args.loss_type,
        "train_file": args.train_file,
        "validation_file": args.validation_file,
        "test_file": args.test_file,
        "label_columns": label_cols,
        "checkpoint_dir": trainer.state.best_model_checkpoint,
        "args": vars(args),  # full command line (seed, encoder, model), kept after checkpoints are deleted
    }
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metrics: {metrics_path}")

    report_path = os.path.join(results_dir, f"{run_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ABSA baseline test report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Data files:\n")
        f.write(f"  train: {args.train_file}\n")
        f.write(f"  valid: {args.validation_file}\n")
        f.write(f"  test:  {args.test_file}\n\n")
        f.write(f"Training config:\n")
        f.write(f"  loss_type:       {args.loss_type}\n")
        if args.loss_type == "focal":
            f.write(f"  focal_gamma:     {args.focal_gamma}\n")
        elif args.loss_type == "asl":
            f.write(f"  asl_gamma_pos:   {args.asl_gamma_pos}\n")
            f.write(f"  asl_gamma_neg:   {args.asl_gamma_neg}\n")
        f.write(f"  max_pos_weight:  {args.max_pos_weight}\n")
        if args.tune_threshold:
            f.write(f"  thresholds:      per-label tuned (see thresholds.json)\n")
        else:
            f.write(f"  threshold:       {args.threshold}\n")
        f.write("\n")
        f.write("Overall metrics:\n")
        f.write(f"  Micro F1:           {test_metrics.get('eval_micro_f1', 0):.4f}\n")
        f.write(f"  Macro F1:           {test_metrics.get('eval_macro_f1', 0):.4f}\n")
        f.write(f"  Sample F1:          {test_metrics.get('eval_sample_f1', 0):.4f}\n")
        f.write(f"  Exact match acc:    {test_metrics.get('eval_exact_match_accuracy', 0):.4f}\n\n")
        f.write("Aggregated polarity F1 (all aspects pooled):\n")
        for pol, key in [("ポジティブ (positive)", "agg_positive"), ("ネガティブ (negative)", "agg_negative")]:
            f1  = test_metrics.get(f"eval_{key}_f1", 0)
            pre = test_metrics.get(f"eval_{key}_precision", 0)
            rec = test_metrics.get(f"eval_{key}_recall", 0)
            f.write(f"  {pol}:  F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}\n")
        f.write("\nPer-aspect F1:\n")
        aspects = ["朝食", "夕食", "風呂", "サービス", "立地", "設備・アメニティ", "部屋"]
        for asp in aspects:
            pos_label = f"{asp}_ポジティブ"
            neg_label = f"{asp}_ネガティブ"
            pos_safe = pos_label.replace("・", "_").replace("ー", "_").replace("ポジティブ", "positive")
            neg_safe = neg_label.replace("・", "_").replace("ー", "_").replace("ネガティブ", "negative")
            pf = test_metrics.get(f"eval_{pos_safe}_f1", 0)
            nf = test_metrics.get(f"eval_{neg_safe}_f1", 0)
            f.write(f"  {asp:<12}  pos_F1={pf:.4f}  neg_F1={nf:.4f}\n")
        f.write("\nFull per-label F1:\n")
        for label in label_cols:
            safe = label.replace("・", "_").replace("ー", "_").replace("ポジティブ", "positive").replace("ネガティブ", "negative")
            f.write(f"  {label}: {test_metrics.get(f'eval_{safe}_f1', 0):.4f}\n")
    print(f"[OK] Report: {report_path}")


if __name__ == "__main__":
    main()
