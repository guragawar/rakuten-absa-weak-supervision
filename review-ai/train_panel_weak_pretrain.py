# -*- coding: utf-8 -*-
"""E7: aspect-rating weak-supervision pretraining of XLM-R (Task 6).

Shared XLM-RoBERTa encoder + N binary heads (one per panel aspect). Each head is
a masked, class-weighted BCE over the review-level weak labels produced by Task 5:
    label = 1  -> aspect rated negative (rating 1-2)
    label = 0  -> aspect rated non-negative (rating 3-5)
    masked     -> aspect not evaluated (rating 0); encoded as -100 and excluded
Per-head pos_weight (neg/pos, ~12-38x) counters the heavy class imbalance so the
heads actually learn the rare negative signal instead of collapsing to "non-neg".

The trained encoder is saved encoder-only (a plain XLM-R base dir) so E8 can load
it via AutoModelForSequenceClassification for the 14-label ABSA head.

Smoke test first, then full run:
    CUDA_VISIBLE_DEVICES=1 python train_panel_weak_pretrain.py --smoke
    CUDA_VISIBLE_DEVICES=1 python train_panel_weak_pretrain.py \
        --output_dir checkpoints/absa_weak_pretrain_v1 --experiment_name E7_panel_weak_pretrain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

HERE = Path(__file__).resolve().parent

# Aspect heads (order fixed by Task 5 / official spec): 立地/部屋/食事/風呂/サービス/設備.
ASPECT_HEADS = ["1", "2", "3", "4", "5", "6"]
ASPECT_NAMES = {"1": "立地", "2": "部屋", "3": "食事", "4": "風呂", "5": "サービス", "6": "設備", "overall": "総合",
                "3b": "朝食", "3d": "夕食"}  # 3b/3d: A5 meal-split ablation (tools/prepare_ablation_weak_labels.py)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E7 panel weak-supervision pretraining")
    ap.add_argument("--train_file", default=str(HERE / "data/panel_weak_train.jsonl"))
    ap.add_argument("--validation_file", default=str(HERE / "data/panel_weak_valid.jsonl"))
    ap.add_argument("--stats_file", default=str(HERE / "data/panel_weak_label_stats.json"))
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--output_dir", default=str(HERE / "checkpoints/absa_weak_pretrain_v1"))
    ap.add_argument("--experiment_name", default="E7_panel_weak_pretrain")
    ap.add_argument("--include_overall", action="store_true", help="Add 7th head for 総合 (A4 ablation)")
    ap.add_argument("--heads", default=",".join(ASPECT_HEADS),
                    help="Comma-separated head ids present in the jsonl (default: the 6 aspects). "
                         "A5 meal split uses 1,2,3b,3d,4,5,6.")
    ap.add_argument("--pos_weight_mode", default="sqrt", choices=["raw", "sqrt", "cap"],
                    help="How to derive per-head pos_weight from neg/pos: raw=neg/pos (aggressive), "
                         "sqrt=sqrt(neg/pos) (balanced, default), cap=min(neg/pos, cap)")
    ap.add_argument("--pos_weight_cap", type=float, default=8.0, help="Cap value for --pos_weight_mode cap")
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=float, default=2.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--eval_steps", type=int, default=2000)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Tiny run: 1000 train / 50 steps, for sanity")
    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="Scaling curve: subsample train to N rows (shuffled with --seed first, since the "
                         "pool is written in year order and head-N would only cover early years). 0 = all.")
    ap.add_argument("--max_valid_samples", type=int, default=0,
                    help="Subsample valid to N rows (shuffled). The full-pool valid is ~675k, which makes "
                         "periodic eval slower than training; 50k is plenty for monitoring. 0 = all.")
    ap.add_argument("--min_source_year", type=int, default=0,
                    help="Recency filter: keep only reviews with source_year >= this (needs source_year in "
                         "the jsonl). Used to test the domain-drift hypothesis (recent-only vs random).")
    ap.add_argument("--max_source_year", type=int, default=0,
                    help="Upper-bound year filter: keep only source_year <= this (0 = no upper bound). "
                         "min+max together select a year window (e.g. an early-years control).")
    ap.add_argument("--results_dir", default=str(HERE / "results"))
    return ap.parse_args()


class PanelWeakModel(nn.Module):
    """Shared encoder + independent binary heads with masked, weighted BCE."""

    def __init__(self, model_name: str, n_heads: int, pos_weight: torch.Tensor):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.heads = nn.Linear(hidden, n_heads)
        self.n_heads = n_heads
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # <s> token
        logits = self.heads(self.dropout(cls))  # (B, n_heads)
        loss = None
        if labels is not None:
            mask = (labels != -100).float()
            target = torch.clamp(labels, min=0.0)  # -100 -> 0, excluded by mask
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight, reduction="none"
            )
            loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def save_encoder(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out_dir)


def load_pos_weight(stats_file: Path, heads: list[str], mode: str = "sqrt", cap: float = 8.0) -> torch.Tensor:
    payload = json.loads(stats_file.read_text(encoding="utf-8"))
    train_heads = payload["splits"]["train"]["heads"]
    weights = []
    for h in heads:
        v = train_heads[h]
        pos, neg = v["pos_neg1_2"], v["neg_nonneg3_5"]
        raw = neg / pos if pos else 1.0
        if mode == "sqrt":
            w = raw ** 0.5
        elif mode == "cap":
            w = min(raw, cap)
        else:
            w = raw
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataset(train_file, valid_file, tokenizer, max_len, heads, smoke,
                  max_train_samples=0, max_valid_samples=0, seed=42, min_source_year=0, max_source_year=0):
    ds = load_dataset("json", data_files={"train": train_file, "validation": valid_file})
    if smoke:
        ds["train"] = ds["train"].select(range(min(1000, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(1000, len(ds["validation"]))))
    else:
        if min_source_year or max_source_year:
            n0 = len(ds["train"])
            lo = min_source_year or 0
            hi = max_source_year or 9999
            ds["train"] = ds["train"].filter(lambda ex: lo <= int(ex.get("source_year") or 0) <= hi)
            print(f"[INFO] year filter {lo}-{hi}: train {n0:,} -> {len(ds['train']):,}")
        if max_train_samples and max_train_samples < len(ds["train"]):
            # Shuffle first: the pool is in year order, so head-N would be early-years only.
            ds["train"] = ds["train"].shuffle(seed=seed).select(range(max_train_samples))
            print(f"[INFO] scaling: subsampled train to {max_train_samples:,} (shuffled, seed={seed})")
        if max_valid_samples and max_valid_samples < len(ds["validation"]):
            ds["validation"] = ds["validation"].shuffle(seed=seed).select(range(max_valid_samples))
            print(f"[INFO] subsampled valid to {max_valid_samples:,} (shuffled, seed={seed})")

    label_keys = [(f"label_{h}", f"mask_{h}") if h != "overall" else ("label_overall", "mask_overall")
                  for h in heads]

    def prep(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len)
        n = len(batch["text"])
        labels = []
        for j in range(n):
            row = []
            for lk, mk in label_keys:
                m = batch[mk][j]
                row.append(float(batch[lk][j]) if m == 1 else -100.0)
            labels.append(row)
        enc["labels"] = labels
        return enc

    keep_cols = ds["train"].column_names
    ds = ds.map(prep, batched=True, remove_columns=keep_cols, desc="tokenize+labels")
    return ds


class WeakCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def make_compute_metrics(heads: list[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
        labels = labels.astype(float)
        out = {}
        f1s = []
        for i, h in enumerate(heads):
            valid = labels[:, i] != -100
            if valid.sum() == 0:
                continue
            yt = labels[valid, i].astype(int)
            yp = preds[valid, i]
            f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
            out[f"f1_{ASPECT_NAMES.get(h, h)}"] = round(float(f1), 4)
            out[f"prec_{ASPECT_NAMES.get(h, h)}"] = round(float(precision_score(yt, yp, pos_label=1, zero_division=0)), 4)
            out[f"rec_{ASPECT_NAMES.get(h, h)}"] = round(float(recall_score(yt, yp, pos_label=1, zero_division=0)), 4)
            f1s.append(f1)
        out["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
        return out

    return compute_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()] + (["overall"] if args.include_overall else [])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    pos_weight = load_pos_weight(Path(args.stats_file), heads, mode=args.pos_weight_mode, cap=args.pos_weight_cap)
    print(f"[INFO] heads={[ASPECT_NAMES.get(h,h) for h in heads]}")
    print(f"[INFO] pos_weight_mode={args.pos_weight_mode} pos_weight={[round(float(x),2) for x in pos_weight]}")

    model = PanelWeakModel(args.model_name, n_heads=len(heads), pos_weight=pos_weight)
    ds = build_dataset(args.train_file, args.validation_file, tokenizer,
                       args.max_seq_length, heads, args.smoke,
                       max_train_samples=args.max_train_samples,
                       max_valid_samples=args.max_valid_samples, seed=args.seed,
                       min_source_year=args.min_source_year, max_source_year=args.max_source_year)
    print(f"[INFO] train={len(ds['train']):,} valid={len(ds['validation']):,}")

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    output_dir = args.output_dir if not args.smoke else str(HERE / "checkpoints/smoke_e7")

    targs = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs if not args.smoke else 1,
        max_steps=50 if args.smoke else -1,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps if not args.smoke else 10,
        eval_strategy="steps",
        eval_steps=args.eval_steps if not args.smoke else 25,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=bf16,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=WeakCollator(tokenizer),
        compute_metrics=make_compute_metrics(heads),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("[RESULT] valid metrics:")
    for k, v in metrics.items():
        if k.startswith(("eval_f1", "eval_macro", "eval_rec", "eval_prec")):
            print(f"       {k} = {v}")

    # Save encoder-only for E8.
    enc_dir = Path(output_dir) / "encoder"
    model.save_encoder(str(enc_dir))
    tokenizer.save_pretrained(str(enc_dir))
    print(f"[INFO] encoder saved -> {enc_dir}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = results_dir / f"{args.experiment_name}_metrics.json"
    payload = {
        "experiment": args.experiment_name,
        "model_name": args.model_name,
        "heads": [ASPECT_NAMES.get(h, h) for h in heads],
        "pos_weight": [round(float(x), 3) for x in pos_weight],
        "max_seq_length": args.max_seq_length,
        "num_epochs": args.num_epochs,
        "smoke": args.smoke,
        "valid_metrics": {k: v for k, v in metrics.items()},
        "encoder_dir": str(enc_dir),
        "train_rows": len(ds["train"]),
        "args": vars(args),  # full command line, so the run can be reproduced from this file alone
    }
    out_metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out_metrics}")
    macro = metrics.get("eval_macro_f1", 0.0)
    print(f"[ACCEPTANCE] valid macro_f1 = {macro} (target > 0.60): "
          f"{'PASS' if macro > 0.60 else 'REVIEW'}")


if __name__ == "__main__":
    main()
