# -*- coding: utf-8 -*-
"""E7-SCL: SCAPT-style supervised contrastive weak-supervision pretraining.

Controlled counterpart to train_panel_weak_pretrain.py (E7-BCE). Everything is
held fixed (backbone, data pool, downstream E8 protocol) EXCEPT the pretraining
objective: instead of masked per-aspect BCE, this aligns sentence [CLS]
representations by their OVERALL review polarity via a supervised contrastive
(SupCon, Khosla et al. 2020) loss — the core component of SCAPT (Li et al.,
EMNLP 2021), which targets implicit sentiment by pulling same-polarity
expressions together regardless of surface cues.

Polarity anchor: label_overall (総合 rating). 1 = negative (rating<=2),
0 = non-negative (rating>=3). Because negatives are ~5% of the pool, SupCon needs
class-balanced batches to form positive pairs for the rare class, so we build a
balanced set: ALL negatives + an equal number of randomly sampled non-negatives.
This is a necessary property of contrastive learning (documented as the one
unavoidable difference from E7-BCE; both use the same recent review pool).

Only the SCL component is implemented (not SCAPT's auxiliary review-reconstruction
and masked-aspect-prediction) — SCL is the part that targets implicit sentiment.

The encoder is saved encoder-only (a plain HF base dir), identical contract to
E7-BCE, so E8 loads it via AutoModelForSequenceClassification unchanged.

    CUDA_VISIBLE_DEVICES=1 python train_panel_scl_pretrain.py --smoke
    CUDA_VISIBLE_DEVICES=1 python train_panel_scl_pretrain.py \
        --model_name ku-nlp/deberta-v3-base-japanese --min_source_year 2021 \
        --output_dir checkpoints/E7_scl_deberta_recent2M --experiment_name E7_scl_deberta_recent2M
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E7-SCL supervised-contrastive weak pretraining (SCAPT-core)")
    ap.add_argument("--train_file", default=str(HERE / "data/weak_all/panel_weak_train.jsonl"))
    ap.add_argument("--validation_file", default=str(HERE / "data/weak_all/panel_weak_valid.jsonl"))
    ap.add_argument("--model_name", default="ku-nlp/deberta-v3-base-japanese")
    ap.add_argument("--output_dir", default=str(HERE / "checkpoints/E7_scl_deberta_recent2M"))
    ap.add_argument("--experiment_name", default="E7_scl_deberta_recent2M")
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=128)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=float, default=2.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.07, help="SupCon temperature")
    ap.add_argument("--proj_dim", type=int, default=128, help="Projection head output dim")
    ap.add_argument("--min_source_year", type=int, default=2021)
    ap.add_argument("--max_source_year", type=int, default=0)
    ap.add_argument("--max_neg", type=int, default=0, help="Cap negatives (0=all); balanced set = 2x this")
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--results_dir", default=str(HERE / "results"))
    return ap.parse_args()


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020). feats L2-normalized (B, d)."""
    device = feats.device
    B = feats.shape[0]
    sim = (feats @ feats.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # numerical stability
    self_mask = torch.eye(B, device=device)
    logits_mask = 1.0 - self_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask = label_eq * logits_mask
    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
    valid = pos_count > 0
    if valid.any():
        return -mean_log_prob_pos[valid].mean()
    return (feats.sum() * 0.0)  # no positive pairs in batch -> zero grad


class SclModel(nn.Module):
    def __init__(self, model_name: str, proj_dim: int, temperature: float):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, proj_dim))
        self.temperature = temperature

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        z = F.normalize(self.proj(self.dropout(cls)), dim=-1)
        loss = None
        if labels is not None:
            loss = supcon_loss(z, labels, self.temperature)
        return {"loss": loss, "logits": z} if loss is not None else {"logits": z}

    def save_encoder(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out_dir)


def build_balanced(train_file, valid_file, tokenizer, max_len, min_year, max_year,
                   max_neg, seed, smoke):
    ds = load_dataset("json", data_files={"train": train_file, "validation": valid_file})

    def in_window(ex):
        lo = min_year or 0
        hi = max_year or 9999
        return lo <= int(ex.get("source_year") or 0) <= hi

    def make_balanced(split, cap_neg):
        d = split.filter(in_window) if (min_year or max_year) else split
        labels = np.array(d["label_overall"], dtype=np.int64)
        neg_idx = np.where(labels == 1)[0]
        pos_idx = np.where(labels == 0)[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(neg_idx)
        rng.shuffle(pos_idx)
        if cap_neg:
            neg_idx = neg_idx[:cap_neg]
        n = len(neg_idx)
        pos_idx = pos_idx[:n]  # equal number of non-negatives
        sel = np.concatenate([neg_idx, pos_idx])
        rng.shuffle(sel)
        return d.select(sel.tolist()), n

    if smoke:
        tr, n_neg = make_balanced(ds["train"], 500)
        va, _ = make_balanced(ds["validation"], 100)
    else:
        tr, n_neg = make_balanced(ds["train"], max_neg or 0)
        va, _ = make_balanced(ds["validation"], 5000)
    print(f"[INFO] balanced train: {len(tr):,} ({n_neg:,} neg + {n_neg:,} non-neg)  valid: {len(va):,}")

    def prep(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len)
        enc["labels"] = [int(x) for x in batch["label_overall"]]
        return enc

    cols = tr.column_names
    tr = tr.map(prep, batched=True, remove_columns=cols, desc="tok train")
    va = va.map(prep, batched=True, remove_columns=cols, desc="tok valid")
    return tr, va


class SclCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = SclModel(args.model_name, args.proj_dim, args.temperature)
    tr, va = build_balanced(args.train_file, args.validation_file, tokenizer,
                            args.max_seq_length, args.min_source_year, args.max_source_year,
                            args.max_neg, args.seed, args.smoke)

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    output_dir = args.output_dir if not args.smoke else str(HERE / "checkpoints/smoke_scl")

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
        save_total_limit=1,
        bf16=bf16,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tr,
        eval_dataset=va,
        data_collator=SclCollator(tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[RESULT] valid supcon loss = {metrics.get('eval_loss')}")

    enc_dir = Path(output_dir) / "encoder"
    model.save_encoder(str(enc_dir))
    tokenizer.save_pretrained(str(enc_dir))
    print(f"[INFO] encoder saved -> {enc_dir}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{args.experiment_name}_metrics.json"
    out.write_text(json.dumps({
        "experiment": args.experiment_name,
        "objective": "supervised_contrastive (SCAPT-core SCL)",
        "model_name": args.model_name,
        "temperature": args.temperature,
        "proj_dim": args.proj_dim,
        "min_source_year": args.min_source_year,
        "num_epochs": args.num_epochs,
        "polarity_anchor": "label_overall (総合 rating<=2 -> neg, >=3 -> non-neg)",
        "balanced_batches": True,
        "valid_supcon_loss": metrics.get("eval_loss"),
        "encoder_dir": str(enc_dir),
        "smoke": args.smoke,
        "args": vars(args),  # full command line
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
