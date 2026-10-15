# -*- coding: utf-8 -*-
"""Verify panel rating column -> aspect mapping using the E1 ABSA model (P0-3).

For each ``rating_1..6`` column we sample low-scored reviews (rating in {1,2}),
run the frozen ``absa_asl_v1`` ABSA model, and measure which aspect's negative
label fires. Because frequent aspects (部屋 / サービス) appear in nearly every
review, we score each column by *lift* over the cross-column baseline, then
assign aspects to columns bijectively (greedy on lift). The gold model has 7
aspects; the panel has 6 columns, so 食事 is the union of 朝食 + 夕食.

Example:
    python tools/verify_panel_rating_columns.py \
        --panel_csv data/panel_train_pool.csv \
        --checkpoint_dir checkpoints/absa_asl_v1/checkpoint-3984 \
        --out data/panel_rating_column_map.json \
        --n_per_col 200 --gpu 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from travel_review_panel_utils import project_root

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

RATING_COLS = [f"rating_{i}" for i in range(1, 7)]

# The two aspects a guest may legitimately leave unrated (value 0), ordered by
# expected "not evaluated" rate: 食事 (素泊まり / no meal plan) > 風呂 (no bath).
# These anchor the two columns with clearly elevated missing rates.
STRUCTURAL_ASPECTS_BY_MISSING = ["食事", "風呂"]

# Panel aspect candidates -> the gold negative label(s) that represent them.
PANEL_ASPECT_NEG_LABELS = {
    "サービス": ["サービス_ネガティブ"],
    "立地": ["立地_ネガティブ"],
    "部屋": ["部屋_ネガティブ"],
    "設備・アメニティ": ["設備・アメニティ_ネガティブ"],
    "風呂": ["風呂_ネガティブ"],
    "食事": ["朝食_ネガティブ", "夕食_ネガティブ"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify panel rating column -> aspect mapping")
    ap.add_argument(
        "--panel_csv",
        default=str(project_root() / "data/panel_train_pool.csv"),
        help="Panel pool/parsed CSV with rating_1..6 + review_text",
    )
    ap.add_argument(
        "--checkpoint_dir",
        default=str(project_root() / "checkpoints/absa_asl_v1/checkpoint-3984"),
        help="ABSA model checkpoint (E1 / absa_asl_v1)",
    )
    ap.add_argument(
        "--labels_file",
        default=str(project_root() / "data/absa_label_columns.json"),
    )
    ap.add_argument(
        "--out",
        default=str(project_root() / "data/panel_rating_column_map.json"),
    )
    ap.add_argument("--n_per_col", type=int, default=200, help="Negative reviews sampled per column")
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=1, help="CUDA device index (default 1 per project convention)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_lift", type=float, default=0.08, help="Min lift margin for 'high' confidence")
    ap.add_argument(
        "--structural_min_missing",
        type=float,
        default=0.015,
        help="Min per-column 'not evaluated' (0) rate to treat a column as an optional aspect (食事/風呂)",
    )
    ap.add_argument(
        "--low_base_rate",
        type=float,
        default=0.05,
        help="If an aspect's cross-column baseline neg-hit rate is below this, the model cannot "
        "identify it (立地); such an aspect is assigned by elimination and flagged low-confidence",
    )
    return ap.parse_args()


def load_label_columns(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["label_columns"]


def load_thresholds(checkpoint_dir: Path, label_cols: list[str], default: float = 0.5) -> np.ndarray:
    # thresholds.json lives at the experiment root (parent of checkpoint-XXXX).
    for candidate in (checkpoint_dir / "thresholds.json", checkpoint_dir.parent / "thresholds.json"):
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return np.array([float(payload.get(c, default)) for c in label_cols], dtype=np.float32)
    return np.full(len(label_cols), default, dtype=np.float32)


def reservoir_sample_negatives(
    panel_csv: Path, n_per_col: int, seed: int
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int], int]:
    """Reservoir-sample n_per_col rating-{1,2} reviews per column.

    Also tallies, over the whole file, how many reviews leave each column
    unrated (value 0 / empty). The missing-rate is the structural signal that
    identifies the two "optional" aspects (食事 when 素泊まり, 風呂 when no bath).
    """
    rng = random.Random(seed)
    reservoirs: dict[str, list[str]] = {c: [] for c in RATING_COLS}
    seen_counts: dict[str, int] = {c: 0 for c in RATING_COLS}
    missing_counts: dict[str, int] = {c: 0 for c in RATING_COLS}
    total_rows = 0

    with panel_csv.open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            text = row.get("review_text", "").strip()
            if not text:
                continue
            total_rows += 1
            for col in RATING_COLS:
                val = row.get(col, "").strip()
                if val in ("", "0"):
                    missing_counts[col] += 1
                    continue
                if val in ("1", "2"):
                    seen_counts[col] += 1
                    res = reservoirs[col]
                    if len(res) < n_per_col:
                        res.append(text)
                    else:
                        j = rng.randint(0, seen_counts[col] - 1)
                        if j < n_per_col:
                            res[j] = text
    return reservoirs, seen_counts, missing_counts, total_rows


@torch.no_grad()
def predict_probs(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_seq_length: int,
    batch_size: int,
) -> np.ndarray:
    all_probs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, model.config.num_labels))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    panel_csv = Path(args.panel_csv)
    if not panel_csv.exists():
        raise SystemExit(f"Panel CSV not found: {panel_csv}")

    label_cols = load_label_columns(Path(args.labels_file))
    col_to_idx = {c: i for i, c in enumerate(label_cols)}
    thresholds = load_thresholds(Path(args.checkpoint_dir), label_cols)

    print(f"[INFO] Sampling up to {args.n_per_col} negative reviews per column ...")
    reservoirs, seen_counts, missing_counts, total_rows = reservoir_sample_negatives(
        panel_csv, args.n_per_col, args.seed
    )
    missing_rate = {
        c: (missing_counts[c] / total_rows if total_rows else 0.0) for c in RATING_COLS
    }
    for col in RATING_COLS:
        print(f"       {col}: sampled {len(reservoirs[col])} of {seen_counts[col]:,} negatives; "
              f"missing(0)={missing_rate[col]*100:.1f}%")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu < torch.cuda.device_count() else "cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Loading model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    # aspect negative-fire rate per column: hit_rate[col][aspect]
    aspect_names = list(PANEL_ASPECT_NEG_LABELS.keys())
    hit_rate: dict[str, dict[str, float]] = {c: {} for c in RATING_COLS}
    n_used: dict[str, int] = {}

    for col in RATING_COLS:
        texts = reservoirs[col]
        n_used[col] = len(texts)
        if not texts:
            for a in aspect_names:
                hit_rate[col][a] = 0.0
            continue
        probs = predict_probs(texts, tokenizer, model, device, args.max_seq_length, args.batch_size)
        preds = (probs >= thresholds).astype(int)  # (N, 14)
        for aspect, neg_labels in PANEL_ASPECT_NEG_LABELS.items():
            idxs = [col_to_idx[l] for l in neg_labels if l in col_to_idx]
            fired = preds[:, idxs].max(axis=1) if idxs else np.zeros(len(texts), dtype=int)
            hit_rate[col][aspect] = float(fired.mean())

    # Baseline per aspect = mean hit rate across columns; lift = hit - baseline.
    baseline = {
        a: float(np.mean([hit_rate[c][a] for c in RATING_COLS])) for a in aspect_names
    }
    lift = {
        c: {a: hit_rate[c][a] - baseline[a] for a in aspect_names} for c in RATING_COLS
    }

    # --- Assignment ------------------------------------------------------
    # Step 1: structural anchor. The two aspects a guest may leave unrated are
    # 食事 and 風呂; the two columns with clearly elevated missing(0) rates map
    # to them (highest -> 食事, next -> 風呂). This structural signal is far more
    # reliable than the noisy negative-hit model for these two columns.
    missing_sorted = sorted(RATING_COLS, key=lambda c: missing_rate[c], reverse=True)
    structural_ok = (
        len(RATING_COLS) >= 3
        and missing_rate[missing_sorted[1]] >= args.structural_min_missing
        and missing_rate[missing_sorted[2]] < args.structural_min_missing
    )
    anchored: dict[str, str] = {}
    if structural_ok:
        for aspect, col in zip(STRUCTURAL_ASPECTS_BY_MISSING, missing_sorted[:2]):
            anchored[col] = aspect
        print(f"[INFO] Structural anchor via missing-rate: "
              + ", ".join(f"{c}->{a}" for c, a in anchored.items()))
    else:
        print("[WARN] Missing-rate structure unclear; falling back to full model assignment")

    # Step 2: optimal (brute-force) assignment of the remaining columns to the
    # remaining aspects, maximizing summed lift. 立地 has near-zero model signal
    # (the E1 model rarely predicts 立地_ネガティブ), so it naturally falls to the
    # left-over column by elimination.
    remaining_cols = [c for c in RATING_COLS if c not in anchored]
    remaining_aspects = [a for a in aspect_names if a not in anchored.values()]
    assigned_col: dict[str, str] = dict(anchored)
    best_perm, best_score = None, float("-inf")
    for perm in itertools.permutations(remaining_aspects, len(remaining_cols)):
        score = sum(lift[c][a] for c, a in zip(remaining_cols, perm))
        if score > best_score:
            best_score, best_perm = score, perm
    if best_perm:
        for c, a in zip(remaining_cols, best_perm):
            assigned_col[c] = a

    # Step 3: confidence + provenance per column.
    mapping: dict[str, dict] = {}
    n_high = 0
    for col in RATING_COLS:
        aspect = assigned_col[col]
        col_lift = lift[col]
        others = sorted((col_lift[a] for a in aspect_names if a != aspect), reverse=True)
        runner_up = others[0] if others else 0.0
        margin = col_lift[aspect] - runner_up
        top_hit = hit_rate[col][aspect]
        col_argmax = max(aspect_names, key=lambda a: col_lift[a])

        if col in anchored:
            method = "structural_missing_rate"
            # High when the model evidence is at least consistent (this aspect is
            # also the column's strongest lift); otherwise medium.
            confidence = "high" if col_argmax == aspect else "medium"
            note = f"missing_rate={missing_rate[col]:.3f}; model_argmax={col_argmax}"
        elif baseline[aspect] < args.low_base_rate:
            method = "elimination"
            confidence = "low"
            note = (f"model rarely predicts {aspect}_ネガティブ (base_hit={baseline[aspect]:.3f}); "
                    "assigned by elimination, not model-confirmed")
        else:
            method = "model_negative_hit_lift"
            confidence = "high" if (margin >= args.min_lift and top_hit >= 0.15) else "low"
            note = ""
        if confidence == "high":
            n_high += 1

        mapping[col] = {
            "aspect": aspect,
            "confidence": confidence,
            "method": method,
            "neg_hit_rate": round(top_hit, 4),
            "lift": round(col_lift[aspect], 4),
            "margin_over_runner_up": round(margin, 4),
            "missing_rate": round(missing_rate[col], 4),
            # Low rating but model fires NO negative for the mapped aspect.
            "rating_text_inconsistency_rate": round(1.0 - top_hit, 4),
            "n_samples": n_used[col],
            "note": note,
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel_csv": str(panel_csv),
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_per_col_requested": args.n_per_col,
        "seed": args.seed,
        "total_reviews_scanned": total_rows,
        "structural_anchor_applied": structural_ok,
        "missing_rate": {c: round(missing_rate[c], 4) for c in RATING_COLS},
        "aspect_baseline_neg_hit_rate": {a: round(baseline[a], 4) for a in aspect_names},
        "hit_rate_matrix": {c: {a: round(hit_rate[c][a], 4) for a in aspect_names} for c in RATING_COLS},
        "lift_matrix": {c: {a: round(lift[c][a], 4) for a in aspect_names} for c in RATING_COLS},
        "n_high_confidence": n_high,
        "acceptance_high_ge_5_of_6": n_high >= 5,
        "mapping": mapping,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[RESULT] column -> aspect mapping:")
    for col in RATING_COLS:
        m = mapping[col]
        print(f"  {col} -> {m['aspect']:<8} conf={m['confidence']:<6} via={m['method']:<24} "
              f"hit={m['neg_hit_rate']:.3f} lift={m['lift']:+.3f} "
              f"miss={m['missing_rate']:.3f} inconsist={m['rating_text_inconsistency_rate']:.3f}")
    print(f"[RESULT] high-confidence columns: {n_high}/6 "
          f"(acceptance >=5/6: {'PASS' if n_high >= 5 else 'REVIEW'})")
    print(f"[INFO] Wrote mapping -> {out_path}")


if __name__ == "__main__":
    main()
