# -*- coding: utf-8 -*-
"""Format the panel training pool into weak-label JSONL for E7 pretraining (Task 5).

Reads ``data/panel_train_pool.csv`` (with its ``split`` column) plus the
authoritative ``panel_rating_column_map.json`` (from the official Rakuten data
spec), and emits one JSONL row per review with six aspect weak labels + masks:

    rating == 0 (not evaluated)  -> mask = 0  (excluded from that head's loss)
    rating in {1, 2}             -> label = 1 (negative)
    rating in {3, 4, 5}          -> label = 0 (non-negative)

IMPORTANT (spec correction, 2026-07-16): the six ABSA aspects are
立地/部屋/食事/風呂/サービス/設備 and map to CSV columns
overall_rating/rating_1../rating_5 respectively. The column ``rating_6`` is the
OVERALL rating (総合), NOT 立地; it is emitted separately (label_overall) for the
A4 ablation (7-head incl. overall vs 6-head aspects-only), not as an aspect head.

Output per line: text, text_hash, hotel_id, label_1..6, mask_1..6,
label_overall, mask_overall. Head i corresponds to weak_supervision_aspects[i-1].

Example:
    python tools/prepare_panel_weak_labels.py \
        --pool_csv data/panel_train_pool.csv \
        --rating_map data/panel_rating_column_map.json \
        --out_dir data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from travel_review_panel_utils import project_root

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Format panel weak labels for E7")
    ap.add_argument(
        "--pool_csv",
        default=str(project_root() / "data/panel_train_pool.csv"),
        help="Train pool CSV with a split column (from build_panel_train_pool.py)",
    )
    ap.add_argument(
        "--rating_map",
        default=str(project_root() / "data/panel_rating_column_map.json"),
        help="Authoritative aspect->column mapping (official Rakuten data spec)",
    )
    ap.add_argument(
        "--out_dir",
        default=str(project_root() / "data"),
        help="Directory for panel_weak_{train,valid}.jsonl + stats",
    )
    ap.add_argument(
        "--stats_path",
        default=str(project_root() / "data/panel_weak_label_stats.json"),
    )
    return ap.parse_args()


def load_mapping(path: Path) -> tuple[list[str], dict[str, str], str]:
    """Return (aspect_order, aspect->column, overall_column) from the map JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    aspects = payload.get("weak_supervision_aspects")
    aspect_columns = payload.get("aspect_columns")
    overall_column = payload.get("overall_column")
    if not aspects or not aspect_columns:
        raise SystemExit(
            f"{path} is not the authoritative spec-based map "
            f"(missing weak_supervision_aspects / aspect_columns). "
            f"Regenerate it from docs/data_spec_rakuten_travel.md."
        )
    return aspects, aspect_columns, overall_column


def weak_label_and_mask(raw: str) -> tuple[int, int]:
    """Return (label, mask) for one rating cell."""
    v = raw.strip()
    if v in ("", "0"):
        return 0, 0  # masked out; label value is irrelevant
    if v in ("1", "2"):
        return 1, 1  # negative
    if v in ("3", "4", "5"):
        return 0, 1  # non-negative
    return 0, 0  # unexpected value -> treat as missing/masked


def main() -> None:
    args = parse_args()
    pool_csv = Path(args.pool_csv)
    if not pool_csv.exists():
        raise SystemExit(f"Pool CSV not found: {pool_csv}")

    aspects, aspect_columns, overall_column = load_mapping(Path(args.rating_map))
    # head index (1..N) -> (aspect_name, csv_column)
    heads = list(enumerate(aspects, start=1))
    head_cols = {i: aspect_columns[a] for i, a in heads}
    print("[INFO] aspect heads (spec-based):")
    for i, a in heads:
        print(f"       head {i}: {a:<6} <- column {head_cols[i]}")
    print(f"[INFO] overall (総合, A4 only) <- column {overall_column}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "panel_weak_train.jsonl",
        "valid": out_dir / "panel_weak_valid.jsonl",
    }
    handles = {split: path.open("w", encoding="utf-8") for split, path in paths.items()}

    def new_head_counter() -> dict:
        d = {i: Counter() for i, _ in heads}
        d["overall"] = Counter()
        return d

    stats = {
        "train": {"rows": 0, "heads": new_head_counter()},
        "valid": {"rows": 0, "heads": new_head_counter()},
    }
    rows_skipped_unknown_split = 0

    def tally(hc: Counter, label: int, mask: int) -> None:
        if mask == 1:
            hc["valid"] += 1
            hc["pos" if label == 1 else "neg"] += 1
        else:
            hc["masked"] += 1

    try:
        with pool_csv.open(encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                split = row.get("split", "").strip()
                if split not in handles:
                    rows_skipped_unknown_split += 1
                    continue

                record = {
                    "text": row.get("review_text", ""),
                    "text_hash": row.get("text_hash", ""),
                    "hotel_id": row.get("hotel_id", ""),
                    "source_year": row.get("source_year", ""),
                }
                for i, _aspect in heads:
                    label, mask = weak_label_and_mask(row.get(head_cols[i], ""))
                    record[f"label_{i}"] = label
                    record[f"mask_{i}"] = mask
                    tally(stats[split]["heads"][i], label, mask)
                # overall (総合) -- separate, for the A4 ablation only
                o_label, o_mask = weak_label_and_mask(row.get(overall_column, ""))
                record["label_overall"] = o_label
                record["mask_overall"] = o_mask
                tally(stats[split]["heads"]["overall"], o_label, o_mask)

                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                stats[split]["rows"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    def head_report(hc: Counter) -> dict:
        valid, pos, neg = hc["valid"], hc["pos"], hc["neg"]
        return {
            "valid": valid,
            "masked": hc["masked"],
            "pos_neg1_2": pos,
            "neg_nonneg3_5": neg,
            "pos_ratio": round(pos / valid, 4) if valid else 0.0,
        }

    head_key_to_name = {i: a for i, a in heads}
    head_key_to_name["overall"] = "総合(overall)"
    head_key_to_col = dict(head_cols)
    head_key_to_col["overall"] = overall_column

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pool_csv": str(pool_csv),
        "rating_map": str(args.rating_map),
        "head_to_aspect": {str(i): a for i, a in heads},
        "head_to_column": {str(i): head_cols[i] for i, _ in heads},
        "overall_column": overall_column,
        "rows_skipped_unknown_split": rows_skipped_unknown_split,
        "splits": {},
        "totals_per_head": {},
        "outputs": {s: str(p) for s, p in paths.items()},
    }
    for split in ("train", "valid"):
        report["splits"][split] = {
            "rows": stats[split]["rows"],
            "heads": {
                str(k): {"aspect": head_key_to_name[k], "column": head_key_to_col[k],
                         **head_report(stats[split]["heads"][k])}
                for k in list(head_key_to_name.keys())
            },
        }
    for k in list(head_key_to_name.keys()):
        tot = Counter()
        for split in ("train", "valid"):
            tot.update(stats[split]["heads"][k])
        report["totals_per_head"][str(k)] = {
            "aspect": head_key_to_name[k],
            "column": head_key_to_col[k],
            **head_report(tot),
        }

    Path(args.stats_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[INFO] rows: train={stats['train']['rows']:,} valid={stats['valid']['rows']:,} "
          f"skipped_unknown_split={rows_skipped_unknown_split}")
    print("[INFO] per-head (train+valid totals): valid=有効(mask=1) / pos=負面(1-2) / pos_ratio")
    for k in list(head_key_to_name.keys()):
        r = report["totals_per_head"][str(k)]
        tag = "  [A4のみ]" if k == "overall" else ""
        print(f"       head {str(k):<7} {r['aspect']:<12}({r['column']:<14}): "
              f"valid={r['valid']:,} pos={r['pos_neg1_2']:,} pos_ratio={r['pos_ratio']:.4f}{tag}")
    print(f"[INFO] Wrote {paths['train']}")
    print(f"[INFO] Wrote {paths['valid']}")
    print(f"[INFO] Wrote stats -> {args.stats_path}")


if __name__ == "__main__":
    main()
