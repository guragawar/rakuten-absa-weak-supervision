# -*- coding: utf-8 -*-
"""Build the weak-supervision training pool from the parsed 2024 panel (P0-3).

Reads the full parsed panel CSV, drops reviews that overlap Gold (via the
exclude-hash list) and empty reviews, then assigns a deterministic 90/10
train/valid split (seed=42) recorded in a ``split`` column.

Example:
    python tools/build_panel_train_pool.py \
        --panel_csv data/panel_parsed/panel_2024.csv \
        --exclude_hashes data/panel_exclude_text_hashes.jsonl \
        --out_pool data/panel_train_pool.csv \
        --out_stats data/panel_pool_stats.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from travel_review_panel_utils import project_root

# Some panel reviews are long; the default csv field-size limit can be hit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

RATING_COLS = [f"rating_{i}" for i in range(1, 7)]
POOL_COLUMNS = [
    "text_hash",
    "source_year",
    "hotel_id",
    "posted_at",
    "overall_rating",
    *RATING_COLS,
    "review_text",
    "split",
]

VALID_FRACTION = 0.10
SEED = 42


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build panel weak-supervision train pool")
    ap.add_argument(
        "--panel_csv",
        default=str(project_root() / "data/panel_parsed/panel_2024.csv"),
        help="Parsed panel CSV, or a glob for multi-year (e.g. 'data/panel_parsed/panel_*.csv'). "
             "Duplicate review texts (same text_hash) are dropped, within a year and across years; "
             "see --no_dedup.",
    )
    ap.add_argument(
        "--no_dedup",
        action="store_true",
        help="Keep duplicate review texts. Reproduces the 2024 pool of the main E7 run "
             "(data/panel_train_pool.csv, 583,777 rows), which was built before dedup was added.",
    )
    ap.add_argument(
        "--exclude_hashes",
        default=str(project_root() / "data/panel_exclude_text_hashes.jsonl"),
        help="JSONL of text_hash values to exclude (Gold overlap). Empty = no exclusions.",
    )
    ap.add_argument(
        "--out_pool",
        default=str(project_root() / "data/panel_train_pool.csv"),
        help="Output pool CSV (includes a train/valid split column)",
    )
    ap.add_argument(
        "--out_stats",
        default=str(project_root() / "data/panel_pool_stats.json"),
        help="Output stats JSON",
    )
    ap.add_argument("--valid_fraction", type=float, default=VALID_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def load_exclude_hashes(path: Path) -> set[str]:
    excluded: set[str] = set()
    if not path.exists():
        return excluded
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a bare-hash-per-line format as well.
                excluded.add(line)
                continue
            if isinstance(payload, str):
                excluded.add(payload)
            elif isinstance(payload, dict):
                for key in ("text_hash", "hash"):
                    if key in payload:
                        excluded.add(str(payload[key]))
                        break
    return excluded


def assign_split(text_hash: str, seed: int, valid_fraction: float) -> str:
    """Deterministic per-review split, independent of row order."""
    digest = hashlib.md5(f"{seed}:{text_hash}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "valid" if bucket < valid_fraction else "train"


def rating_is_missing(value: str) -> bool:
    return value.strip() in ("", "0")


def rating_is_negative(value: str) -> bool:
    v = value.strip()
    return v in ("1", "2")


def main() -> None:
    args = parse_args()
    import glob as _glob
    if any(ch in args.panel_csv for ch in "*?["):
        panel_files = [Path(p) for p in sorted(_glob.glob(args.panel_csv))]
    else:
        panel_files = [Path(args.panel_csv)]
    if not panel_files:
        raise SystemExit(f"No panel CSV matched: {args.panel_csv}")
    for p in panel_files:
        if not p.exists():
            raise SystemExit(f"Panel CSV not found: {p}")
    print(f"[INFO] {len(panel_files)} panel file(s): {', '.join(p.name for p in panel_files)}")

    excluded = load_exclude_hashes(Path(args.exclude_hashes))
    print(f"[INFO] Loaded {len(excluded):,} exclusion hashes")

    out_pool = Path(args.out_pool)
    out_pool.parent.mkdir(parents=True, exist_ok=True)

    rows_total = 0
    rows_excluded_overlap = 0
    rows_empty_review = 0
    rows_written = 0
    rows_dup_cross_year = 0  # all dropped duplicates (name kept for older stats files)
    rows_dup_within_file = 0
    rows_dup_across_files = 0
    seen_hashes: set[str] = set()  # dedup: the same review text recurs within and across yearly dumps
    per_file_written: dict[str, int] = {}
    split_counts = Counter()
    # Per rating column: missing (0), negative (1-2), non-negative (3-5) tallies.
    rating_missing = Counter()
    rating_negative = Counter()
    rating_nonneg = Counter()

    with out_pool.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=POOL_COLUMNS)
        writer.writeheader()

        for panel_csv in panel_files:
            file_written = 0
            file_hashes: set[str] = set()
            with panel_csv.open(encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    rows_total += 1
                    text_hash = row["text_hash"].strip()
                    if text_hash in excluded:
                        rows_excluded_overlap += 1
                        continue
                    if text_hash in seen_hashes and not args.no_dedup:
                        rows_dup_cross_year += 1
                        if text_hash in file_hashes:
                            rows_dup_within_file += 1
                        else:
                            rows_dup_across_files += 1
                        continue
                    review_text = row.get("review_text", "")
                    if not review_text.strip():
                        rows_empty_review += 1
                        continue
                    seen_hashes.add(text_hash)
                    file_hashes.add(text_hash)

                    split = assign_split(text_hash, args.seed, args.valid_fraction)
                    split_counts[split] += 1

                    for col in RATING_COLS:
                        val = row.get(col, "")
                        if rating_is_missing(val):
                            rating_missing[col] += 1
                        elif rating_is_negative(val):
                            rating_negative[col] += 1
                        else:
                            rating_nonneg[col] += 1

                    writer.writerow(
                        {
                            "text_hash": text_hash,
                            "source_year": row.get("source_year", ""),
                            "hotel_id": row.get("hotel_id", ""),
                            "posted_at": row.get("posted_at", ""),
                            "overall_rating": row.get("overall_rating", ""),
                            **{col: row.get(col, "") for col in RATING_COLS},
                            "review_text": review_text,
                            "split": split,
                        }
                    )
                    rows_written += 1
                    file_written += 1
            per_file_written[panel_csv.name] = file_written
            print(f"       {panel_csv.name}: +{file_written:,} (running total {rows_written:,})")

    def pct(n: int) -> float:
        return round(100.0 * n / rows_written, 2) if rows_written else 0.0

    rating_distribution = {}
    for col in RATING_COLS:
        rating_distribution[col] = {
            "missing_0": rating_missing[col],
            "missing_0_pct": pct(rating_missing[col]),
            "negative_1_2": rating_negative[col],
            "negative_1_2_pct": pct(rating_negative[col]),
            "nonneg_3_5": rating_nonneg[col],
            "nonneg_3_5_pct": pct(rating_nonneg[col]),
        }

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel_files": [str(p) for p in panel_files],
        "rows_written_per_file": per_file_written,
        "exclude_hashes_file": str(args.exclude_hashes),
        "n_exclusion_hashes": len(excluded),
        "seed": args.seed,
        "valid_fraction": args.valid_fraction,
        "rows_total_in": rows_total,
        "rows_excluded_overlap": rows_excluded_overlap,
        "dedup": not args.no_dedup,
        "rows_dup_cross_year": rows_dup_cross_year,
        "rows_dup_within_file": rows_dup_within_file,
        "rows_dup_across_files": rows_dup_across_files,
        "rows_empty_review": rows_empty_review,
        "rows_written": rows_written,
        "split_counts": {
            "train": split_counts["train"],
            "valid": split_counts["valid"],
            "valid_pct": round(100.0 * split_counts["valid"] / rows_written, 2)
            if rows_written
            else 0.0,
        },
        "rating_distribution": rating_distribution,
        "output_pool": str(out_pool),
    }

    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] rows_in={rows_total:,} excluded_overlap={rows_excluded_overlap:,} "
          f"duplicates={rows_dup_cross_year:,} (within-year {rows_dup_within_file:,}, across-year {rows_dup_across_files:,}) empty={rows_empty_review:,} written={rows_written:,}")
    print(f"[INFO] split: train={split_counts['train']:,} valid={split_counts['valid']:,} "
          f"({stats['split_counts']['valid_pct']}% valid)")
    print("[INFO] per-rating missing(0) / negative(1-2) share:")
    for col in RATING_COLS:
        d = rating_distribution[col]
        print(f"       {col}: missing={d['missing_0_pct']}%  neg(1-2)={d['negative_1_2_pct']}%")
    print(f"[INFO] Wrote pool -> {out_pool}")
    print(f"[INFO] Wrote stats -> {out_stats}")


if __name__ == "__main__":
    main()
