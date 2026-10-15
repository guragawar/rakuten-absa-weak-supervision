# -*- coding: utf-8 -*-
"""Check text overlap between travel review panel (2009–2024) and Gold ABSA data.

Outputs:
    data/panel_overlap_with_gold.json          — summary stats
    data/panel_exclude_text_hashes.jsonl       — panel rows to exclude from weak supervision

Examples:
    python tools/check_panel_gold_overlap.py
    python tools/check_panel_gold_overlap.py --minhash --minhash_threshold 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_absa_dataset import parse_aspect_tsv
from travel_review_panel_utils import (
    PanelRow,
    char_shingles,
    discover_panel_files,
    iter_panel_rows,
    minhash_signature,
    minhash_similarity,
    normalize_review_text,
    project_root,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect overlap between panel reviews and Gold ABSA corpus")
    ap.add_argument(
        "--raw_dir",
        default=str(project_root() / "gold/data/raw"),
        help="Directory with 01_Travel_Review.{YYYY}.tsv",
    )
    ap.add_argument(
        "--gold_tsv",
        default=str(project_root() / "gold/data/travel_aspect_sentiment.tsv"),
        help="Gold ABSA source TSV",
    )
    ap.add_argument(
        "--absa_dir",
        default=str(project_root() / "data"),
        help="Directory with absa_{train,valid,test}.csv for split-aware stats",
    )
    ap.add_argument(
        "--output",
        default=str(project_root() / "data/panel_overlap_with_gold.json"),
        help="Summary JSON output path",
    )
    ap.add_argument(
        "--exclude_hashes",
        default=str(project_root() / "data/panel_exclude_text_hashes.jsonl"),
        help="JSONL of overlapping panel text hashes to exclude from weak supervision",
    )
    ap.add_argument(
        "--include_containment",
        action="store_true",
        help="Also check whether a Gold full review is a substring of a panel review (slower)",
    )
    ap.add_argument(
        "--containment_min_len",
        type=int,
        default=30,
        help="Minimum normalized Gold review length for containment checks",
    )
    ap.add_argument(
        "--minhash",
        action="store_true",
        help="Also run MinHash near-duplicate detection (slower)",
    )
    ap.add_argument(
        "--minhash_threshold",
        type=float,
        default=0.85,
        help="MinHash similarity threshold (only with --minhash)",
    )
    ap.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Number of overlap examples to include in the JSON report",
    )
    return ap.parse_args()


def load_gold_reviews(gold_tsv: Path) -> tuple[pd.DataFrame, dict[str, str], dict]:
    df, _labels, extra_tab_rows = parse_aspect_tsv(gold_tsv)

    ordered = df.sort_values(["review_id", "sentence_id"])
    full_reviews: dict[str, str] = {}
    for review_id, group in ordered.groupby("review_id", sort=False):
        full_reviews[str(review_id)] = "".join(group["text"].astype(str).tolist())

    meta = {"rows_with_extra_tab": extra_tab_rows}
    return df, full_reviews, meta


def load_review_split_map(absa_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        path = absa_dir / f"absa_{split}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path, usecols=["review_id"])
        for review_id in part["review_id"].astype(str).unique():
            split_map[review_id] = split
    return split_map


def build_gold_indexes(
    full_reviews: dict[str, str],
    *,
    containment_min_len: int,
    enable_minhash: bool,
) -> dict:
    full_norm_to_review_ids: dict[str, set[str]] = defaultdict(set)
    review_id_to_norm: dict[str, str] = {}
    gold_norm_items: list[tuple[str, str]] = []
    gold_minhash: list[tuple[str, list[int], int]] = []

    for review_id, text in full_reviews.items():
        norm = normalize_review_text(text)
        if not norm:
            continue
        review_id_to_norm[review_id] = norm
        full_norm_to_review_ids[norm].add(review_id)
        if len(norm) >= containment_min_len:
            gold_norm_items.append((review_id, norm))
        if enable_minhash:
            gold_minhash.append(
                (review_id, minhash_signature(char_shingles(norm)), len(norm))
            )

    return {
        "full_norm_to_review_ids": full_norm_to_review_ids,
        "review_id_to_norm": review_id_to_norm,
        "gold_norm_items": gold_norm_items,
        "gold_minhash": gold_minhash,
    }


def find_containment_matches(
    panel_norm: str,
    gold_norm_items: list[tuple[str, str]],
) -> set[str]:
    if not panel_norm:
        return set()
    panel_len = len(panel_norm)
    matched: set[str] = set()
    for review_id, gold_norm in gold_norm_items:
        gold_len = len(gold_norm)
        if gold_len > panel_len:
            continue
        if gold_norm in panel_norm:
            matched.add(review_id)
    return matched


def find_minhash_match(
    panel_norm: str,
    gold_minhash: list[tuple[str, list[int], int]],
    threshold: float,
) -> tuple[str, float] | None:
    if not panel_norm:
        return None
    panel_sig = minhash_signature(char_shingles(panel_norm))
    panel_len = len(panel_norm)
    best_review_id = ""
    best_score = 0.0

    for review_id, gold_sig, gold_len in gold_minhash:
        # Cheap length filter before full signature comparison.
        if gold_len == 0:
            continue
        ratio = panel_len / gold_len
        if ratio < 0.8 or ratio > 1.25:
            continue
        score = minhash_similarity(panel_sig, gold_sig)
        if score > best_score:
            best_score = score
            best_review_id = review_id

    if best_score >= threshold:
        return best_review_id, best_score
    return None


def record_overlap(
    *,
    row: PanelRow,
    year: int,
    panel_norm: str,
    matched_review_ids: set[str],
    match_types: list[str],
    split_map: dict[str, str],
    exclude_records: dict[str, dict],
    examples: list[dict],
    max_examples: int,
    minhash_score: float = 0.0,
) -> None:
    if not matched_review_ids:
        return
    exclude_records[row.text_hash] = {
        "text_hash": row.text_hash,
        "source_year": year,
        "hotel_id": row.hotel_id,
        "posted_at": row.posted_at,
        "match_types": match_types,
        "gold_review_ids": sorted(matched_review_ids),
        "gold_splits": sorted({split_map.get(rid, "unknown") for rid in matched_review_ids}),
        "minhash_score": round(minhash_score, 4) if minhash_score else None,
    }
    if len(examples) < max_examples:
        examples.append(
            {
                "source_year": year,
                "hotel_id": row.hotel_id,
                "match_types": match_types,
                "gold_review_ids": sorted(matched_review_ids),
                "gold_splits": sorted({split_map.get(rid, "unknown") for rid in matched_review_ids}),
                "panel_excerpt": row.review_text[:120],
                "minhash_score": round(minhash_score, 4) if minhash_score else None,
            }
        )


def update_split_counters(
    matched_review_ids: set[str],
    match_types: list[str],
    split_map: dict[str, str],
    exact_by_split: Counter,
    containment_by_split: Counter,
    minhash_by_split: Counter,
) -> None:
    primary = match_types[0]
    for review_id in matched_review_ids:
        split = split_map.get(review_id, "unknown")
        if primary == "exact_full":
            exact_by_split[split] += 1
        elif primary == "gold_full_containment":
            containment_by_split[split] += 1
        elif primary == "minhash":
            minhash_by_split[split] += 1


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    gold_tsv = Path(args.gold_tsv)
    absa_dir = Path(args.absa_dir)
    output_path = Path(args.output)
    exclude_path = Path(args.exclude_hashes)

    discovered = discover_panel_files(raw_dir)
    if not discovered:
        raise SystemExit(f"No panel files found in {raw_dir}")

    print("[INFO] Loading Gold ABSA corpus...")
    gold_df, full_reviews, gold_meta = load_gold_reviews(gold_tsv)
    split_map = load_review_split_map(absa_dir)
    indexes = build_gold_indexes(
        full_reviews,
        containment_min_len=args.containment_min_len,
        enable_minhash=args.minhash,
    )

    full_norm_to_review_ids = indexes["full_norm_to_review_ids"]
    gold_norm_items = indexes["gold_norm_items"]
    gold_minhash = indexes["gold_minhash"]

    panel_total = 0
    panel_nonempty = 0
    panel_by_year = Counter()

    exact_full_hits = 0
    containment_hits = 0
    minhash_hits = 0

    exact_by_year = Counter()
    containment_by_year = Counter()
    minhash_by_year = Counter()
    exact_by_split = Counter()
    containment_by_split = Counter()
    minhash_by_split = Counter()

    exclude_records: dict[str, dict] = {}
    examples: list[dict] = []
    unmatched_for_containment: list[tuple[PanelRow, int, str]] = []

    print("[INFO] Scanning panel files (pass 1: exact match)...")
    for year, path in discovered:
        print(f"       {path.name}")
        for row in iter_panel_rows(path, source_year=year):
            panel_total += 1
            panel_by_year[year] += 1
            panel_norm = row.review_text_norm
            if not panel_norm:
                continue
            panel_nonempty += 1

            matched_review_ids: set[str] = set()
            match_types: list[str] = []

            if panel_norm in full_norm_to_review_ids:
                exact_full_hits += 1
                exact_by_year[year] += 1
                matched_review_ids.update(full_norm_to_review_ids[panel_norm])
                match_types.append("exact_full")
                update_split_counters(
                    matched_review_ids,
                    match_types,
                    split_map,
                    exact_by_split,
                    containment_by_split,
                    minhash_by_split,
                )
                record_overlap(
                    row=row,
                    year=year,
                    panel_norm=panel_norm,
                    matched_review_ids=matched_review_ids,
                    match_types=match_types,
                    split_map=split_map,
                    exclude_records=exclude_records,
                    examples=examples,
                    max_examples=args.max_examples,
                )
                continue

            if args.include_containment:
                unmatched_for_containment.append((row, year, panel_norm))
                continue

            minhash_score = 0.0
            if args.minhash:
                found = find_minhash_match(panel_norm, gold_minhash, args.minhash_threshold)
                if found is not None:
                    review_id, minhash_score = found
                    matched_review_ids.add(review_id)
                    minhash_hits += 1
                    minhash_by_year[year] += 1
                    match_types.append("minhash")
                    update_split_counters(
                        matched_review_ids,
                        match_types,
                        split_map,
                        exact_by_split,
                        containment_by_split,
                        minhash_by_split,
                    )
                    record_overlap(
                        row=row,
                        year=year,
                        panel_norm=panel_norm,
                        matched_review_ids=matched_review_ids,
                        match_types=match_types,
                        split_map=split_map,
                        exclude_records=exclude_records,
                        examples=examples,
                        max_examples=args.max_examples,
                        minhash_score=minhash_score,
                    )

    if args.include_containment and unmatched_for_containment:
        print(
            f"[INFO] Pass 2: containment check on {len(unmatched_for_containment):,} "
            f"unmatched panel rows..."
        )
        for idx, (row, year, panel_norm) in enumerate(unmatched_for_containment, start=1):
            if idx % 50000 == 0:
                print(f"       processed {idx:,}/{len(unmatched_for_containment):,}")
            matched_review_ids = find_containment_matches(panel_norm, gold_norm_items)
            if not matched_review_ids:
                if args.minhash:
                    found = find_minhash_match(panel_norm, gold_minhash, args.minhash_threshold)
                    if found is not None:
                        review_id, minhash_score = found
                        matched_review_ids = {review_id}
                        minhash_hits += 1
                        minhash_by_year[year] += 1
                        match_types = ["minhash"]
                        update_split_counters(
                            matched_review_ids,
                            match_types,
                            split_map,
                            exact_by_split,
                            containment_by_split,
                            minhash_by_split,
                        )
                        record_overlap(
                            row=row,
                            year=year,
                            panel_norm=panel_norm,
                            matched_review_ids=matched_review_ids,
                            match_types=match_types,
                            split_map=split_map,
                            exclude_records=exclude_records,
                            examples=examples,
                            max_examples=args.max_examples,
                            minhash_score=minhash_score,
                        )
                continue

            containment_hits += 1
            containment_by_year[year] += 1
            match_types = ["gold_full_containment"]
            update_split_counters(
                matched_review_ids,
                match_types,
                split_map,
                exact_by_split,
                containment_by_split,
                minhash_by_split,
            )
            record_overlap(
                row=row,
                year=year,
                panel_norm=panel_norm,
                matched_review_ids=matched_review_ids,
                match_types=match_types,
                split_map=split_map,
                exclude_records=exclude_records,
                examples=examples,
                max_examples=args.max_examples,
            )

    unique_gold_reviews_overlapped = {
        rid
        for record in exclude_records.values()
        for rid in record["gold_review_ids"]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold": {
            "source": str(gold_tsv),
            "sentences": int(len(gold_df)),
            "unique_reviews": int(gold_df["review_id"].nunique()),
            "reconstructed_full_reviews": len(full_reviews),
            "rows_with_extra_tab": gold_meta["rows_with_extra_tab"],
            "absa_split_map_size": len(split_map),
        },
        "panel": {
            "raw_dir": str(raw_dir),
            "files_scanned": [str(path) for _, path in discovered],
            "years_scanned": [year for year, _ in discovered],
            "years_missing_2009_2024": [
                y for y in range(2009, 2025) if y not in {year for year, _ in discovered}
            ],
            "total_rows": panel_total,
            "nonempty_review_rows": panel_nonempty,
            "rows_per_year": dict(sorted(panel_by_year.items())),
        },
        "overlap": {
            "exact_full_review_match": {
                "panel_rows": exact_full_hits,
                "panel_pct": round(exact_full_hits / panel_nonempty * 100, 4) if panel_nonempty else 0.0,
                "by_year": dict(sorted(exact_by_year.items())),
                "by_gold_split": dict(sorted(exact_by_split.items())),
            },
            "gold_full_containment_match": {
                "enabled": args.include_containment,
                "panel_rows": containment_hits,
                "panel_pct": round(containment_hits / panel_nonempty * 100, 4)
                if panel_nonempty and args.include_containment
                else 0.0,
                "containment_min_len": args.containment_min_len if args.include_containment else None,
                "by_year": dict(sorted(containment_by_year.items())),
                "by_gold_split": dict(sorted(containment_by_split.items())),
            },
            "minhash_near_duplicate": {
                "enabled": args.minhash,
                "threshold": args.minhash_threshold if args.minhash else None,
                "panel_rows": minhash_hits,
                "panel_pct": round(minhash_hits / panel_nonempty * 100, 4) if panel_nonempty and args.minhash else 0.0,
                "by_year": dict(sorted(minhash_by_year.items())),
                "by_gold_split": dict(sorted(minhash_by_split.items())),
            },
            "total_panel_rows_to_exclude": len(exclude_records),
            "total_panel_rows_to_exclude_pct": round(len(exclude_records) / panel_nonempty * 100, 4)
            if panel_nonempty
            else 0.0,
            "unique_gold_reviews_matched": len(unique_gold_reviews_overlapped),
            "unique_gold_reviews_matched_pct": round(
                len(unique_gold_reviews_overlapped) / gold_df["review_id"].nunique() * 100, 4
            ),
            "test_split_rows_excluded": sum(
                1
                for record in exclude_records.values()
                if "test" in record.get("gold_splits", [])
            ),
        },
        "outputs": {
            "summary_json": str(output_path),
            "exclude_hashes_jsonl": str(exclude_path),
        },
        "examples": examples,
        "recommendation": (
            "Exclude all rows listed in panel_exclude_text_hashes.jsonl before weak supervision "
            "or distant-supervision mining. Re-run this script after adding 2009–2023 panel files."
        ),
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with exclude_path.open("w", encoding="utf-8") as handle:
        for record in exclude_records.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[INFO] Overlap check complete")
    print(f"       panel rows (non-empty): {panel_nonempty:,}")
    print(f"       exact full matches    : {exact_full_hits:,}")
    if args.include_containment:
        print(f"       gold full containment : {containment_hits:,}")
    if args.minhash:
        print(f"       minhash near-dup      : {minhash_hits:,}")
    print(f"       rows to exclude       : {len(exclude_records):,}")
    print(f"       gold reviews matched  : {len(unique_gold_reviews_overlapped):,}")
    print(f"[INFO] Wrote summary -> {output_path}")
    print(f"[INFO] Wrote exclude list -> {exclude_path}")


if __name__ == "__main__":
    main()
