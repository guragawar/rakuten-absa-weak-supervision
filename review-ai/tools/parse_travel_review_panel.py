# -*- coding: utf-8 -*-
"""Parse and validate 01_Travel_Review.{YYYY}.tsv panel files (Phase 0).

Examples:
    # Validate all available yearly files and write a JSON report
    python tools/parse_travel_review_panel.py --report

    # Export a slim CSV for one year (review text + metadata)
    python tools/parse_travel_review_panel.py --years 2024 --export_dir data/panel_parsed
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

from travel_review_panel_utils import (
    N_FIELDS,
    discover_panel_files,
    iter_panel_rows,
    project_root,
)

EXPORT_COLUMNS = [
    "text_hash",
    "source_year",
    "hotel_id",
    "posted_at",
    "overall_rating",
    "rating_1",
    "rating_2",
    "rating_3",
    "rating_4",
    "rating_5",
    "rating_6",
    "review_text",
    "review_text_norm",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parse Rakuten Travel review panel TSV files")
    ap.add_argument(
        "--raw_dir",
        default=str(project_root() / "gold/data/raw"),
        help="Directory containing 01_Travel_Review.{YYYY}.tsv files",
    )
    ap.add_argument(
        "--years",
        default="",
        help="Comma-separated years to process (default: all discovered files)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Write parse/validation summary JSON",
    )
    ap.add_argument(
        "--report_path",
        default=str(project_root() / "results/travel_review_panel_parse_report.json"),
        help="Output path for --report",
    )
    ap.add_argument(
        "--export_dir",
        default="",
        help="If set, export slim CSV per year into this directory",
    )
    ap.add_argument(
        "--max_rows_per_year",
        type=int,
        default=0,
        help="Debug cap per year (0 = no cap)",
    )
    return ap.parse_args()


def profile_year(path: Path, year: int, max_rows: int = 0) -> dict:
    rows_ok = 0
    rows_malformed = 0
    rows_empty_review = 0
    rows_embedded_tab = 0
    hotels = Counter()
    months = Counter()
    review_lens: list[int] = []

    # errors="replace": the 2016 dump is truncated mid-character at EOF; tolerate it
    # so the ~82k intact rows are still usable (the trailing partial line becomes malformed
    # and is skipped by the field-count check below).
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_rows and rows_ok >= max_rows:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < N_FIELDS:
                rows_malformed += 1
                continue
            if len(fields) > N_FIELDS:
                rows_embedded_tab += 1
                fields = fields[:16] + ["\t".join(fields[16:-1]), fields[-1]]

            rows_ok += 1
            review = fields[16].strip()
            if not review:
                rows_empty_review += 1
            else:
                review_lens.append(len(review))
            hotels[fields[2].strip()] += 1
            posted = fields[1].strip()
            if len(posted) >= 7:
                months[posted[:7]] += 1

    review_lens.sort()
    n = rows_ok

    def quantile(q: float) -> int:
        if not review_lens:
            return 0
        idx = min(int(len(review_lens) * q), len(review_lens) - 1)
        return review_lens[idx]

    return {
        "file": str(path),
        "year": year,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "rows_ok": rows_ok,
        "rows_malformed_short": rows_malformed,
        "rows_embedded_tab": rows_embedded_tab,
        "empty_review": rows_empty_review,
        "unique_hotels": len(hotels),
        "months_covered": len(months),
        "date_range": [min(months), max(months)] if months else [],
        "review_len_median": quantile(0.5),
        "review_len_p90": quantile(0.9),
        "review_len_mean": round(sum(review_lens) / len(review_lens), 1) if review_lens else 0,
    }


def export_year_csv(path: Path, year: int, export_dir: Path, max_rows: int = 0) -> dict:
    out_path = export_dir / f"panel_{year}.csv"
    export_dir.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_skipped_empty = 0

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for row in iter_panel_rows(path, source_year=year):
            if max_rows and rows_written >= max_rows:
                break
            if not row.review_text:
                rows_skipped_empty += 1
                continue
            writer.writerow(
                {
                    "text_hash": row.text_hash,
                    "source_year": row.source_year,
                    "hotel_id": row.hotel_id,
                    "posted_at": row.posted_at,
                    "overall_rating": row.overall_rating,
                    "rating_1": row.aspect_ratings[0],
                    "rating_2": row.aspect_ratings[1],
                    "rating_3": row.aspect_ratings[2],
                    "rating_4": row.aspect_ratings[3],
                    "rating_5": row.aspect_ratings[4],
                    "rating_6": row.aspect_ratings[5],
                    "review_text": row.review_text,
                    "review_text_norm": row.review_text_norm,
                }
            )
            rows_written += 1

    return {
        "year": year,
        "export_path": str(out_path),
        "rows_written": rows_written,
        "rows_skipped_empty": rows_skipped_empty,
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    discovered = discover_panel_files(raw_dir)
    if not discovered:
        raise SystemExit(f"No panel files found in {raw_dir}")

    if args.years.strip():
        wanted = {int(y.strip()) for y in args.years.split(",") if y.strip()}
        discovered = [(year, path) for year, path in discovered if year in wanted]
        if not discovered:
            raise SystemExit(f"No files matched --years {args.years}")

    per_year: list[dict] = []
    exports: list[dict] = []

    for year, path in discovered:
        print(f"[INFO] Parsing {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        stats = profile_year(path, year, max_rows=args.max_rows_per_year)
        per_year.append(stats)
        print(
            f"       rows_ok={stats['rows_ok']:,} malformed={stats['rows_malformed_short']:,} "
            f"embedded_tab={stats['rows_embedded_tab']:,}"
        )

        if args.export_dir:
            export_stats = export_year_csv(
                path,
                year,
                Path(args.export_dir),
                max_rows=args.max_rows_per_year,
            )
            exports.append(export_stats)
            print(f"       exported {export_stats['rows_written']:,} rows -> {export_stats['export_path']}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "years_found": [year for year, _ in discovered],
        "years_missing_2009_2024": [
            y for y in range(2009, 2025) if y not in {year for year, _ in discovered}
        ],
        "total_rows_ok": sum(item["rows_ok"] for item in per_year),
        "per_year": per_year,
        "exports": exports,
    }

    if args.report:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote report -> {report_path}")

    if not args.report and not args.export_dir:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
