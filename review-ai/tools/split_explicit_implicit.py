# -*- coding: utf-8 -*-
"""Tag each active (sentence, aspect, polarity) label as explicit or implicit via oseti.

Input:
    data/absa_{train,valid,test}.csv

Output:
    data/absa_explicit_implicit_tags.csv   (long table)
    results/task1_explicit_implicit_report.json  (with --report)

Examples:
    python tools/split_explicit_implicit.py --report
    python tools/split_explicit_implicit.py --report --max_rows 200   # smoke test
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

PROJECT_ROOT = TOOLS_DIR.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Split ABSA labels into explicit vs implicit using oseti")
    ap.add_argument("--data_dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--labels_file", default=str(PROJECT_ROOT / "data/absa_label_columns.json"))
    ap.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"),
        help="Long-format output CSV",
    )
    ap.add_argument(
        "--report_path",
        default=str(PROJECT_ROOT / "results/task1_explicit_implicit_report.json"),
        help="JSON stats output",
    )
    ap.add_argument("--report", action="store_true", help="Write report JSON and print summary tables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug cap per split (0 = all)")
    ap.add_argument(
        "--cache_path",
        default=str(PROJECT_ROOT / "data/oseti_sentence_cache.json"),
        help="Cache oseti results keyed by text_id",
    )
    return ap.parse_args()


def load_label_columns(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["label_columns"])


def parse_label_name(label_col: str) -> tuple[str, str]:
    if "_ポジティブ" in label_col:
        return label_col.replace("_ポジティブ", ""), "ポジティブ"
    if "_ネガティブ" in label_col:
        return label_col.replace("_ネガティブ", ""), "ネガティブ"
    raise ValueError(f"Unexpected label column: {label_col}")


def analyze_oseti(text: str, analyzer) -> dict:
    empty = {
        "positive_hits": [],
        "negative_hits": [],
        "has_positive": False,
        "has_negative": False,
        "score_sum": 0.0,
        "error": None,
    }
    if not text or not str(text).strip():
        return empty
    try:
        details = analyzer.analyze_detail(text)
    except Exception as exc:  # oseti/mecab can fail on odd punctuation or symbols
        out = empty.copy()
        out["error"] = str(exc)
        return out
    pos_hits: list[str] = []
    neg_hits: list[str] = []
    for chunk in details:
        pos_hits.extend(chunk.get("positive") or [])
        neg_hits.extend(chunk.get("negative") or [])
    return {
        "positive_hits": sorted(set(pos_hits)),
        "negative_hits": sorted(set(neg_hits)),
        "has_positive": bool(pos_hits),
        "has_negative": bool(neg_hits),
        "score_sum": float(sum(chunk.get("score", 0.0) for chunk in details)),
        "error": None,
    }


def judge_source(polarity: str, oseti_info: dict) -> str:
    if polarity == "ポジティブ":
        return "explicit" if oseti_info["has_positive"] else "implicit"
    if polarity == "ネガティブ":
        return "explicit" if oseti_info["has_negative"] else "implicit"
    raise ValueError(f"Unknown polarity: {polarity}")


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def process_split(
    split: str,
    df: pd.DataFrame,
    label_cols: list[str],
    analyzer,
    cache: dict,
    max_rows: int,
) -> list[dict]:
    rows_out: list[dict] = []
    subset = df if max_rows <= 0 else df.head(max_rows)

    for idx, row in subset.iterrows():
        text_id = str(row["text_id"])
        text = str(row["text"])
        if text_id not in cache:
            cache[text_id] = analyze_oseti(text, analyzer)
            if (idx + 1) % 500 == 0:
                print(f"       [{split}] analyzed {idx + 1:,}/{len(subset):,} sentences")
        oseti_info = cache[text_id]

        for label_col in label_cols:
            label_value = int(row[label_col])
            if label_value != 1:
                continue
            aspect, polarity = parse_label_name(label_col)
            source = judge_source(polarity, oseti_info)
            rows_out.append(
                {
                    "text_id": text_id,
                    "review_id": str(row["review_id"]),
                    "sentence_id": str(row["sentence_id"]),
                    "split": split,
                    "text": text,
                    "aspect": aspect,
                    "polarity": polarity,
                    "label_column": label_col,
                    "label": 1,
                    "source": source,
                    "oseti_positive_hits": "|".join(oseti_info["positive_hits"]),
                    "oseti_negative_hits": "|".join(oseti_info["negative_hits"]),
                    "oseti_score_sum": oseti_info["score_sum"],
                }
            )
    return rows_out


def build_report(long_df: pd.DataFrame) -> dict:
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_active_labels": int(len(long_df)),
        "by_split": {},
        "by_aspect": {},
        "by_polarity": {},
        "by_aspect_polarity": {},
    }

    for split, part in long_df.groupby("split"):
        total = len(part)
        explicit = int((part["source"] == "explicit").sum())
        implicit = int((part["source"] == "implicit").sum())
        report["by_split"][split] = {
            "total_active_labels": total,
            "explicit": explicit,
            "implicit": implicit,
            "explicit_pct": round(explicit / total * 100, 2) if total else 0.0,
            "implicit_pct": round(implicit / total * 100, 2) if total else 0.0,
        }

    aspect_stats = {}
    for aspect, part in long_df.groupby("aspect"):
        total = len(part)
        implicit = int((part["source"] == "implicit").sum())
        aspect_stats[aspect] = {
            "total": total,
            "implicit": implicit,
            "implicit_pct": round(implicit / total * 100, 2) if total else 0.0,
        }
    report["by_aspect"] = dict(sorted(aspect_stats.items(), key=lambda x: -x[1]["implicit_pct"]))

    for polarity, part in long_df.groupby("polarity"):
        total = len(part)
        implicit = int((part["source"] == "implicit").sum())
        report["by_polarity"][polarity] = {
            "total": total,
            "implicit": implicit,
            "implicit_pct": round(implicit / total * 100, 2) if total else 0.0,
        }

    asp_pol = {}
    for (aspect, polarity), part in long_df.groupby(["aspect", "polarity"]):
        total = len(part)
        implicit = int((part["source"] == "implicit").sum())
        asp_pol[f"{aspect}_{polarity}"] = {
            "total": total,
            "implicit": implicit,
            "implicit_pct": round(implicit / total * 100, 2) if total else 0.0,
        }
    report["by_aspect_polarity"] = dict(
        sorted(asp_pol.items(), key=lambda x: -x[1]["implicit_pct"])
    )

    neg = long_df[long_df["polarity"] == "ネガティブ"]
    pos = long_df[long_df["polarity"] == "ポジティブ"]
    report["hypothesis_check"] = {
        "negative_implicit_pct": round((neg["source"] == "implicit").mean() * 100, 2) if len(neg) else 0.0,
        "positive_implicit_pct": round((pos["source"] == "implicit").mean() * 100, 2) if len(pos) else 0.0,
        "negative_implicit_higher_than_positive": bool(
            len(neg) and len(pos) and (neg["source"] == "implicit").mean() > (pos["source"] == "implicit").mean()
        ),
    }
    return report


def print_report_tables(report: dict) -> None:
    print("\n=== Task 1: explicit / implicit summary ===")
    print("\n[By split]")
    for split, stats in report["by_split"].items():
        print(
            f"  {split:5s}  total={stats['total_active_labels']:6,}  "
            f"explicit={stats['explicit_pct']:5.1f}%  implicit={stats['implicit_pct']:5.1f}%"
        )

    print("\n[By polarity]")
    for polarity, stats in report["by_polarity"].items():
        print(
            f"  {polarity:8s}  total={stats['total']:6,}  implicit={stats['implicit_pct']:5.1f}%"
        )

    print("\n[Top aspects by implicit %]")
    for aspect, stats in list(report["by_aspect"].items())[:7]:
        print(f"  {aspect:12s}  implicit={stats['implicit_pct']:5.1f}%  (n={stats['total']:,})")

    hyp = report["hypothesis_check"]
    print("\n[Hypothesis: negative implicit % > positive implicit %]")
    print(
        f"  negative implicit: {hyp['negative_implicit_pct']:.1f}%  |  "
        f"positive implicit: {hyp['positive_implicit_pct']:.1f}%  |  "
        f"pass={hyp['negative_implicit_higher_than_positive']}"
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    label_cols = load_label_columns(Path(args.labels_file))
    cache = load_cache(Path(args.cache_path))

    import oseti

    analyzer = oseti.Analyzer()
    all_rows: list[dict] = []

    for split in ("train", "valid", "test"):
        path = data_dir / f"absa_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[INFO] Processing {path.name} ...")
        df = pd.read_csv(path)
        split_rows = process_split(split, df, label_cols, analyzer, cache, args.max_rows)
        all_rows.extend(split_rows)
        print(f"       active labels: {len(split_rows):,}")

    save_cache(Path(args.cache_path), cache)

    long_df = pd.DataFrame(all_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote long table -> {output_path} ({len(long_df):,} rows)")

    report = build_report(long_df)
    if args.report:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote report -> {report_path}")
        print_report_tables(report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
