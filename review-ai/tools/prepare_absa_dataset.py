# -*- coding: utf-8 -*-
"""
Prepare the hotel aspect sentiment dataset for ABSA baseline training.

Input:
    gold/data/travel_aspect_sentiment.tsv

Outputs:
    data/absa_train.csv
    data/absa_valid.csv
    data/absa_test.csv
    data/absa_label_columns.json
    data/absa_split_summary.json

The source TSV has a known layout quirk: the header has an extra leading empty
column name, while data rows are 17 columns:
    レビューID, 文ID, レビュー文, + 14 aspect-sentiment label columns

Some review texts also contain literal tabs, so parsing must split labels from
the right side instead of relying on pandas.read_csv directly.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("gold/data/travel_aspect_sentiment.tsv")
DEFAULT_OUTPUT_DIR = Path("data")


def parse_aspect_tsv(path: Path) -> tuple[pd.DataFrame, list[str], int]:
    """Parse the source TSV robustly and return (dataframe, label columns, extra-tab rows)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty input file: {path}")

    header = lines[0].split("\t")
    if len(header) < 5:
        raise ValueError(f"Unexpected header in {path}: {header}")

    # Header columns: "", レビューID, 文ID, レビュー文, <14 label columns>
    label_cols = header[4:]
    n_labels = len(label_cols)
    min_fields = 2 + 1 + n_labels  # review_id, sentence_id, text, labels

    rows: list[list[str]] = []
    rows_with_extra_tab = 0
    for line_no, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) < min_fields:
            fields = fields + [""] * (min_fields - len(fields))
        if len(fields) > min_fields:
            rows_with_extra_tab += 1

        review_id = fields[0]
        sentence_id = fields[1]
        labels = fields[-n_labels:]
        text = "\t".join(fields[2:-n_labels])
        rows.append([review_id, sentence_id, text] + labels)

    df = pd.DataFrame(rows, columns=["review_id", "sentence_id", "text"] + label_cols)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # Stable row identifier for traceability.
    df.insert(0, "text_id", df["review_id"].astype(str) + "_" + df["sentence_id"].astype(str))
    df["num_labels"] = df[label_cols].sum(axis=1).astype(int)
    return df.reset_index(drop=True), label_cols, rows_with_extra_tab


def split_by_review_id(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Group split by review_id to avoid sentence leakage across splits."""
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Require train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1")

    review_ids = sorted(df["review_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(review_ids)

    n = len(review_ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train_ids = set(review_ids[:n_train])
    valid_ids = set(review_ids[n_train:n_train + n_valid])
    test_ids = set(review_ids[n_train + n_valid:])

    splits = {
        "train": df[df["review_id"].astype(str).isin(train_ids)].copy(),
        "valid": df[df["review_id"].astype(str).isin(valid_ids)].copy(),
        "test": df[df["review_id"].astype(str).isin(test_ids)].copy(),
    }
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


def summarize_split(df: pd.DataFrame, label_cols: list[str]) -> dict:
    label_counts = {col: int(df[col].sum()) for col in label_cols}
    rows = int(len(df))
    annotated = int((df[label_cols].sum(axis=1) > 0).sum())
    return {
        "rows": rows,
        "unique_reviews": int(df["review_id"].nunique()),
        "annotated_rows": annotated,
        "unlabeled_rows": rows - annotated,
        "multi_label_rows": int((df[label_cols].sum(axis=1) > 1).sum()),
        "total_label_marks": int(df[label_cols].sum().sum()),
        "label_counts": label_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ABSA train/valid/test CSV files")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Source travel_aspect_sentiment.tsv")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--drop_unlabeled",
        action="store_true",
        help="Drop rows with no aspect labels. Default keeps them as all-zero background examples.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ABSA dataset preparation")
    print("=" * 60)
    print(f"[INFO] Input: {input_path}")

    df, label_cols, rows_with_extra_tab = parse_aspect_tsv(input_path)
    if args.drop_unlabeled:
        before = len(df)
        df = df[df[label_cols].sum(axis=1) > 0].reset_index(drop=True)
        print(f"[INFO] Dropped unlabeled rows: {before - len(df)}")

    splits = split_by_review_id(df, args.train_ratio, args.valid_ratio, args.seed)

    for split_name, split_df in splits.items():
        out_file = output_dir / f"absa_{split_name}.csv"
        split_df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"[OK] {split_name}: {len(split_df)} rows -> {out_file}")

    label_payload = {
        "label_columns": label_cols,
        "label_to_id": {label: i for i, label in enumerate(label_cols)},
        "id_to_label": {i: label for i, label in enumerate(label_cols)},
    }
    (output_dir / "absa_label_columns.json").write_text(
        json.dumps(label_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "source_file": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": round(1 - args.train_ratio - args.valid_ratio, 6),
        "drop_unlabeled": bool(args.drop_unlabeled),
        "rows_with_extra_tab_in_text": rows_with_extra_tab,
        "label_columns": label_cols,
        "overall": summarize_split(df, label_cols),
        "splits": {name: summarize_split(part, label_cols) for name, part in splits.items()},
    }
    summary_file = output_dir / "absa_split_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Label columns: {output_dir / 'absa_label_columns.json'}")
    print(f"[OK] Summary: {summary_file}")


if __name__ == "__main__":
    main()
