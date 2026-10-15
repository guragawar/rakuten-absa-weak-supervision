# -*- coding: utf-8 -*-
"""Build human verification sample for oseti explicit/implicit judgments (Task 2).

Examples:
    python tools/build_verification_sample.py
    python tools/build_verification_sample.py --target_size 400 --explicit_controls 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from travel_review_panel_utils import normalize_review_text

PROJECT_ROOT = TOOLS_DIR.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build verification annotation template for Task 2")
    ap.add_argument(
        "--tags_file",
        default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"),
        help="Task 1 long-format output",
    )
    ap.add_argument(
        "--boundary_dir",
        default=str(PROJECT_ROOT / "gold/data/reviewed/boundary_samples"),
    )
    ap.add_argument("--target_size", type=int, default=400)
    ap.add_argument("--explicit_controls", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output_xlsx",
        default=str(PROJECT_ROOT / "gold/data/verification_sample_to_annotate.xlsx"),
    )
    ap.add_argument(
        "--output_csv",
        default=str(PROJECT_ROOT / "gold/data/verification_sample_to_annotate.csv"),
    )
    return ap.parse_args()


def load_boundary_matches(boundary_dir: Path, tags_df: pd.DataFrame) -> pd.DataFrame:
    checklist = boundary_dir / "boundary_samples_annotation_checklist.csv"
    if not checklist.exists():
        return pd.DataFrame()

    boundary = pd.read_csv(checklist)
    text_col = "text_ja" if "text_ja" in boundary.columns else "text"
    boundary["text_norm"] = boundary[text_col].astype(str).map(normalize_review_text)

    unique_sentences = tags_df.drop_duplicates("text_id")[["text_id", "text"]].copy()
    unique_sentences["text_norm"] = unique_sentences["text"].map(normalize_review_text)
    norm_to_id = dict(zip(unique_sentences["text_norm"], unique_sentences["text_id"]))

    boundary["matched_text_id"] = boundary["text_norm"].map(norm_to_id)
    matched = boundary[boundary["matched_text_id"].notna()].copy()
    return matched


def stratified_sample_tags(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    if len(df) <= n:
        return df.copy()

    rng = np.random.default_rng(seed)
    group_cols = ["aspect", "polarity"]
    parts = []
    total = len(df)
    for _, part in df.groupby(group_cols):
        take = max(1, round(n * len(part) / total))
        idx = rng.choice(part.index.to_numpy(), size=min(take, len(part)), replace=False)
        parts.append(df.loc[idx])
    sampled = pd.concat(parts).drop_duplicates(subset=["text_id", "aspect", "polarity"])
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remaining = df[~df.index.isin(sampled.index)]
        extra_n = min(n - len(sampled), len(remaining))
        if extra_n > 0:
            extra = remaining.sample(n=extra_n, random_state=seed)
            sampled = pd.concat([sampled, extra]).drop_duplicates(subset=["text_id", "aspect", "polarity"])
    return sampled.reset_index(drop=True)


def sentence_level_table(tags_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for text_id, part in tags_df.groupby("text_id"):
        row = part.iloc[0]
        implicit_labels = part[part["source"] == "implicit"]
        explicit_labels = part[part["source"] == "explicit"]
        rows.append(
            {
                "text_id": text_id,
                "review_id": row["review_id"],
                "sentence_id": row["sentence_id"],
                "split": row["split"],
                "sentence_text": row["text"],
                "num_active_labels": len(part),
                "num_implicit_labels": len(implicit_labels),
                "num_explicit_labels": len(explicit_labels),
                "has_implicit": len(implicit_labels) > 0,
                "has_explicit": len(explicit_labels) > 0,
                "implicit_aspects": "|".join(sorted(implicit_labels["aspect"].unique())),
                "explicit_aspects": "|".join(sorted(explicit_labels["aspect"].unique())),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tags_df = pd.read_csv(args.tags_file)
    boundary_dir = Path(args.boundary_dir)
    selected_ids: set[str] = set()
    records: list[dict] = []

    # 1) Boundary samples matched to hotel corpus (Mercari-origin; often 0 matches)
    boundary_matches = load_boundary_matches(boundary_dir, tags_df)
    if not boundary_matches.empty:
        for _, brow in boundary_matches.iterrows():
            text_id = str(brow["matched_text_id"])
            if text_id in selected_ids:
                continue
            tag_rows = tags_df[tags_df["text_id"] == text_id]
            for _, trow in tag_rows.iterrows():
                records.append(
                    {
                        "sample_origin": "boundary_matched",
                        "text_id": text_id,
                        "sentence_text": trow["text"],
                        "aspect": trow["aspect"],
                        "polarity": trow["polarity"],
                        "oseti_judgment": trow["source"],
                        "oseti_positive_hits": trow["oseti_positive_hits"],
                        "oseti_negative_hits": trow["oseti_negative_hits"],
                        "human_judgment": "",
                        "implicit_type": "",
                        "notes": f"boundary_sample_type={brow.get('sample_type', '')}",
                    }
                )
            selected_ids.add(text_id)

    print(f"[INFO] Boundary samples matched to hotel text_id: {len(selected_ids)}")

    # 2) Implicit stratified sample at label-pair level
    implicit_pool = tags_df[tags_df["source"] == "implicit"].copy()
    implicit_target = max(0, args.target_size - len(selected_ids) - args.explicit_controls)
    implicit_sample = stratified_sample_tags(implicit_pool, implicit_target, seed=args.seed)
    for _, trow in implicit_sample.iterrows():
        text_id = str(trow["text_id"])
        records.append(
            {
                "sample_origin": "implicit_stratified",
                "text_id": text_id,
                "sentence_text": trow["text"],
                "aspect": trow["aspect"],
                "polarity": trow["polarity"],
                "oseti_judgment": trow["source"],
                "oseti_positive_hits": trow["oseti_positive_hits"],
                "oseti_negative_hits": trow["oseti_negative_hits"],
                "human_judgment": "",
                "implicit_type": "",
                "notes": "",
            }
        )
        selected_ids.add(text_id)

    # 3) Explicit controls (label-pair level)
    explicit_pool = tags_df[
        (tags_df["source"] == "explicit") & (~tags_df["text_id"].isin(selected_ids))
    ].copy()
    explicit_sample = explicit_pool.sample(
        n=min(args.explicit_controls, len(explicit_pool)),
        random_state=args.seed,
    )
    for _, trow in explicit_sample.iterrows():
        text_id = str(trow["text_id"])
        records.append(
            {
                "sample_origin": "explicit_control",
                "text_id": text_id,
                "sentence_text": trow["text"],
                "aspect": trow["aspect"],
                "polarity": trow["polarity"],
                "oseti_judgment": trow["source"],
                "oseti_positive_hits": trow["oseti_positive_hits"],
                "oseti_negative_hits": trow["oseti_negative_hits"],
                "human_judgment": "",
                "implicit_type": "",
                "notes": "",
            }
        )
        selected_ids.add(text_id)

    out_df = pd.DataFrame(records).drop_duplicates(
        subset=["text_id", "aspect", "polarity"]
    )
    out_df.insert(0, "row_id", range(1, len(out_df) + 1))

    out_xlsx = Path(args.output_xlsx)
    out_csv = Path(args.output_csv)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_xlsx, index=False)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "rows": int(len(out_df)),
        "unique_sentences": int(out_df["text_id"].nunique()),
        "by_origin": out_df["sample_origin"].value_counts().to_dict(),
        "implicit_type_choices": [
            "比较级隐含",
            "间接抱怨",
            "条件句暗示",
            "程度弱化",
            "反问",
            "其他",
        ],
        "human_judgment_choices": ["explicit", "implicit", "unclear"],
        "boundary_matches": int(len(boundary_matches)),
        "note": (
            "boundary_samples/ is Mercari-domain; only rows matched to hotel text_id are included. "
            "Primary sample pool is implicit_stratified from Task 1."
        ),
    }
    summary_path = out_xlsx.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote template -> {out_xlsx}")
    print(f"[OK] Wrote CSV backup -> {out_csv}")
    print(f"[OK] Summary -> {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
