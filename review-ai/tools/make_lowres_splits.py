# -*- coding: utf-8 -*-
"""Nested low-resource subsets of the Gold training split (experiment L, docs/followup_experiments_plan.md).

For every subset seed s, the review ids of data/absa_train.csv are shuffled with
numpy.random.RandomState(s) and the first ceil(f * N) reviews are kept, so for one seed
10% is a subset of 25%, which is a subset of 50%. Sentences keep their original order.
Subsets are sampled by review (not by sentence), like the train/valid/test split itself.

Output (licensed text, git-ignored):  data/ablation/lowres/absa_train_f{10,25,50}_s{seed}.csv
Statistics only (tracked):            results/lowres_split_stats.json

    .venv/bin/python tools/make_lowres_splits.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_file", default=str(ROOT / "data/absa_train.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/ablation/lowres"))
    ap.add_argument("--fractions", type=int, nargs="+", default=[10, 25, 50], help="percent of training reviews")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--stats_out", default=str(ROOT / "results/lowres_split_stats.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.train_file)
    label_cols = [c for c in df.columns if c.endswith("_ポジティブ") or c.endswith("_ネガティブ")]
    neg_cols = [c for c in label_cols if c.endswith("_ネガティブ")]
    reviews = np.array(sorted(df["review_id"].astype(str).unique()))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def describe(part: pd.DataFrame) -> dict:
        return {"reviews": int(part["review_id"].nunique()), "sentences": int(len(part)),
                "labels": int(part[label_cols].to_numpy().sum()), "negative_labels": int(part[neg_cols].to_numpy().sum())}

    stats = {"train_file": Path(args.train_file).name, "full": describe(df), "subsets": {}}
    rid = df["review_id"].astype(str)
    for seed in args.seeds:
        order = np.random.RandomState(seed).permutation(reviews)
        previous: set[str] = set()
        for frac in sorted(args.fractions):
            keep = set(order[: math.ceil(len(order) * frac / 100)])
            assert previous <= keep, "subsets must be nested"
            previous = keep
            part = df[rid.isin(keep)]
            name = f"absa_train_f{frac}_s{seed}.csv"
            part.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
            stats["subsets"][name] = describe(part)

    Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(stats['subsets'])} subsets to {out_dir}; statistics in {args.stats_out}")
    for name, s in stats["subsets"].items():
        print(f"  {name}: {s['reviews']:,} reviews, {s['sentences']:,} sentences, {s['negative_labels']:,} negative labels")


if __name__ == "__main__":
    main()
