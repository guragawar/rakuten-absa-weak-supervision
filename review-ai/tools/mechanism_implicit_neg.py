# -*- coding: utf-8 -*-
"""Where does model B beat model A on implicit negatives? (mechanism analysis)

Unit: a (test sentence, negative label) cell whose gold label is 1 and whose
explicit/implicit tag is "implicit". A model "detects" the cell when at least
half of its seeds predict 1 (majority vote), which removes seed noise.

Outputs
  - won/lost/both/neither counts and simple text profiles per group
  - recall of A and B on surface-cue strata (negation form, contrast/regret
    markers, sentence length)

Surface-cue definitions are fixed below so the thesis can cite them exactly.

Example:
    .venv/bin/python tools/mechanism_implicit_neg.py \
        --probs_a results/test_probs/E1_xlmr_seed4{2,3,4,5,6}.npz \
        --probs_b results/test_probs/deberta_v3_seed4{2,3,4,5,6}.npz \
        --label_a E1_xlmr --label_b deberta_v3 --out results/repro/mechanism_xlmr_vs_deberta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.multiseed_gap_bootstrap import build_source_masks, load_preds  # noqa: E402

# Negation morphology: ない/なかった/なく, classical ず/ぬ, difficulty にくい/づらい, polite ません
NEGATION = re.compile(r"ない|なかっ|なく|ず|ぬ|にくい|づらい|ません")
# Contrast or regret markers that often signal a complaint without a sentiment word
CONTRAST = re.compile(r"が、|けど|けれど|しかし|ただ|でも|のに|残念|惜しい|もう少し|もっと")
# Degree adverbs
DEGREE = re.compile(r"とても|すごく|かなり|非常に|大変|全然|全く|ちょっと|少し|やや")
SHORT_MAX_CHARS = 35


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Implicit-negative mechanism analysis (majority vote over seeds)")
    ap.add_argument("--probs_a", nargs="+", required=True)
    ap.add_argument("--probs_b", nargs="+", required=True)
    ap.add_argument("--label_a", required=True)
    ap.add_argument("--label_b", required=True)
    ap.add_argument("--test_file", default=str(PROJECT_ROOT / "data/absa_test.csv"))
    ap.add_argument("--tags_file", default=str(PROJECT_ROOT / "data/absa_explicit_implicit_tags.csv"))
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def profile(texts: list[str], aspects: list[str]) -> dict:
    lens = np.array([len(t) for t in texts]) if texts else np.array([0])
    rate = lambda pat: round(float(np.mean([bool(pat.search(t)) for t in texts])), 3) if texts else 0.0
    return {
        "n": len(texts), "mean_len": round(float(lens.mean()), 1), "median_len": int(np.median(lens)),
        "negation_rate": rate(NEGATION), "contrast_rate": rate(CONTRAST), "degree_rate": rate(DEGREE),
        "aspects": dict(Counter(aspects).most_common()),
    }


def main() -> None:
    args = parse_args()
    runs_a = [load_preds(p) for p in args.probs_a]
    runs_b = [load_preds(p) for p in args.probs_b]
    cols, tids = runs_a[0]["label_cols"], runs_a[0]["text_ids"]
    y = runs_a[0]["y_true"].astype(int)
    for r in runs_a + runs_b:
        assert r["text_ids"] == tids and r["label_cols"] == cols, r["path"]

    _, implicit = build_source_masks(pd.read_csv(args.tags_file), tids, cols)
    neg = np.array(["ネガティブ" in c for c in cols])
    rows, cs = np.nonzero(implicit & neg & (y == 1))
    texts = pd.read_csv(args.test_file).set_index("text_id").loc[tids, "text"].astype(str).values[rows].tolist()
    aspects = [cols[c].split("_")[0] for c in cs]

    def detected(runs):
        votes = np.sum([r["pred"][rows, cs] for r in runs], axis=0)
        return votes * 2 >= len(runs)

    da, db = detected(runs_a), detected(runs_b)
    groups = {
        "both_detected": da & db,
        f"{args.label_b}_only(won)": ~da & db,
        f"{args.label_a}_only(lost)": da & ~db,
        "both_missed": ~da & ~db,
    }

    has_neg = np.array([bool(NEGATION.search(t)) for t in texts])
    has_con = np.array([bool(CONTRAST.search(t)) for t in texts])
    short = np.array([len(t) <= SHORT_MAX_CHARS for t in texts])
    strata = {
        "all implicit negative cells": np.ones(len(texts), dtype=bool),
        "with negation form": has_neg,
        "without negation form": ~has_neg,
        "pure implicit (no negation, no contrast/regret)": ~has_neg & ~has_con,
        f"short (<= {SHORT_MAX_CHARS} chars)": short,
        f"long (> {SHORT_MAX_CHARS} chars)": ~short,
    }

    report = {
        "label_a": args.label_a, "label_b": args.label_b,
        "runs_a": [r["path"] for r in runs_a], "runs_b": [r["path"] for r in runs_b],
        "definitions": {
            "unit": "(test sentence, negative label) cell with gold=1 and tag=implicit",
            "detected": "majority vote over seeds (>= half predict 1)",
            "negation": NEGATION.pattern, "contrast_regret": CONTRAST.pattern,
            "degree": DEGREE.pattern, "short_max_chars": SHORT_MAX_CHARS,
        },
        "n_cells": len(texts),
        "recall_majority": {args.label_a: round(float(da.mean()), 4), args.label_b: round(float(db.mean()), 4)},
        "counts": {k: int(v.sum()) for k, v in groups.items()},
        "profiles": {k: profile([t for t, m in zip(texts, v) if m], [a for a, m in zip(aspects, v) if m])
                     for k, v in groups.items()},
        "strata": [],
    }
    print(f"cells={len(texts)}  recall {args.label_a}={da.mean():.4f} {args.label_b}={db.mean():.4f}")
    print("counts:", report["counts"])
    for name, m in strata.items():
        ra, rb = float(da[m].mean()), float(db[m].mean())
        report["strata"].append({"name": name, "n": int(m.sum()), f"recall_{args.label_a}": round(ra, 4),
                                 f"recall_{args.label_b}": round(rb, 4), "delta": round(rb - ra, 4)})
        print(f"  {name:50s} n={int(m.sum()):5d}  {ra:.4f} -> {rb:.4f}  Δ={100 * (rb - ra):+.2f}pp")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
