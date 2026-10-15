# -*- coding: utf-8 -*-
"""Score human-annotated verification sample against oseti's explicit/implicit
judgment (Task 2 / RQ5).

Reads the annotated file produced by build_verification_sample.py once the
`human_judgment` column has been filled in (explicit / implicit / unclear),
and reports oseti's reliability: accuracy, confusion matrix, Cohen's kappa,
and a breakdown by implicit_type where available.

`unclear` rows are excluded from the primary accuracy/kappa (they are not a
judgment oseti's binary output can be right or wrong against) but their rate
is reported separately as a data-quality signal.

Example:
    python tools/check_oseti_accuracy.py \
        --annotated_file gold/data/verification_sample_to_annotate.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_JUDGMENTS = {"explicit", "implicit"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score oseti explicit/implicit accuracy against human annotation")
    ap.add_argument(
        "--annotated_file",
        default=str(PROJECT_ROOT / "gold/data/verification_sample_to_annotate.xlsx"),
        help="Annotated sample (xlsx or csv) with a filled-in human_judgment column",
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results/task2_oseti_accuracy_report.json"),
    )
    ap.add_argument(
        "--min_annotated",
        type=int,
        default=100,
        help="Minimum non-empty human_judgment rows required before scoring (else abort)",
    )
    return ap.parse_args()


def load_annotated(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Annotated file not found: {path}")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary Cohen's kappa, computed by hand (no extra dependency)."""
    labels = sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(y_true)
    conf = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        conf[idx[t], idx[p]] += 1
    po = np.trace(conf) / n
    row_marg = conf.sum(axis=1) / n
    col_marg = conf.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> None:
    args = parse_args()
    df = load_annotated(Path(args.annotated_file))

    required_cols = {"oseti_judgment", "human_judgment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required column(s): {missing}")

    total_rows = len(df)
    df["human_judgment"] = df["human_judgment"].astype(str).str.strip().str.lower()
    df.loc[df["human_judgment"].isin(["nan", "none", ""]), "human_judgment"] = np.nan

    n_annotated = int(df["human_judgment"].notna().sum())
    print(f"[INFO] total_rows={total_rows} annotated={n_annotated} "
          f"({100 * n_annotated / total_rows:.1f}%)")

    if n_annotated < args.min_annotated:
        msg = (
            f"Only {n_annotated} rows have human_judgment filled "
            f"(< --min_annotated {args.min_annotated}). "
            f"Task 2 requires human annotation of the sample at "
            f"{args.annotated_file} before this script can score reliability. "
            f"This step cannot be automated — it needs an annotator to read each "
            f"sentence and judge explicit/implicit/unclear."
        )
        print(f"[BLOCKED] {msg}")
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "annotated_file": str(args.annotated_file),
            "total_rows": total_rows,
            "n_annotated": n_annotated,
            "status": "blocked_pending_human_annotation",
            "message": msg,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] wrote status report -> {out}")
        return

    ann = df[df["human_judgment"].notna()].copy()
    unclear = ann[ann["human_judgment"] == "unclear"]
    scored = ann[ann["human_judgment"].isin(VALID_JUDGMENTS)].copy()

    unclear_rate = len(unclear) / len(ann) if len(ann) else 0.0
    print(f"[INFO] unclear rate: {len(unclear)}/{len(ann)} = {unclear_rate:.3f}")

    if scored.empty:
        raise SystemExit("No rows with a valid explicit/implicit human_judgment to score.")

    y_true = scored["human_judgment"].values
    y_pred = scored["oseti_judgment"].astype(str).str.strip().str.lower().values

    accuracy = float((y_true == y_pred).mean())
    kappa = cohens_kappa(y_true, y_pred)

    # confusion matrix
    labels = ["explicit", "implicit"]
    conf = pd.crosstab(
        pd.Series(y_true, name="human"), pd.Series(y_pred, name="oseti")
    ).reindex(index=labels, columns=labels, fill_value=0)

    # precision/recall for oseti "implicit" as positive class (the label the
    # thesis's diagnostic pipeline actually consumes downstream)
    tp = int(conf.loc["implicit", "implicit"])
    fp = int(conf.loc["explicit", "implicit"])
    fn = int(conf.loc["implicit", "explicit"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    by_origin = {}
    if "sample_origin" in scored.columns:
        for origin, grp in scored.groupby("sample_origin"):
            gt = grp["human_judgment"].values
            gp = grp["oseti_judgment"].astype(str).str.strip().str.lower().values
            by_origin[origin] = {
                "n": len(grp),
                "accuracy": float((gt == gp).mean()),
            }

    by_type = {}
    if "implicit_type" in scored.columns:
        for itype, grp in scored[scored["implicit_type"].notna()].groupby("implicit_type"):
            gt = grp["human_judgment"].values
            gp = grp["oseti_judgment"].astype(str).str.strip().str.lower().values
            by_type[str(itype)] = {"n": len(grp), "accuracy": float((gt == gp).mean())}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "annotated_file": str(args.annotated_file),
        "total_rows": total_rows,
        "n_annotated": n_annotated,
        "n_unclear": len(unclear),
        "unclear_rate": round(unclear_rate, 4),
        "n_scored": len(scored),
        "status": "scored",
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "kappa_interpretation": (
            "almost perfect" if kappa > 0.8 else
            "substantial" if kappa > 0.6 else
            "moderate" if kappa > 0.4 else
            "fair" if kappa > 0.2 else
            "slight/poor"
        ),
        "implicit_precision": round(precision, 4),
        "implicit_recall": round(recall, 4),
        "implicit_f1": round(f1, 4),
        "confusion_matrix": conf.to_dict(),
        "by_sample_origin": by_origin,
        "by_implicit_type": by_type,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[RESULT] accuracy={accuracy:.4f}  Cohen's kappa={kappa:.4f} ({report['kappa_interpretation']})")
    print(f"[RESULT] oseti 'implicit' precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print("\n[Confusion matrix] rows=human, cols=oseti")
    print(conf)
    if by_origin:
        print("\n[By sample origin]")
        for k, v in by_origin.items():
            print(f"  {k}: n={v['n']} accuracy={v['accuracy']:.4f}")
    print(f"\n[INFO] wrote report -> {out}")


if __name__ == "__main__":
    main()
