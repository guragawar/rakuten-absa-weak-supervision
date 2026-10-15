# -*- coding: utf-8 -*-
"""Summary of experiment L (label-efficiency curve), docs/followup_experiments_plan.md.

For each fraction of the Gold training reviews (10 / 25 / 50 / 100 %) and each epoch schedule
(a: 3 epochs; b: epochs scaled to keep the number of updates), compares E8 (weak pretraining) with
E1 (no pretraining) on the test set with the seed-averaged paired bootstrap, then evaluates the
three pre-registered hypotheses:
    H-L1  delta > 0 at every fraction
    H-L2  delta(10%) - delta(100%) > 0    (difference in differences, one set of resamples for all four groups)
    H-L3  E8 at 50% (and 25%) is not below E1 at 100%
The schedule used for the conclusions is, per fraction, the one with the higher E1 valid negative F1.
Also reports the descriptive "equivalent label fraction" f*: where the E8 curve, interpolated linearly
in log2(fraction), reaches the E1 full-data score.

Output: results/label_efficiency_summary.json and a printed table (numbers only).

    .venv/bin/python tools/label_efficiency_summary.py
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.multiseed_f1_bootstrap import pooled_f1  # noqa: E402
from tools.multiseed_gap_bootstrap import ci, load_preds  # noqa: E402

RES = ROOT / "results"
PROBS = RES / "test_probs"
OUT_DIR = RES / "followup"
SEEDS = [42, 43, 44, 45, 46]
FRACTIONS = [10, 25, 50]
METRICS = ("agg_neg", "implicit_neg", "agg_pos", "micro")


def runs(cond: str, frac: int, sched: str) -> list[str]:
    if frac == 100:
        if cond == "E1":
            return [f"E1_xlmr_seed{s}" for s in SEEDS]
        return [f"A0_ref_E8_seed{s}" for s in (42, 43, 44)] + [f"L_E8_f100_seed{s}" for s in (45, 46)]
    return [f"L{sched}_{cond}_f{frac}_seed{s}" for s in SEEDS]


def have(names: list[str]) -> bool:
    return all((PROBS / f"{n}.npz").exists() for n in names)


def valid_neg(names: list[str]) -> float | None:
    vals = []
    for n in names:
        p = RES / f"{n}_metrics.json"
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text(encoding="utf-8")).get("valid_agg_negative_f1"))
    return None if any(v is None for v in vals) else float(np.mean(vals))


def bootstrap(a: list[str], b: list[str], tag: str) -> dict | None:
    """E(b) - E(a) with tools/multiseed_f1_bootstrap.py; cached in results/followup/."""
    if not (have(a) and have(b)):
        return None
    out = OUT_DIR / f"L_{tag}.json"
    if not out.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/multiseed_f1_bootstrap.py"),
                        "--probs_a", *[str(PROBS / f"{n}.npz") for n in a],
                        "--probs_b", *[str(PROBS / f"{n}.npz") for n in b],
                        "--label_a", tag.split("__vs__")[1], "--label_b", tag.split("__vs__")[0],
                        "--out", str(out)], check=True, capture_output=True)
    m = json.loads(out.read_text(encoding="utf-8"))["metrics"]
    return {k: {"a": m[k]["a_mean"], "b": m[k]["b_mean"], "delta": m[k]["delta_point"], "ci": m[k]["delta_ci"],
                "sig": m[k]["significant_at_95"]} for k in METRICS}


def diff_in_diff(groups: dict[str, list[str]], n_boot: int = 2000, seed: int = 42) -> dict | None:
    """(E8_small - E1_small) - (E8_full - E1_full) on negative aggregate F1, same resamples for all four groups."""
    if not all(have(g) for g in groups.values()):
        return None
    loaded = {k: [load_preds(str(PROBS / f"{n}.npz")) for n in g] for k, g in groups.items()}
    ref = loaded["E1_full"][0]
    y = ref["y_true"].astype(int)
    neg = np.broadcast_to(np.array(["ネガティブ" in c for c in ref["label_cols"]]), y.shape)
    for rs in loaded.values():
        for r in rs:
            assert r["text_ids"] == ref["text_ids"] and r["label_cols"] == ref["label_cols"], r["path"]

    def did(idx=None) -> float:
        yy, cc = (y, neg) if idx is None else (y[idx], neg[idx])
        mean = {k: np.mean([pooled_f1(yy[cc], (r["pred"] if idx is None else r["pred"][idx])[cc]) for r in rs])
                for k, rs in loaded.items()}
        return float((mean["E8_small"] - mean["E1_small"]) - (mean["E8_full"] - mean["E1_full"]))

    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.array([did(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = ci(d)
    return {"delta": did(), "ci": [lo, hi], "P_gt_0": float((d > 0).mean()), "supported": bool(lo > 0)}


def equivalent_fraction(e8_curve: dict[int, float], target: float) -> str:
    """Fraction of Gold at which the E8 curve (linear in log2 fraction) reaches the E1 full-data score."""
    pts = sorted(e8_curve.items())
    if pts[0][1] >= target:
        return f"< {pts[0][0]}%"
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if v0 < target <= v1:
            x = math.log2(f0) + (target - v0) / (v1 - v0) * (math.log2(f1) - math.log2(f0))
            return f"{2 ** x:.0f}%"
    return "> 100%"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"fractions": {}, "schedule_choice": {}, "hypotheses": {}}

    full = bootstrap(runs("E1", 100, ""), runs("E8", 100, ""), "E8_f100__vs__E1_f100")
    summary["fractions"]["100"] = {"both": full}

    for frac in FRACTIONS:
        row = {}
        for sched in ("a", "b"):
            e1, e8 = runs("E1", frac, sched), runs("E8", frac, sched)
            row[sched] = {"valid_neg_E1": valid_neg(e1), "valid_neg_E8": valid_neg(e8),
                          "test": bootstrap(e1, e8, f"L{sched}_E8_f{frac}__vs__L{sched}_E1_f{frac}")}
        summary["fractions"][str(frac)] = row
        va, vb = row["a"]["valid_neg_E1"], row["b"]["valid_neg_E1"]
        if va is not None and vb is not None:  # pre-registered: the schedule that is better for the baseline, on valid
            summary["schedule_choice"][str(frac)] = "a" if va >= vb else "b"

    chosen = summary["schedule_choice"]
    if full and len(chosen) == len(FRACTIONS) and all(summary["fractions"][str(f)][chosen[str(f)]]["test"] for f in FRACTIONS):
        deltas = {f: summary["fractions"][str(f)][chosen[str(f)]]["test"]["agg_neg"] for f in FRACTIONS}
        deltas[100] = full["agg_neg"]
        summary["hypotheses"]["H-L1"] = {
            "per_fraction": {str(f): {"delta": d["delta"], "ci": d["ci"], "sig": d["sig"]} for f, d in sorted(deltas.items())},
            "supported": all(d["ci"][0] > 0 for d in deltas.values())}
        s10 = chosen["10"]
        summary["hypotheses"]["H-L2"] = diff_in_diff({
            "E1_small": runs("E1", 10, s10), "E8_small": runs("E8", 10, s10),
            "E1_full": runs("E1", 100, ""), "E8_full": runs("E8", 100, "")})
        h3 = {}
        for f in (50, 25):
            b = bootstrap(runs("E1", 100, ""), runs("E8", f, chosen[str(f)]), f"L{chosen[str(f)]}_E8_f{f}__vs__E1_f100")
            h3[str(f)] = {"delta": b["agg_neg"]["delta"], "ci": b["agg_neg"]["ci"], "not_below": bool(b["agg_neg"]["ci"][1] >= 0)}
        summary["hypotheses"]["H-L3"] = h3
        e8_curve = {f: deltas[f]["b"] for f in deltas}
        summary["equivalent_fraction"] = {"E8_curve": {str(f): v for f, v in sorted(e8_curve.items())},
                                          "E1_full": full["agg_neg"]["a"],
                                          "f_star": equivalent_fraction(e8_curve, full["agg_neg"]["a"])}

    (RES / "label_efficiency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def cell(t: dict | None, k: str = "agg_neg") -> str:
        if not t:
            return "pending"
        d = t[k]
        return f"{d['a']:.4f} -> {d['b']:.4f}  {100 * d['delta']:+.2f} [{100 * d['ci'][0]:+.2f}, {100 * d['ci'][1]:+.2f}]{'*' if d['sig'] else ''}"

    print("fraction  schedule  valid neg E1/E8    test negative F1: E1 -> E8  delta [95% CI]         implicit negative")
    for frac in FRACTIONS:
        for sched in ("a", "b"):
            r = summary["fractions"][str(frac)][sched]
            v = "pending        " if r["valid_neg_E1"] is None else f"{r['valid_neg_E1']:.4f}/{r['valid_neg_E8']:.4f}"
            mark = "<- used" if chosen.get(str(frac)) == sched else ""
            print(f"{frac:>5}%    {sched}         {v}    {cell(r['test'])}   {cell(r['test'], 'implicit_neg')}  {mark}")
    print(f"  100%    -         -                  {cell(full)}   {cell(full, 'implicit_neg')}")
    for h, v in summary["hypotheses"].items():
        print(h, json.dumps(v, ensure_ascii=False))
    if "equivalent_fraction" in summary:
        print("equivalent label fraction f* =", summary["equivalent_fraction"]["f_star"])
    print("* = 95% CI excludes 0. Written results/label_efficiency_summary.json")


if __name__ == "__main__":
    main()
