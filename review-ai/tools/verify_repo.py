# -*- coding: utf-8 -*-
"""Repository self-check. CPU only, about a minute. Exit code 1 if any check fails.

    .venv/bin/python tools/verify_repo.py

1. scripts   every thesis script compiles; scripts with a CLI also start (--help)
2. numbers   headline numbers recomputed from results/test_probs/*.npz match the
             values written in docs/thesis_numbers.md
3. sources   every results/... and data/... file cited in docs/thesis_numbers.md exists
4. licensing no git-tracked file contains review text (long Japanese lines),
             except documents we wrote ourselves (allowlist below)

Run it before committing and after any change to results or the ledger.
"""

from __future__ import annotations

import py_compile
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # review-ai/
REPO = ROOT.parent                                      # git root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.multiseed_gap_bootstrap import build_source_masks, load_preds  # noqa: E402

LEDGER = ROOT / "docs/thesis_numbers.md"
PROBS = ROOT / "results/test_probs"

# Scripts that are libraries or run work without a CLI: compile only.
NO_CLI = {"travel_review_panel_utils.py", "analyze_uploaded_data.py",
          "profile_travel_review_panel.py", "profile_travel_review_raw.py"}

# Tracked files that legitimately contain long Japanese lines (our own writing).
TEXT_ALLOWLIST = (
    "meetings/", "review-ai/docs/", "review-ai/legacy/mercari_3class/gold/docs/",
    "review-ai/legacy/mercari_3class/gold/reports/", "review-ai/legacy/mercari_3class/gold/metadata/",
    "review-ai/legacy/mercari_3class/gold/data/reviewed/", "review-ai/legacy/mercari_3class/docs/",
    # the Japanese LLM prompt (aspect definitions and annotation rules, written by us); checked on 2026-10-15:
    # none of its Japanese runs occurs in a Gold sentence and no Gold sentence occurs in it
    "review-ai/tools/llm_absa_eval.py",
)
LONG_JA = re.compile(r"[぀-ヿ㐀-鿿、。！？ー]{25,}")

failures: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


# ── 1. scripts ──────────────────────────────────────────────────────────────
def check_scripts() -> None:
    print("1. scripts")
    scripts = sorted(ROOT.glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    for path in scripts:
        rel = path.relative_to(ROOT)
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            report(False, str(rel), f"compile error: {exc.msg.strip().splitlines()[-1]}")
            continue
        if path.name in NO_CLI or path.name == Path(__file__).name:
            report(True, str(rel), "compiles (no CLI)")
            continue
        proc = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=180)
        ok = proc.returncode == 0 and "usage" in proc.stdout.lower()
        report(ok, str(rel), "" if ok else (proc.stderr.strip().splitlines() or ["no output"])[-1][:120])


# ── 2. numbers ──────────────────────────────────────────────────────────────
def pooled_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & (p == 1)))
    denom = 2 * tp + int(np.sum((y == 0) & (p == 1))) + int(np.sum((y == 1) & (p == 0)))
    return 2 * tp / denom if denom else 0.0


def seed_mean(names: list[str], metric: str) -> float:
    runs = [load_preds(str(PROBS / f"{n}.npz")) for n in names]
    y = runs[0]["y_true"].astype(int)
    cols = runs[0]["label_cols"]
    neg = np.array(["ネガティブ" in c for c in cols])
    if metric == "agg_neg":
        cells = np.broadcast_to(neg, y.shape)
    elif metric == "implicit_neg":
        _, implicit = build_source_masks(pd.read_csv(ROOT / "data/absa_explicit_implicit_tags.csv"),
                                         runs[0]["text_ids"], cols)
        cells = implicit & neg
    else:
        raise ValueError(metric)
    return float(np.mean([pooled_f1(y[cells], r["pred"][cells]) for r in runs]))


def seeds(prefix: str, ss=(42, 43, 44)) -> list[str]:
    return [f"{prefix}{s}" for s in ss]


E1_XLMR = seeds("E1_xlmr_seed", (42, 43, 44, 45, 46))
E8_XLMR = seeds("A0_ref_E8_seed") + seeds("A7_E7s43_E8_seed") + seeds("A7_E7s44_E8_seed")

# (description, runs, metric, value as written in the ledger)
HEADLINE = [
    ("A1 E1 XLM-R, negative F1 (5 runs)", E1_XLMR, "agg_neg", "0.7066"),
    ("A1 E8 XLM-R, negative F1 (9 runs)", E8_XLMR, "agg_neg", "0.7257"),
    ("A1 E1 XLM-R, implicit negative F1", E1_XLMR, "implicit_neg", "0.8403"),
    ("A1 E8 XLM-R, implicit negative F1", E8_XLMR, "implicit_neg", "0.8546"),
    ("DeBERTa E1 + E7/E8, negative F1 (3 runs)", seeds("E8_deberta_recent2M_seed"), "agg_neg", "0.7447"),
    ("DeBERTa E1 + E7-SCL/E8, negative F1 (3 runs)", seeds("E8_scl_deberta_recent2M_seed"), "agg_neg", "0.7259"),
]


def check_numbers() -> None:
    print("2. numbers (recomputed from results/test_probs vs docs/thesis_numbers.md)")
    if not PROBS.is_dir():
        report(False, "results/test_probs/ missing", "predictions are not in git; run on the lab server")
        return
    ledger = LEDGER.read_text(encoding="utf-8")
    for desc, names, metric, written in HEADLINE:
        missing = [n for n in names if not (PROBS / f"{n}.npz").exists()]
        if missing:
            report(False, desc, f"missing {missing[0]}.npz")
            continue
        value = seed_mean(names, metric)
        ok = f"{value:.4f}" == written and written in ledger
        report(ok, desc, f"recomputed {value:.4f}, ledger {written}"
               + ("" if written in ledger else " (value not found in ledger)"))
    delta = 100 * (seed_mean(E8_XLMR, "agg_neg") - seed_mean(E1_XLMR, "agg_neg"))
    report(f"+{delta:.2f}pp" in ledger, "A1 main delta", f"recomputed +{delta:.2f}pp")


# ── 3. sources ──────────────────────────────────────────────────────────────
def check_sources() -> None:
    print("3. sources cited in docs/thesis_numbers.md")
    ledger = LEDGER.read_text(encoding="utf-8")
    cited = sorted(set(re.findall(r"`((?:results|data)/[^`*{}\s]+\.(?:json|jsonl|csv))`", ledger)))
    missing = [c for c in cited if not (ROOT / c).exists()]
    report(not missing, f"{len(cited)} cited files exist",
           ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))


# ── 4. licensing ────────────────────────────────────────────────────────────
def check_licensing() -> None:
    print("4. licensed text guard (git-tracked files)")
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    offenders = []
    for rel in tracked:
        if rel.startswith(TEXT_ALLOWLIST) or rel.endswith((".png", ".pdf", ".pptx", ".npz")):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8", errors="ignore")
        except (IsADirectoryError, FileNotFoundError):
            continue
        hits = sum(1 for line in text.splitlines() if LONG_JA.search(line))
        if hits >= 5:
            offenders.append(f"{rel} ({hits} lines)")
    report(not offenders, f"{len(tracked)} tracked files scanned",
           "; ".join(offenders[:5]) if offenders else "no review text outside the allowlist")


def main() -> None:
    check_scripts()
    check_numbers()
    check_sources()
    check_licensing()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
