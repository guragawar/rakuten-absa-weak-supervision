# -*- coding: utf-8 -*-
"""Derive ablation variants of the E7 weak-label pool (A5 meal split, A6 noise filter).

Input is the mainline pool (data/panel_weak_{train,valid}.jsonl, 2024). Output keeps
the same jsonl layout and writes a stats file in the format load_pos_weight() reads,
so train_panel_weak_pretrain.py only needs --heads / --train_file / --stats_file.

A5 meal_split: replace head 3 (食事) with 3b (朝食) and 3d (夕食).
  Meal target of a review = text keywords first, plan title second:
    text mentions only breakfast -> breakfast;  only dinner -> dinner;  both -> ambiguous
    text mentions neither -> plan title: breakfast-only plan -> breakfast,
                                         dinner-only plan -> dinner, otherwise ambiguous
  Negative rating (1-2): label the target head 1; ambiguous -> mask both.
  Non-negative rating (3-5): set 0 on every meal the review covers (plan includes it or
  the text mentions it); mask the others. Room-only plans are masked on both heads.

A6 noise_filter: mask a negative label (rating 1-2) when the review text mentions none
  of that aspect's keywords (lexicon of tools/rq4_rating_text_consistency.py).
  Non-negative labels are left unchanged.

Example:
    python3 tools/prepare_ablation_weak_labels.py --mode meal_split --out_dir data/ablation/A5_meal_split
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.rq4_rating_text_consistency import HEADS as RQ4_HEADS  # noqa: E402
from tools.travel_review_panel_utils import iter_panel_rows  # noqa: E402

BREAKFAST_TEXT = re.compile(r"朝食|朝ごはん|朝ご飯|朝ゴハン|モーニング|朝の食事|朝バイキング|朝ビュッフェ")
DINNER_TEXT = re.compile(r"夕食|夕飯|晩ご飯|晩御飯|晩ごはん|夕ご飯|夜ご飯|ディナー|会席|懐石")
PLAN_BOTH = re.compile(r"2食|二食|朝夕|夕朝|夕・朝|朝・夕|夕食・朝食|朝食・夕食")
PLAN_DINNER = re.compile(r"夕食付|夕食のみ|夕食つき")
PLAN_BREAKFAST = re.compile(r"朝食付|朝食つき|朝食無料|朝食サービス|朝ごはん付|朝ご飯付|モーニング付")
PLAN_NONE = re.compile(r"素泊|食事なし|食事無し|食事無|朝食なし|朝食無し")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build A5/A6 ablation weak-label pools")
    ap.add_argument("--mode", required=True, choices=["meal_split", "noise_filter"])
    ap.add_argument("--train_file", default=str(PROJECT_ROOT / "data/panel_weak_train.jsonl"))
    ap.add_argument("--valid_file", default=str(PROJECT_ROOT / "data/panel_weak_valid.jsonl"))
    ap.add_argument("--raw_tsv", default=str(PROJECT_ROOT / "gold/data/raw/01_Travel_Review.2024.tsv"),
                    help="raw panel file(s) for plan titles (meal_split only); a glob such as "
                         "'gold/data/raw/01_Travel_Review.*.tsv' reads all years (A5b). With duplicates, "
                         "the earliest year's plan title is used, matching the pool's first-occurrence dedup.")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def plan_meals(plan: str) -> str:
    if PLAN_BOTH.search(plan):
        return "both"
    if PLAN_DINNER.search(plan):
        return "dinner"
    if PLAN_BREAKFAST.search(plan):
        return "breakfast"
    if PLAN_NONE.search(plan):
        return "none"
    return "unknown"


def meal_split(d: dict, plan: str, why: Counter) -> None:
    label, mask = d.pop("label_3"), d.pop("mask_3")
    d.update(label_3b=0, mask_3b=0, label_3d=0, mask_3d=0)
    if not mask:
        return
    text = d["text"]
    b, n = bool(BREAKFAST_TEXT.search(text)), bool(DINNER_TEXT.search(text))
    pm = plan_meals(plan)
    if label == 1:
        if b and not n:
            target, src = "b", "text"
        elif n and not b:
            target, src = "d", "text"
        elif not b and not n and pm == "breakfast":
            target, src = "b", "plan"
        elif not b and not n and pm == "dinner":
            target, src = "d", "plan"
        else:
            why["neg_ambiguous"] += 1
            return
        d[f"label_3{target}"], d[f"mask_3{target}"] = 1, 1
        why[f"neg_{target}_{src}"] += 1
    else:
        covers_b = b or pm in ("breakfast", "both")
        covers_d = n or pm in ("dinner", "both")
        if pm == "none" and not (b or n):
            covers_b = covers_d = False
        if covers_b:
            d["mask_3b"] = 1
        if covers_d:
            d["mask_3d"] = 1
        why["non_b" if covers_b else "non_b_masked"] += 1
        why["non_d" if covers_d else "non_d_masked"] += 1


def noise_filter(d: dict, pats: dict, why: Counter) -> None:
    for h, pat in pats.items():
        if d.get(f"mask_{h}") and d.get(f"label_{h}") == 1:
            if pat.search(d["text"]):
                why[f"{h}_kept"] += 1
            else:
                d[f"mask_{h}"] = 0
                why[f"{h}_masked"] += 1


def head_stats(rows_heads: dict) -> dict:
    out = {}
    for h, c in rows_heads.items():
        valid = c["pos"] + c["neg"]
        out[h] = {"valid": valid, "masked": c["masked"], "pos_neg1_2": c["pos"], "neg_nonneg3_5": c["neg"],
                  "pos_ratio": round(c["pos"] / valid, 4) if valid else 0.0}
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans: dict[str, str] = {}
    if args.mode == "meal_split":
        import glob as _glob
        raws = sorted(_glob.glob(args.raw_tsv)) if any(c in args.raw_tsv for c in "*?[") else [args.raw_tsv]
        for raw in raws:
            for row in iter_panel_rows(Path(raw)):
                plans.setdefault(row.text_hash, row.plan_name)
            print(f"[INFO] {Path(raw).name}: plan titles so far {len(plans):,}", flush=True)
        print(f"[INFO] plan titles loaded: {len(plans):,}")
        heads = ["1", "2", "3b", "3d", "4", "5", "6"]
    else:
        heads = ["1", "2", "3", "4", "5", "6"]
    pats = {str(h): re.compile(p) for h, (_, p) in RQ4_HEADS.items()}

    stats = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
             "source_train": args.train_file, "source_valid": args.valid_file, "heads_order": heads, "splits": {}}
    for split, src in (("train", args.train_file), ("valid", args.valid_file)):
        why: Counter = Counter()
        per_head = {h: {"pos": 0, "neg": 0, "masked": 0} for h in heads}
        missing_plan = 0
        dst = out_dir / f"panel_weak_{split}.jsonl"
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                d = json.loads(line)
                if args.mode == "meal_split":
                    plan = plans.get(d["text_hash"])
                    if plan is None:
                        missing_plan += 1
                        plan = ""
                    meal_split(d, plan, why)
                else:
                    noise_filter(d, pats, why)
                for h in heads:
                    if d.get(f"mask_{h}"):
                        per_head[h]["pos" if d[f"label_{h}"] == 1 else "neg"] += 1
                    else:
                        per_head[h]["masked"] += 1
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
        stats["splits"][split] = {"heads": head_stats(per_head), "decisions": dict(why), "missing_plan": missing_plan}
        print(f"[{split}] decisions: {dict(sorted(why.items()))}  missing_plan={missing_plan}")
        for h in heads:
            c = per_head[h]
            print(f"   head {h:3s} pos(neg-rating)={c['pos']:7,d} non={c['neg']:8,d} masked={c['masked']:8,d}")
    (out_dir / "panel_weak_label_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
