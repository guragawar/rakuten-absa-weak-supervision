# -*- coding: utf-8 -*-
"""Approximate (near-duplicate) overlap check: panel train pool vs Gold reviews.

The exact-match check (P0-1) found 0 overlaps. This adds a MinHash + LSH
near-duplicate pass to catch reviews that are highly similar but not byte-equal
after normalization (e.g. Gold extraction did light cleaning/truncation).

Efficient design (no external LSH lib): build LSH band buckets over the ~12.5k
Gold reviews, then stream the panel; for each panel review look up candidate
Gold ids in its bands and compute the exact shingle Jaccard only for those
candidates. Any panel review with Jaccard >= threshold to some Gold review is
reported and (optionally) appended to the exclude-hash list.

Example:
    python tools/approx_dedup_panel_gold.py --threshold 0.85 --update_exclude
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xxhash

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from travel_review_panel_utils import normalize_review_text, project_root
from check_panel_gold_overlap import load_gold_reviews, load_review_split_map

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

P = (1 << 31) - 1  # Mersenne prime for universal hashing


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MinHash+LSH near-duplicate panel vs Gold")
    ap.add_argument("--panel_csv", default=str(project_root() / "data/panel_train_pool.csv"))
    ap.add_argument("--gold_tsv", default=str(project_root() / "gold/data/travel_aspect_sentiment.tsv"))
    ap.add_argument("--absa_dir", default=str(project_root() / "data"))
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--num_perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=32)  # r = num_perm // bands = 4
    ap.add_argument("--shingle_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="Debug cap on panel rows (0=all)")
    ap.add_argument("--out", default=str(project_root() / "results/approx_dedup_panel_gold_2024.json"))
    ap.add_argument("--exclude_path", default=str(project_root() / "data/panel_exclude_text_hashes.jsonl"))
    ap.add_argument("--update_exclude", action="store_true",
                    help="Append near-duplicate panel text_hashes to the exclude list")
    return ap.parse_args()


def shingle_int_set(norm: str, k: int) -> set[int]:
    if len(norm) < k:
        toks = {norm} if norm else set()
    else:
        toks = {norm[i:i + k] for i in range(len(norm) - k + 1)}
    return {xxhash.xxh32(t).intdigest() & 0x7FFFFFFF for t in toks}


def minhash_signature(shingles: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if shingles.size == 0:
        return np.full(A.shape[0], P, dtype=np.int64)
    vals = (A[:, None] * shingles[None, :] + B[:, None]) % P  # (num_perm, n)
    return vals.min(axis=1)


def band_keys(sig: np.ndarray, bands: int, r: int) -> list[int]:
    return [xxhash.xxh64(sig[b * r:(b + 1) * r].tobytes()).intdigest() for b in range(bands)]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)
    num_perm, bands = args.num_perm, args.bands
    if num_perm % bands != 0:
        raise SystemExit("num_perm must be divisible by bands")
    r = num_perm // bands
    A = rng.integers(1, P, size=num_perm, dtype=np.int64)
    B = rng.integers(0, P, size=num_perm, dtype=np.int64)

    # ---- Build Gold LSH index ----
    gold_tsv = Path(args.gold_tsv)
    if not gold_tsv.exists():
        raise SystemExit(f"Gold TSV not found: {gold_tsv}")
    _df, full_reviews, _meta = load_gold_reviews(gold_tsv)
    split_map = load_review_split_map(Path(args.absa_dir))

    gold_shingles: dict[str, set[int]] = {}
    buckets: list[dict[int, set]] = [defaultdict(set) for _ in range(bands)]
    n_gold = 0
    for rid, text in full_reviews.items():
        norm = normalize_review_text(text)
        if not norm:
            continue
        sset = shingle_int_set(norm, args.shingle_k)
        gold_shingles[rid] = sset
        sig = minhash_signature(np.fromiter(sset, dtype=np.int64, count=len(sset)), A, B)
        for bi, key in enumerate(band_keys(sig, bands, r)):
            buckets[bi][key].add(rid)
        n_gold += 1
    print(f"[INFO] indexed {n_gold:,} Gold reviews (num_perm={num_perm}, bands={bands}, r={r}, k={args.shingle_k})")

    # ---- Stream panel, query ----
    panel_csv = Path(args.panel_csv)
    hits = []
    near_hashes: set[str] = set()
    n_panel = 0
    n_candidate_rows = 0
    with panel_csv.open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if args.limit and n_panel >= args.limit:
                break
            n_panel += 1
            norm = normalize_review_text(row.get("review_text", ""))
            if not norm:
                continue
            pset = shingle_int_set(norm, args.shingle_k)
            sig = minhash_signature(np.fromiter(pset, dtype=np.int64, count=len(pset)), A, B)
            cand: set[str] = set()
            for bi, key in enumerate(band_keys(sig, bands, r)):
                b = buckets[bi].get(key)
                if b:
                    cand |= b
            if not cand:
                continue
            n_candidate_rows += 1
            best_rid, best_j = None, 0.0
            for rid in cand:
                j = jaccard(pset, gold_shingles[rid])
                if j > best_j:
                    best_j, best_rid = j, rid
            if best_j >= args.threshold:
                th = row.get("text_hash", "")
                near_hashes.add(th)
                if len(hits) < 50:
                    hits.append({
                        "panel_text_hash": th,
                        "gold_review_id": best_rid,
                        "jaccard": round(best_j, 4),
                        "gold_split": split_map.get(best_rid, "unknown"),
                        "panel_excerpt": row.get("review_text", "")[:80],
                    })
            if n_panel % 100000 == 0:
                print(f"[INFO] scanned {n_panel:,} panel rows, near-dups so far={len(near_hashes):,}")

    test_hits = sum(1 for h in hits if h["gold_split"] == "test")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel_csv": str(panel_csv),
        "gold_tsv": str(gold_tsv),
        "params": {"threshold": args.threshold, "num_perm": num_perm, "bands": bands,
                   "r": r, "shingle_k": args.shingle_k},
        "gold_reviews_indexed": n_gold,
        "panel_rows_scanned": n_panel,
        "panel_rows_with_candidates": n_candidate_rows,
        "near_duplicate_panel_rows": len(near_hashes),
        "near_duplicate_pct": round(100.0 * len(near_hashes) / n_panel, 4) if n_panel else 0.0,
        "near_dup_matching_test_split_examples": test_hits,
        "examples": hits,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[RESULT] panel scanned={n_panel:,} candidates={n_candidate_rows:,} "
          f"near-duplicates(J>={args.threshold})={len(near_hashes):,} "
          f"({report['near_duplicate_pct']}%)")
    if near_hashes and args.update_exclude:
        exclude_path = Path(args.exclude_path)
        existing = set()
        if exclude_path.exists():
            for line in exclude_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        existing.add(json.loads(line).get("text_hash", json.loads(line)) if line.startswith("{") else line)
                    except Exception:
                        existing.add(line)
        new = near_hashes - existing
        with exclude_path.open("a", encoding="utf-8") as fh:
            for h in sorted(new):
                fh.write(json.dumps({"text_hash": h, "reason": "minhash_near_dup_gold"}, ensure_ascii=False) + "\n")
        print(f"[INFO] appended {len(new):,} new near-dup hashes to {exclude_path}")
    print(f"[INFO] Wrote report -> {out}")


if __name__ == "__main__":
    main()
