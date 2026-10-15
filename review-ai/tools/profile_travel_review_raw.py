# -*- coding: utf-8 -*-
"""Profile gold/data/raw/01_Travel_Review.2024.tsv to assess research usability.

Outputs JSON stats to results/travel_review_2024_profile.json.
Read-only with respect to the raw data.
"""
import json
import random
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "gold/data/raw/01_Travel_Review.2024.tsv"
HOTEL_MASTER = BASE / "gold/data/02_Travel_HotelMaster.tsv"
OUT = BASE / "results/travel_review_2024_profile.json"

N_FIELDS = 18

users = Counter()
hotels = Counter()
dates = []
purpose = Counter()
companion = Counter()
col10 = Counter()
ratings_sum = [Counter() for _ in range(6)]  # cols 11-16
review_lens = []
empty_review = 0
empty_reply = 0
n_ok = 0
n_bad = 0
sample_reviews = []

random.seed(42)

with RAW.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < N_FIELDS:
            n_bad += 1
            continue
        if len(fields) > N_FIELDS:
            # embedded tabs in text; merge extras into review text (best effort)
            fields = fields[:16] + ["\t".join(fields[16:-1]), fields[-1]]
        n_ok += 1
        users[fields[0]] += 1
        dates.append(fields[1][:7])  # YYYY-MM
        hotels[fields[2]] += 1
        purpose[fields[7]] += 1
        companion[fields[8]] += 1
        col10[fields[9]] += 1
        for j in range(6):
            ratings_sum[j][fields[10 + j]] += 1
        review = fields[16].strip()
        reply = fields[17].strip()
        if not review:
            empty_review += 1
        else:
            review_lens.append(len(review))
        if not reply:
            empty_reply += 1
        if review and random.random() < 0.0001:
            sample_reviews.append(review[:120])

# Hotel master overlap
master_ids = set()
with HOTEL_MASTER.open(encoding="utf-8") as f:
    for line in f:
        parts = line.split("\t", 1)
        if parts:
            master_ids.add(parts[0].strip())

review_hotel_ids = set(hotels.keys())
overlap = review_hotel_ids & master_ids

date_counter = Counter(dates)
review_lens.sort()


def pct(x, n):
    return round(x / n * 100, 2) if n else 0


def quantile(sorted_list, q):
    if not sorted_list:
        return 0
    idx = min(int(len(sorted_list) * q), len(sorted_list) - 1)
    return sorted_list[idx]


result = {
    "file": str(RAW),
    "size_mb": round(RAW.stat().st_size / 1e6, 1),
    "rows_ok": n_ok,
    "rows_malformed_short": n_bad,
    "unique_users": len(users),
    "unique_hotels": len(review_hotel_ids),
    "hotels_in_master": len(overlap),
    "hotels_in_master_pct": pct(len(overlap), len(review_hotel_ids)),
    "date_range": [min(date_counter), max(date_counter)],
    "reviews_per_month_top": dict(date_counter.most_common(5)),
    "months_covered": len(date_counter),
    "purpose_dist": dict(purpose.most_common(10)),
    "companion_dist": dict(companion.most_common(10)),
    "col10_dist_top": dict(col10.most_common(10)),
    "rating_cols_11_16_dist": [dict(c.most_common(8)) for c in ratings_sum],
    "empty_review": empty_review,
    "empty_reply": empty_reply,
    "empty_reply_pct": pct(empty_reply, n_ok),
    "review_len_median": quantile(review_lens, 0.5),
    "review_len_p90": quantile(review_lens, 0.9),
    "review_len_mean": round(sum(review_lens) / len(review_lens), 1) if review_lens else 0,
    "top_users": dict(users.most_common(5)),
    "reviews_per_hotel_median": sorted(hotels.values())[len(hotels) // 2] if hotels else 0,
    "sample_reviews": sample_reviews[:8],
}

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
