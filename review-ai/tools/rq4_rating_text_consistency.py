# -*- coding: utf-8 -*-
"""RQ4: how well do aspect ratings agree with what the review text talks about?

Keyword-based estimate on the E7 weak-label pool. For each panel aspect head:
    mention_rate | negative label (rating 1-2)
    mention_rate | non-negative label (rating 3-5)
    lift = ratio of the two
    silent_negative_rate = share of negative labels whose text never mentions the
                           aspect (an upper-bound proxy for "rating not grounded in
                           text" label noise; keyword recall is imperfect)

Head order follows prepare_panel_weak_labels.py: 1=立地 2=部屋 3=食事 4=風呂 5=サービス 6=設備.
Keyword lists are deliberately simple and are reported verbatim in the output.

Example:
    python3 tools/rq4_rating_text_consistency.py \
        --inputs data/panel_weak_train.jsonl data/panel_weak_valid.jsonl \
        --out results/rq4_rating_text_consistency_2024.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEADS = {
    1: ("立地", r"立地|場所|駅|アクセス|徒歩|近い|近く|遠い|周辺|コンビニ|繁華街|便利|不便"),
    2: ("部屋", r"部屋|客室|お部屋|ベッド|寝具|枕|狭い|狭く|広い|広く|眺め|景色|窓"),
    3: ("食事", r"食事|朝食|夕食|朝ごはん|朝ご飯|夕飯|料理|バイキング|ビュッフェ|ご飯|ディナー|会席|味"),
    4: ("風呂", r"風呂|浴場|温泉|大浴場|露天|シャワー|湯船|浴槽|お湯|サウナ|湯"),
    5: ("サービス", r"スタッフ|対応|接客|フロント|サービス|従業員|店員|仲居|応対|親切|丁寧|感じ"),
    6: ("設備", r"設備|アメニティ|エアコン|冷蔵庫|テレビ|Wi-?Fi|wifi|WiFi|清掃|掃除|古い|老朽|トイレ|エレベーター|駐車場|館内"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RQ4 rating-text consistency (keyword-based)")
    ap.add_argument("--inputs", nargs="+", required=True, help="panel weak-label jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pats = {h: re.compile(p) for h, (_, p) in HEADS.items()}
    cnt = {h: {"neg": 0, "neg_mention": 0, "non": 0, "non_mention": 0} for h in HEADS}
    rows = 0
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                text = d["text"]
                for h in HEADS:
                    if not d.get(f"mask_{h}", 0):
                        continue
                    mentioned = bool(pats[h].search(text))
                    key = "neg" if d[f"label_{h}"] == 1 else "non"
                    cnt[h][key] += 1
                    cnt[h][key + "_mention"] += mentioned
                rows += 1
                if args.max_rows and rows >= args.max_rows:
                    break
        if args.max_rows and rows >= args.max_rows:
            break

    report = {"inputs": args.inputs, "n_reviews": rows,
              "method": "keyword mention rate by weak label; silent_negative_rate is an upper bound on ungrounded negatives",
              "heads": {}}
    print(f"reviews={rows:,}")
    print(f"{'aspect':6s} {'neg n':>8s} {'mention|neg':>12s} {'mention|non':>12s} {'lift':>6s} {'silent neg':>11s}")
    for h, (name, pat) in HEADS.items():
        c = cnt[h]
        m_neg = c["neg_mention"] / c["neg"] if c["neg"] else 0.0
        m_non = c["non_mention"] / c["non"] if c["non"] else 0.0
        report["heads"][name] = {
            "keywords": pat, "n_negative": c["neg"], "n_non_negative": c["non"],
            "mention_rate_negative": round(m_neg, 4), "mention_rate_non_negative": round(m_non, 4),
            "lift": round(m_neg / m_non, 3) if m_non else None,
            "silent_negative_rate": round(1 - m_neg, 4),
        }
        print(f"{name:6s} {c['neg']:8,d} {m_neg:12.1%} {m_non:12.1%} {m_neg / m_non if m_non else 0:6.2f} {1 - m_neg:11.1%}")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
