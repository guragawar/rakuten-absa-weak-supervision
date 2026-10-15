"""一次性分析新上传的 3 份数据：HotelMaster / 筑波语料 / travel_aspect_sentiment。
输出 JSON 统计，供报告使用。仅读，不修改原始数据。
"""
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path("/home/etl/gujincheng/review-ai/gold/data")
OUT = Path("/home/etl/gujincheng/review-ai/results/uploaded_data_profile.json")

result = {}


def analyze_hotel_master():
    f = DATA / "02_Travel_HotelMaster.tsv"
    df = pd.read_csv(f, sep="\t", header=None, names=["hotel_id", "hotel_name"], dtype=str, quoting=3)
    info = {
        "file": f.name,
        "rows": int(len(df)),
        "columns": ["hotel_id", "hotel_name"],
        "unique_hotel_id": int(df["hotel_id"].nunique()),
        "unique_hotel_name": int(df["hotel_name"].nunique()),
        "dup_hotel_id": int(df["hotel_id"].duplicated().sum()),
        "missing_name": int(df["hotel_name"].isna().sum() + (df["hotel_name"].astype(str).str.strip() == "").sum()),
        "sample": df.head(3).values.tolist(),
    }
    return info


def analyze_tsukuba():
    f = DATA / "annotation01_tsukuba_corpus_20140930.tsv"
    df = pd.read_csv(
        f, sep="\t", header=None,
        names=["seq", "review_id", "sent_id", "label1", "label2", "text"],
        dtype=str, quoting=3,
    )
    n = len(df)
    label1 = df["label1"].fillna("(空)").value_counts()
    label2 = df["label2"].fillna("(空)").value_counts()
    agree = int((df["label1"] == df["label2"]).sum())
    empty_text = int(df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum())
    info = {
        "file": f.name,
        "rows": int(n),
        "columns": ["seq", "review_id", "sent_id", "label1", "label2", "text"],
        "unique_reviews": int(df["review_id"].nunique()),
        "label1_distribution": {str(k): int(v) for k, v in label1.items()},
        "label2_distribution": {str(k): int(v) for k, v in label2.items()},
        "label_agreement": agree,
        "label_agreement_pct": round(agree / n * 100, 1) if n else 0,
        "empty_text": empty_text,
        "sample": df.head(3).values.tolist(),
    }
    return info


def analyze_aspect():
    f = DATA / "travel_aspect_sentiment.tsv"
    lines = f.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    # 表头有 18 列(开头多一个空索引名)，但数据行是 17 列：
    # レビューID, 文ID, レビュー文, + 14 个方面标注列
    aspect_cols = header[4:]
    n_aspect = len(aspect_cols)  # 14
    n_meta = 2  # レビューID, 文ID (レビュー文 在中间)

    rows = []
    bad_extra_tab = 0
    for ln in lines[1:]:
        fields = ln.split("\t")
        min_fields = n_meta + 1 + n_aspect  # 17
        if len(fields) < min_fields:
            fields = fields + [""] * (min_fields - len(fields))
        if len(fields) > min_fields:
            bad_extra_tab += 1
        review_id, sent_id = fields[0], fields[1]
        aspects = fields[-n_aspect:]
        text = "\t".join(fields[n_meta:-n_aspect])
        rows.append([review_id, sent_id, text] + aspects)

    df = pd.DataFrame(rows, columns=["レビューID", "文ID", "レビュー文"] + aspect_cols)
    meta_cols = ["レビューID", "文ID", "レビュー文"]

    n = len(df)
    def to01(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)

    ann = df[aspect_cols].apply(to01)
    per_col = {c: int(ann[c].sum()) for c in aspect_cols}

    # 按方面汇总（去掉 _ポジティブ / _ネガティブ 后缀）
    aspects = {}
    for c in aspect_cols:
        if "_" in c:
            asp, pol = c.rsplit("_", 1)
        else:
            asp, pol = c, "?"
        aspects.setdefault(asp, {"ポジティブ": 0, "ネガティブ": 0})
        if pol in aspects[asp]:
            aspects[asp][pol] += per_col[c]

    marks_per_sentence = ann.sum(axis=1)
    sentences_with_any = int((marks_per_sentence > 0).sum())
    sentences_none = int((marks_per_sentence == 0).sum())
    multi_label = int((marks_per_sentence > 1).sum())
    total_marks = int(marks_per_sentence.sum())

    text = df["レビュー文"].astype(str)
    empty_text = int((text.str.strip() == "").sum() + df["レビュー文"].isna().sum())

    pos_total = sum(v["ポジティブ"] for v in aspects.values())
    neg_total = sum(v["ネガティブ"] for v in aspects.values())

    info = {
        "file": f.name,
        "rows": int(n),
        "meta_columns": meta_cols,
        "aspect_columns_count": len(aspect_cols),
        "unique_reviews": int(df["レビューID"].nunique()),
        "max_sentences_per_review": int(pd.to_numeric(df["文ID"], errors="coerce").max()) + 1,
        "rows_with_extra_tab_in_text": bad_extra_tab,
        "empty_text": empty_text,
        "sentences_with_annotation": sentences_with_any,
        "sentences_no_annotation": sentences_none,
        "sentences_no_annotation_pct": round(sentences_none / n * 100, 1) if n else 0,
        "multi_label_sentences": multi_label,
        "total_marks": total_marks,
        "positive_marks_total": pos_total,
        "negative_marks_total": neg_total,
        "per_aspect": aspects,
        "per_column": per_col,
    }
    return info


result["hotel_master"] = analyze_hotel_master()
result["tsukuba"] = analyze_tsukuba()
result["aspect"] = analyze_aspect()

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
