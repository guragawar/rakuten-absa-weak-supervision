# -*- coding: utf-8 -*-
"""Part B of docs/model_comparison_plan.md: zero-/few-shot ABSA with a local instruction LLM.

The prompt gives the 7 aspect definitions and the annotation rules of the Gold corpus
(Nakayama et al., LREC 2022, Table 3 and Sec. 3.2) and asks for a JSON list of the 14 label
names. Greedy decoding, so one run per setting is deterministic. Inference is local only:
Gold text never leaves this machine.

Outputs (same format as fine-tuned runs, so tools/multiseed_f1_bootstrap.py works unchanged):
    results/test_probs/<name>.npz           predictions as 0/1 "probs", thresholds 0.5
    results/test_probs/<name>.raw.jsonl     text_id + raw model output (not in git)
    results/<name>_metrics.json             summary metrics, parse failures, settings

Example:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/llm_absa_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct --revision a09a3545... --mode few --name LLM_qwen25_7b_few
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_absa_baseline import load_absa_csv, load_label_columns  # noqa: E402

ASPECT_DEFS = [
    ("朝食", "朝の食事。味、料理、食材を含む"),
    ("夕食", "夜の食事。味、料理、食材を含む"),
    ("風呂", "大浴場、または部屋の風呂"),
    ("サービス", "宿泊者の満足のための、従業員の手助けや対応"),
    ("立地", "アクセス、または周辺の景観"),
    ("設備・アメニティ", "部屋と風呂以外の、ホテルの設備や備品"),
    ("部屋", "部屋そのものの特徴"),
]
RULES = [
    "事実を述べているだけで、評価や意見を含まない文にはラベルを付けない。",
    "感情を表す言葉がなくても、宿泊者が満足しているか不満を持っているかが読み取れれば、ラベルを付ける。",
    "同じ観点について良い点と悪い点の両方が書かれていれば、ポジティブとネガティブの両方を付ける。",
    "要望や提案は、宿泊者がその観点に満足しているか、不満を持っているかで判断する。",
    "1つの文に複数の観点が含まれていれば、該当するものをすべて付ける。",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Zero-/few-shot ABSA evaluation with a local LLM")
    ap.add_argument("--model", default="")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mode", choices=["zero", "few"], default="zero")
    ap.add_argument("--name", required=True, help="run name for output files")
    ap.add_argument("--split", choices=["valid", "test"], default="test")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all (use e.g. 200 on valid to check the format)")
    ap.add_argument("--n_shots", type=int, default=20)
    ap.add_argument("--max_shot_chars", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rescore", action="store_true",
                    help="re-score results/test_probs/<name>.raw.jsonl with the current parser (no model run)")
    ap.add_argument("--reasoning_effort", default=None, choices=[None, "low", "medium", "high"],
                    help="gpt-oss only: passed to the chat template; the answer is read from the final channel")
    ap.add_argument("--labels_file", default=str(ROOT / "data/absa_label_columns.json"))
    return ap.parse_args()


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("・", "").replace(" ", "").strip()


def pick_shots(train: pd.DataFrame, label_cols: list[str], n: int, max_chars: int, seed: int) -> pd.DataFrame:
    """One short sentence per label (in label order), then random short sentences up to n."""
    rng = np.random.default_rng(seed)
    short = train[train["text"].str.len() <= max_chars]
    chosen: list[int] = []
    for col in label_cols:
        pool = short.index[(short[col] == 1) & ~short.index.isin(chosen)]
        chosen.append(int(rng.choice(pool)))
    rest = short.index[~short.index.isin(chosen)]
    chosen += [int(i) for i in rng.choice(rest, size=max(0, n - len(chosen)), replace=False)]
    shots = train.loc[chosen]
    return shots.sample(frac=1.0, random_state=seed)  # shuffle so labels are not in a fixed order


def build_prompt(sentence: str, label_cols: list[str], shots: pd.DataFrame | None) -> str:
    lines = ["次のホテル口コミの1文について、言及されている観点ごとに、宿泊者が満足しているか（ポジティブ）、"
             "不満を持っているか（ネガティブ）を判定してください。", "", "観点の定義:"]
    lines += [f"- {a}: {d}" for a, d in ASPECT_DEFS]
    lines += ["", "判定のルール:"] + [f"{i}. {r}" for i, r in enumerate(RULES, 1)]
    lines += ["", "ラベルは次の14個から選んでください: " + ", ".join(label_cols),
              "出力はJSONの配列だけにしてください。該当するラベルがなければ [] を出力してください。"]
    if shots is not None:
        lines += ["", "例:"]
        for _, row in shots.iterrows():
            labs = [c for c in label_cols if row[c] == 1]
            lines += [f"文: {row['text']}", f"出力: {json.dumps(labs, ensure_ascii=False)}"]
    lines += ["", f"文: {sentence}", "出力:"]
    return "\n".join(lines)


def extract_answer(out: str) -> str:
    """gpt-oss writes its reasoning first; the answer follows the final-channel marker."""
    return out.rsplit("assistantfinal", 1)[1] if "assistantfinal" in out else out


def parse_output(text: str, canon: dict[str, int]) -> tuple[list[int], str]:
    """Return (label indices, mode). mode "json": a JSON array was found (the pre-registered,
    strict parse); "names": no array, but label names appear in the text (lenient fallback,
    added 2026-10-12, see docs/model_comparison_plan.md); "none": nothing usable."""
    m = re.search(r"\[[^\[\]]*\]", text)
    if m:
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            items = re.findall(r"[\"'「]([^\"'」]+)[\"'」]", m.group(0))
        return sorted({canon[norm(str(x))] for x in items if norm(str(x)) in canon}), "json"
    flat = norm(text)
    names = sorted({i for key, i in canon.items() if key in flat})
    return names, ("names" if names else "none")


def score(outputs: list[str], y: np.ndarray, label_cols: list[str], canon: dict[str, int]) -> tuple[dict, dict]:
    """Strict (pre-registered) and lenient prediction matrices and their summaries."""
    strict, lenient = np.zeros_like(y), np.zeros_like(y)
    modes = {"json": 0, "names": 0, "none": 0}
    for r, out in enumerate(outputs):
        idx, mode = parse_output(extract_answer(out), canon)
        modes[mode] += 1
        if mode == "json":
            strict[r, idx] = 1
        lenient[r, idx] = 1
    neg = np.array(["ネガティブ" in c for c in label_cols])

    def summary(pred: np.ndarray) -> dict:
        return {"agg_negative_f1": pooled_f1(y[:, neg], pred[:, neg]),
                "agg_positive_f1": pooled_f1(y[:, ~neg], pred[:, ~neg]),
                "micro_f1": pooled_f1(y, pred),
                "per_label_f1": {c: pooled_f1(y[:, j], pred[:, j]) for j, c in enumerate(label_cols)},
                "mean_labels_pred": float(pred.sum(1).mean())}
    out = {**summary(strict), "parse_failures": modes["names"] + modes["none"], "parse_modes": modes,
           "n": len(outputs), "mean_labels_gold": float(y.sum(1).mean()), "lenient": summary(lenient)}
    return out, {"strict": strict, "lenient": lenient}


def pooled_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(((y == 1) & (p == 1)).sum())
    d = 2 * tp + int(((y == 0) & (p == 1)).sum()) + int(((y == 1) & (p == 0)).sum())
    return 2 * tp / d if d else 0.0


def write_results(args, data: pd.DataFrame, label_cols: list[str], outputs: list[str], canon: dict[str, int],
                  meta: dict) -> None:
    y = data[label_cols].values.astype(int)
    summary, preds = score(outputs, y, label_cols, canon)
    print(f"[RESULT] strict  neg F1 {summary['agg_negative_f1']:.4f}  pos F1 {summary['agg_positive_f1']:.4f}  "
          f"micro {summary['micro_f1']:.4f}  parse modes {summary['parse_modes']}")
    print(f"[RESULT] lenient neg F1 {summary['lenient']['agg_negative_f1']:.4f}  "
          f"pos F1 {summary['lenient']['agg_positive_f1']:.4f}  micro {summary['lenient']['micro_f1']:.4f}")
    probs_dir = ROOT / "results/test_probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    text_ids = data["text_id"].astype(str).tolist()
    if args.split == "test" and not args.max_rows:
        for kind, suffix in (("strict", ""), ("lenient", "_lenient")):
            np.savez_compressed(probs_dir / f"{args.name}{suffix}.npz", probs=preds[kind].astype(np.float32),
                                thresholds=np.full(len(label_cols), 0.5, dtype=np.float32),
                                y_true=y.astype(np.int8), text_ids=np.array(text_ids),
                                label_cols=np.array(label_cols),
                                checkpoint_dir=np.array(f"{meta.get('model')}@{meta.get('revision')} mode={meta.get('mode')} parse={kind}"))
    payload = {**summary, **meta}
    (ROOT / "results" / f"{args.name}_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                                 encoding="utf-8")
    print(f"[INFO] wrote results/{args.name}_metrics.json")


def main() -> None:
    args = parse_args()
    label_cols = load_label_columns(args.labels_file)
    canon = {norm(c): i for i, c in enumerate(label_cols)}
    data = load_absa_csv(str(ROOT / f"data/absa_{args.split}.csv"), label_cols)
    if args.max_rows:
        data = data.head(args.max_rows)
    raw_path = ROOT / "results/test_probs" / f"{args.name}.raw.jsonl"

    if args.rescore:
        # Re-score saved outputs with the current parser; no model is loaded.
        old = json.loads((ROOT / "results" / f"{args.name}_metrics.json").read_text(encoding="utf-8"))
        raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        assert [r["text_id"] for r in raw] == data["text_id"].astype(str).tolist(), "raw outputs do not match data"
        keep = ("model", "revision", "mode", "split", "max_rows", "prompt_sha12", "few_shot_text_ids", "seconds", "args")
        meta = {k: old[k] for k in keep if k in old}
        meta["rescored"] = time.strftime("%Y-%m-%d %H:%M")
        write_results(args, data, label_cols, [r["output"] for r in raw], canon, meta)
        return

    shots = None
    if args.mode == "few":
        shots = pick_shots(load_absa_csv(str(ROOT / "data/absa_train.csv"), label_cols),
                           label_cols, args.n_shots, args.max_shot_chars, args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.bfloat16, device_map="cuda").eval()

    template_kw = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(s, label_cols, shots)}],
                                       tokenize=False, add_generation_prompt=True, **template_kw)
               for s in data["text"].astype(str)]
    prompt_sha = hashlib.sha256(build_prompt("{sentence}", label_cols, shots).encode()).hexdigest()[:12]
    print(f"[INFO] {len(prompts):,} sentences, mode={args.mode}, prompt tokens≈{len(tok(prompts[0]).input_ids)}")

    outputs: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        enc = {k: enc[k] for k in ("input_ids", "attention_mask")}  # some tokenizers (llm-jp) add token_type_ids
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if (i // args.batch_size) % 20 == 0:
            print(f"  {len(outputs):,}/{len(prompts):,}  {time.time() - t0:.0f}s", flush=True)

    with open(raw_path, "w", encoding="utf-8") as f:
        for tid, out in zip(data["text_id"].astype(str), outputs):
            f.write(json.dumps({"text_id": tid, "output": out}, ensure_ascii=False) + "\n")
    meta = {"model": args.model, "revision": args.revision, "mode": args.mode, "split": args.split,
            "max_rows": args.max_rows, "prompt_sha12": prompt_sha,
            "few_shot_text_ids": shots["text_id"].astype(str).tolist() if shots is not None else [],
            "seconds": round(time.time() - t0), "args": vars(args)}
    write_results(args, data, label_cols, outputs, canon, meta)


if __name__ == "__main__":
    main()
