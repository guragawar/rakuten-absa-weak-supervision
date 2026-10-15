# -*- coding: utf-8 -*-
"""Build the Chinese model-comparison report (report.html) from result files.

Every number is read from review-ai/results/ (or quoted from docs/related_work.md for
published studies). Re-run after new results arrive:

    review-ai/.venv/bin/python reports/model_comparison/build_report.py
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "review-ai" / "results"


def load(rel: str) -> dict | None:
    p = RES / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


S = load("model_comparison_summary.json") or {}

# ── rows of the main table ────────────────────────────────────────────────────
ENC = [  # key, display name, params, note
    ("mbert", "mBERT（多语言 BERT）", "178M", "经典多语言基线"),
    ("xlmr_base", "XLM-R base", "278M", "本研究主线底座"),
    ("deberta_ja", "DeBERTa-v3-ja（京大）", "160M", "此前最强底座"),
    ("bert_large_ja", "东北大 BERT large", "337M", "先行研究 BERT-MLC 的大模型版"),
    ("xlmr_large", "XLM-R large", "560M", "主线底座的大模型版"),
    ("luke_ja", "LUKE-ja（lite）", "133M", "先行研究（JSAI 2026）所用模型"),
    ("modernbert_ja_130m_lr1e-5", "ModernBERT-ja 130M", "132M", "2025 年新模型；1 个 seed 训练崩溃"),
    ("modernbert_ja_310m", "ModernBERT-ja 310M", "315M", "2025 年新模型；valid 最高"),
]
OURS = [  # label, test neg, sd, n runs, source note
    ("XLM-R base + E7/E8", 0.7257, None, "9 次（E7 3 seed × E8 3 seed）", "ablation_A1_main_E1x5_vs_E8x9.json"),
    ("DeBERTa-ja + E7/E8", 0.7447, None, "3 次", "repro/stacking_3seed.json"),
]
LLMS = [
    ("LLM_qwen25_7b", "Qwen2.5-7B-Instruct"),
    ("LLM_swallow8b", "Llama-3.1-Swallow-8B（日语强化）"),
    ("LLM_llmjp31_13b", "llm-jp-3.1-13B-instruct4（NII）"),
    ("LLM_gptoss20b", "gpt-oss-20B（OpenAI 开源权重）"),
]


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def pp(delta, ci=None, sig=None):
    if delta is None:
        return "—"
    s = f"{100 * delta:+.2f}"
    if ci:
        s += f" <span class=ci>[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]</span>"
    if sig:
        s += " <span class=sig>显著</span>"
    return s


def boot(row: dict, ref: str, metric="agg_neg"):
    b = (row or {}).get("bootstrap", {}).get(ref, {}).get(metric)
    return (b["delta"], b["ci"], b["sig"]) if b else (None, None, None)


# ── encoder table ─────────────────────────────────────────────────────────────
def encoder_rows() -> str:
    out = []
    for key, name, params, note in ENC:
        r = S.get(key)
        if not r:
            out.append(f"<tr><td>{name}</td><td class=num>{params}</td><td colspan=5 class=muted>运行中</td></tr>")
            continue
        vn, tn = r.get("valid_neg"), r.get("test_neg")
        d1 = boot(r, "xlmr_E1")
        d2 = boot(r, "deberta_E7E8")
        out.append(
            f"<tr><td>{name}</td><td class=num>{params}</td><td class=num>{fmt(vn[0]) if vn else '—'}</td>"
            f"<td class=num><b>{fmt(tn[0])}</b> <span class=ci>±{tn[1]:.3f}</span></td>"
            f"<td class=num>{fmt(r['test_pos'][0])}</td>"
            f"<td class=num>{pp(*d1) if key != 'xlmr_base' else '（参照）'}</td>"
            f"<td class=num>{pp(*d2)}</td></tr>")
    return "\n".join(out)


def llm_rows() -> str:
    out = []
    for key, name in LLMS:
        for mode, mlabel in (("zero", "零样本"), ("few", "少样本")):
            st, le = S.get(f"{key}_{mode}"), S.get(f"{key}_{mode}_lenient")
            if not st:
                out.append(f"<tr><td>{name}</td><td class=nw>{mlabel}</td><td colspan=5 class=muted>运行中</td></tr>")
                continue
            d = boot(st, "xlmr_E1")
            out.append(
                f"<tr><td>{name}</td><td class=nw>{mlabel}</td><td class=num><b>{fmt(st['test_neg'][0])}</b></td>"
                f"<td class=num>{fmt(le['test_neg'][0]) if le else '—'}</td>"
                f"<td class=num>{fmt(st['test_pos'][0])}</td><td class=num>{st.get('parse_failures', 0):,}</td>"
                f"<td class=num>{pp(*d)}</td></tr>")
    return "\n".join(out)


# ── chart: test negative F1 for every configuration ──────────────────────────
def chart() -> str:
    items = []
    for key, name, _, _ in ENC:
        r = S.get(key)
        if r:
            items.append((name, r["test_neg"][0], "enc"))
    for label, v, *_ in OURS:
        items.append((label, v, "ours"))
    a = S.get("modernbert_310m_E7E8 (A')")
    if a:
        items.append(("ModernBERT-ja 310M + E7/E8", a["test_neg"][0], "ours"))
    for key, name in LLMS:
        best = None
        for mode in ("zero", "few"):
            r = S.get(f"{key}_{mode}_lenient") or S.get(f"{key}_{mode}")
            if r and (best is None or r["test_neg"][0] > best[1]):
                best = (f"{name.split('（')[0]} {'少样本' if mode == 'few' else '零样本'}", r["test_neg"][0])
        if best:
            items.append((best[0], best[1], "llm"))
    items.sort(key=lambda t: t[1])
    x0, x1, left, width, bar, gap = 0.2, 0.8, 250, 470, 18, 8
    h = len(items) * (bar + gap) + 40
    sx = lambda v: left + (v - x0) / (x1 - x0) * width
    parts = [f'<svg viewBox="0 0 {left + width + 70} {h}" role="img" aria-label="各配置的 test 负面 F1">']
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        parts.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="0" y2="{h - 26}" class="grid"/>'
                     f'<text x="{sx(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for i, (name, v, g) in enumerate(items):
        y = i * (bar + gap) + 4
        parts.append(f'<text x="{left - 10}" y="{y + bar * 0.72:.1f}" class="lab" text-anchor="end">{html.escape(name)}</text>'
                     f'<rect x="{sx(x0):.1f}" y="{y}" width="{sx(v) - sx(x0):.1f}" height="{bar}" rx="4" class="b-{g}"><title>{html.escape(name)}: {v:.4f}</title></rect>'
                     f'<text x="{sx(v) + 6:.1f}" y="{y + bar * 0.72:.1f}" class="val">{v:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ── per-aspect / explicit-implicit table ──────────────────────────────────────
ASP = ["朝食", "夕食", "風呂", "サービス", "立地", "設備・アメニティ", "部屋"]


def detail_rows() -> str:
    pick = [("xlmr_base", "XLM-R base"), ("deberta_ja", "DeBERTa-ja"), ("luke_ja", "LUKE-ja"),
            ("modernbert_ja_310m", "ModernBERT-ja 310M"), ("modernbert_310m_E7E8 (A')", "ModernBERT-ja 310M + E7/E8"),
            ("LLM_swallow8b_few_lenient", "Swallow-8B 少样本"), ("LLM_qwen25_7b_few_lenient", "Qwen2.5-7B 少样本"),
            ("LLM_llmjp31_13b_few_lenient", "llm-jp-13B 少样本"), ("LLM_gptoss20b_few_lenient", "gpt-oss-20B 少样本")]
    out = []
    for key, name in pick:
        d = (S.get(key) or {}).get("detail")
        if not d:
            out.append(f"<tr><td>{name}</td><td colspan=9 class=muted>运行中</td></tr>")
            continue
        cells = "".join(f"<td class=num>{d.get(a, 0):.3f}</td>" for a in ASP)
        out.append(f"<tr><td>{name}</td><td class=num>{d['explicit_neg']:.3f}</td><td class=num>{d['implicit_neg']:.3f}</td>{cells}</tr>")
    return "\n".join(out)


def cross_backbone_rows() -> str:
    a = load("model_comparison/modernbert_310m_E7E8 (A')__vs__modernbert310m_E1.json")
    m = load("ablation_A1_main_E1x5_vs_E8x9.json")
    s = load("repro/stacking_3seed.json")
    rows = [("XLM-R base", "E1 5 次 vs E8 9 次", m), ("DeBERTa-v3-ja", "3 vs 3", s), ("ModernBERT-ja 310M（A'）", "3 vs 3", a)]
    out = []
    for name, n, j in rows:
        if not j:
            out.append(f"<tr><td>{name}</td><td colspan=4 class=muted>运行中</td></tr>")
            continue
        mm = j["metrics"]
        f = lambda k: pp(mm[k]["delta_point"], mm[k]["delta_ci"], mm[k]["significant_at_95"])
        out.append(f"<tr><td>{name}</td><td>{n}</td><td class=num>{f('agg_neg')}</td><td class=num>{f('implicit_neg')}</td><td class=num>{f('agg_pos')}</td></tr>")
    return "\n".join(out)


def aprime_valid() -> str:
    a, e = S.get("modernbert_310m_E7E8 (A')"), S.get("modernbert_ja_310m")
    if not (a and e):
        return "运行中"
    return f"{fmt(e['valid_neg'][0])} → {fmt(a['valid_neg'][0])}（{100 * (a['valid_neg'][0] - e['valid_neg'][0]):+.2f}pp）"


def a5_rows() -> str:
    out = []
    for name, rel, n in (("A5（2024 年池）", "ablation_A5_vs_A0.json", "3 vs 3"),
                         ("A5b（2021–24 年池）", "ablation_A5b_vs_ref.json", "5 vs 5")):
        j = load(rel)
        if not j:
            out.append(f"<tr><td>{name}</td><td colspan=4 class=muted>运行中</td></tr>")
            continue
        mm = j["metrics"]
        f = lambda k: pp(mm[k]["delta_point"], mm[k]["delta_ci"], mm[k]["significant_at_95"])
        out.append(f"<tr><td>{name}</td><td>{n}</td><td class=num>{f('label:朝食_ネガティブ')}</td>"
                   f"<td class=num>{f('label:夕食_ネガティブ')}</td><td class=num>{f('agg_neg')}</td></tr>")
    return "\n".join(out)


done_llm = all(S.get(f"{k}_{m}") for k, _ in LLMS for m in ("zero", "few"))
status = "全部实验已完成" if done_llm else "gpt-oss 等 LLM 设定仍在运行，相应位置标为「运行中」"

page = (HERE / "template.html").read_text(encoding="utf-8")
for k, v in {
    "DATE": date.today().isoformat(), "STATUS": status,
    "ENC_ROWS": encoder_rows(), "LLM_ROWS": llm_rows(), "CHART": chart(), "DETAIL_ROWS": detail_rows(),
    "CROSS_ROWS": cross_backbone_rows(), "APRIME_VALID": aprime_valid(), "A5_ROWS": a5_rows(),
}.items():
    page = page.replace("@@" + k + "@@", v)
(HERE / "report.html").write_text(page, encoding="utf-8")
print("wrote", HERE / "report.html", "|", status)
