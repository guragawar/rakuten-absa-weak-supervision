"""Charts for the 2026-10-15 progress report. Every number is read from review-ai/results/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 13,
    "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8983", "#0b0b0b", "#52514e"
OUT = Path(__file__).resolve().parent
RES = OUT.parent.parent / "review-ai" / "results"


def load(rel):
    p = RES / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


# 1) label-efficiency curve (schedule chosen by E1 valid, see label_efficiency_summary.json)
summ = load("label_efficiency_summary.json")
xs, e1, e8, e1sd, e8sd, deltas = [], [], [], [], [], []
for f in (10, 25, 50, 100):
    if f == 100:
        j = load("followup/L_E8_f100__vs__E1_f100.json")
    else:
        sched = summ["schedule_choice"][str(f)]
        j = load(f"followup/L_L{sched}_E8_f{f}__vs__L{sched}_E1_f{f}.json")
    m = j["metrics"]["agg_neg"]
    xs.append(f)
    e1.append(m["a_mean"]); e8.append(m["b_mean"])
    sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    e1sd.append(sd(m["a_per_seed"])); e8sd.append(sd(m["b_per_seed"]))
    deltas.append(m["delta_point"])

fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=200)
ax.set_xscale("log", base=2)
ax.errorbar(xs, e1, yerr=e1sd, color=GRAY, marker="o", lw=2.2, ms=7, capsize=4, label="事前学習なし（E1）")
ax.errorbar(xs, e8, yerr=e8sd, color=BLUE, marker="o", lw=2.6, ms=7, capsize=4, label="評価点で事前学習（E8）")
ax.axhline(e1[-1], color=GRAY, ls="--", lw=1.2)
ax.text(7.3, e1[-1] + 0.0018, f"事前学習なし・全量 {e1[-1]:.3f}", color=INK2, fontsize=11)
ax.set_xlim(7, 135)
for x, a, b, d in zip(xs, e1, e8, deltas):
    ax.text(x * 1.07, (a + b) / 2, f"+{100 * d:.1f}", color=BLUE, ha="left", va="center", fontsize=12, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([f"{x}%" for x in xs])
ax.set_xlabel("学習に使った Gold データの割合（レビュー単位）")
ax.set_ylabel("テストのネガティブ集約 F1")
ax.set_ylim(0.64, 0.745)
clean(ax)
ax.legend(frameon=False, loc="lower right", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chart_label_efficiency.png")
print("label efficiency:", [f"{x}%: {a:.4f}->{b:.4f}" for x, a, b in zip(xs, e1, e8)])

# 2) gain of E7/E8 per backbone, ordered by the backbone's own (E1) score
BACKBONES = [  # name, bootstrap json, valid direction note
    ("XLM-R base", "ablation_A1_main_E1x5_vs_E8x9.json", ""),
    ("DeBERTa-v3-ja", "repro/stacking_3seed.json", ""),
    ("BERT large（東北大）", "followup/B_bert_large_ja_E7E8_vs_E1.json", "valid 逆"),
    ("ModernBERT-ja 310M", "model_comparison/modernbert_310m_E7E8 (A')__vs__modernbert310m_E1.json", "valid 逆"),
    ("XLM-R large", "followup/B_xlmr_large_E7E8_vs_E1.json", ""),
    ("LUKE-ja", "followup/B_luke_ja_E7E8_vs_E1.json", "valid 逆"),
]
rows = []
for name, rel, note in BACKBONES:
    j = load(rel)
    if not j:
        print("pending:", name)
        continue
    m = j["metrics"]["agg_neg"]
    rows.append((name, m["a_mean"], m["delta_point"], m["delta_ci"], m["significant_at_95"], note))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=200)
for i, (name, base, d, ci, sig, note) in enumerate(rows):
    c = BLUE if sig else GRAY
    ax.barh(i, 100 * d, color=c, height=0.55)
    ax.errorbar(100 * d, i, xerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]], color=INK, capsize=4, lw=1.3)
    lab = f"+{100 * d:.2f}" if d >= 0 else f"{100 * d:.2f}"
    ax.text(100 * ci[1] + 0.12, i, lab + ("" if sig else "（有意差なし）"), va="center", fontsize=11.5, color=INK)
    if note and not note.startswith("@"):
        ax.text(100 * ci[1] + 0.12, i - 0.32, note, va="center", fontsize=10, color=ORANGE)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r[0]}\n事前学習なし {r[1]:.3f}" for r in rows], fontsize=11)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel("事前学習によるネガティブ集約 F1 の改善（ポイント、95% 信頼区間）")
ax.set_xlim(-1.2, 4.6)
clean(ax, grid="x")
fig.tight_layout()
fig.savefig(OUT / "chart_backbones.png")
print("backbones:", [(r[0], round(r[1], 4), round(100 * r[2], 2), r[4]) for r in rows])
