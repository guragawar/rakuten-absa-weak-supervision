"""Charts for the advisor meeting deck. Numbers from review-ai/docs/thesis_numbers.md."""
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
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
OUT = "/home/etl/gujincheng/meetings/2026-10-01/"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


# 1) per-aspect negative F1 delta (E1 -> E8, seed42)
aspects = ["立地", "部屋", "設備", "風呂", "サービス", "朝食", "夕食"]
delta = [6.9, 5.6, 3.8, 0.2, -0.0, -0.4, 0.1]
covered = [True] * 5 + [False] * 2
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
y = list(range(len(aspects)))[::-1]
colors = [BLUE if c else GRAY for c in covered]
ax.barh(y, delta, color=colors, height=0.6, edgecolor="white", linewidth=2)
ax.axvline(0, color="#8a8983", linewidth=1)
for yi, d in zip(y, delta):
    ax.text(d + (0.15 if d >= 0 else -0.15), yi, "±0.0" if abs(d) < 0.05 else f"{d:+.1f}", va="center",
            ha="left" if d >= 0 else "right", color=INK, fontsize=12)
ax.set_yticks(y, aspects)
ax.set_xlim(-1.2, 8)
ax.set_xlabel("ネガティブF1 の改善幅（pp）")
clean(ax)
ax.text(7.9, y[5] + 0.05, "パネル評価は「食事」統合\n→ 朝食/夕食に分けられない", ha="right",
        va="center", color=INK2, fontsize=11)
ax.text(7.9, y[3], "パネルに直接の評価項目あり", ha="right", va="center", color=BLUE, fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "chart_aspect.png")

# 2) data size & year (analysis experiments, single run each)
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
xs, ys = [0.58, 2.0, 6.76], [1.02, 3.08, 2.35]
ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=9,
        markeredgecolor="white", markeredgewidth=2, zorder=3)
for x, v, off in zip(xs, ys, [(0, 11), (-34, 4), (0, 11)]):
    ax.annotate(f"{v:+.2f}", (x, v), textcoords="offset points", xytext=off, ha="center", color=INK, fontsize=12)
ax.plot([2.03], [3.12], "D", color=ORANGE, markersize=9, markeredgecolor="white", markeredgewidth=2, zorder=4)
ax.annotate("近年 2021–24\n+3.12", (2.03, 3.12), textcoords="offset points", xytext=(34, 4), ha="left", va="center", color=INK, fontsize=12)
ax.plot([1.80], [1.55], "s", color=AQUA, markersize=9, markeredgecolor="white", markeredgewidth=2, zorder=4)
ax.annotate("早年 2009–13\n+1.55", (1.80, 1.55), textcoords="offset points", xytext=(34, -4), ha="left", va="center", color=INK, fontsize=12)
ax.text(6.76, 2.02, "全年ランダム抽出", ha="right", color=BLUE, fontsize=11)
ax.set_xscale("log")
ax.set_xticks([0.58, 2.0, 6.76], ["58万", "200万", "676万"])
ax.minorticks_off()
ax.set_ylim(0, 3.9)
ax.set_xlim(0.45, 9)
ax.set_xlabel("事前学習データ量（件、対数軸）")
ax.set_ylabel("ネガF1 改善幅 vs E1（pp）")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT + "chart_scale.png")
print("ok")
