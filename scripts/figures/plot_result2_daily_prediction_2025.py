from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "supporting" / "result2_daily_prediction_2025_corrected.csv"
OUT = REPO_ROOT / "results" / "figures" / "supporting" / "FIG8"
OUT.parent.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7.2,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
})

df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["mae7"] = df["mae"].rolling(7, center=True, min_periods=1).mean()

best_start = pd.Timestamp("2025-01-03")
best_end = pd.Timestamp("2025-01-09")
max_idx = df["mae"].idxmax()
max_date = df.loc[max_idx, "date"]
max_mae = df.loc[max_idx, "mae"]

blue = "#0E5AA7"
blue_light = "#8EB4CF"
red = "#C73E3A"
grey_bar = "#D6D9DC"
grey_text = "#4F5963"
axis_grey = "#B7BEC7"

fig, ax = plt.subplots(figsize=(183 / 25.4, 82 / 25.4), dpi=300)
ax2 = ax.twinx()

ax.set_zorder(2)
ax2.set_zorder(1)
ax.patch.set_alpha(0)

bar_base = 8300
ax2.bar(
    df["date"],
    df["sample_count"] - bar_base,
    bottom=bar_base,
    width=0.82,
    color=grey_bar,
    edgecolor=grey_bar,
    linewidth=0.25,
    zorder=0,
)

ax.plot(
    df["date"],
    df["mae"],
    color=blue_light,
    lw=0.72,
    marker="o",
    markersize=1.6,
    markerfacecolor=blue_light,
    markeredgewidth=0,
    zorder=3,
)
ax.plot(df["date"], df["mae7"], color=blue, lw=1.25, zorder=4)
ax.scatter([max_date], [max_mae], s=18, color=red, edgecolor="white", linewidth=0.45, zorder=5)

bracket_y = 0.0042
ax.plot([best_start, best_end], [bracket_y, bracket_y], color=blue, lw=1.15, zorder=5)
for x in [best_start, best_end]:
    ax.plot([x, x], [bracket_y - 0.0018, bracket_y + 0.0018], color=blue, lw=1.15, zorder=5)

ax.set_xlim(df["date"].min() - pd.Timedelta(days=1), df["date"].max() + pd.Timedelta(days=1))
ax.set_ylim(-0.006, 0.118)
ax2.set_ylim(8200, 9900)
ax2.set_yticks([8300, 9000, 9700])

ax.set_ylabel(r"Daily MAE ($^\circ$C)", color="#1E2328")
ax2.set_ylabel("Sample count", rotation=270, labelpad=14, color="#1E2328")
ax.set_xlabel("Date in 2025")

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis="x", which="minor", length=2.0, width=0.45, color="#8C949D")
ax.tick_params(axis="x", which="major", length=0, pad=5)
ax.tick_params(axis="y", width=0.65, color=axis_grey)
ax2.tick_params(axis="y", width=0.65, color=axis_grey)

ax.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax.spines["left"].set_color("#343A40")
ax.spines["bottom"].set_color("#343A40")
ax2.spines["right"].set_color("#343A40")

handles = [
    Line2D([0], [0], color=blue_light, lw=0.9, marker="o", markersize=2.0, markerfacecolor=blue_light, markeredgewidth=0),
    Line2D([0], [0], color=blue, lw=1.4),
    Line2D([0], [0], color=red, lw=0, marker="o", markersize=4.0),
    Line2D([0], [0], color=grey_bar, lw=4.0),
    Line2D([0], [0], color=blue, lw=1.2),
]
labels = [
    "Daily MAE",
    "7-day mean",
    "Max daily MAE",
    "Sample count",
    "Jan 03-09 lowest 7-d",
]
leg = ax.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(0.035, 0.935),
    ncol=3,
    fontsize=5.9,
    handlelength=2.8,
    columnspacing=1.4,
    borderpad=0.55,
    labelspacing=0.45,
    frameon=True,
)
leg.get_frame().set_facecolor((1, 1, 1, 0.74))
leg.get_frame().set_edgecolor("#E1E6EB")
leg.get_frame().set_linewidth(0.45)

fig.text(
    0.075,
    0.965,
    "Daily profile-prediction error and sampling density (2025)",
    ha="left",
    va="top",
    fontsize=8.8,
    fontweight="bold",
    color="#1E2328",
)
fig.text(
    0.075,
    0.915,
    "Fixed 2024 parameters retain a low-error January window before the late-winter to spring transition",
    ha="left",
    va="top",
    fontsize=7.1,
    color=grey_text,
)

fig.subplots_adjust(left=0.075, right=0.925, bottom=0.20, top=0.86)
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.svg", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight")
print(f"{OUT}.pdf")
