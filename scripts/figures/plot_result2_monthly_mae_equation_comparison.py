from pathlib import Path
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "supporting" / "result2_monthly_seasonal_annual_mae_comparison.csv"
OUT_DIR = REPO_ROOT / "results" / "figures" / "supporting"
OUT_BASE = OUT_DIR / "result2_monthly_mae_equation_comparison"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COLORS = {
    "monthly": "#8066B3",
    "seasonal": "#3F6FAE",
    "annual": "#63AFC7",
}


def load_data():
    rows = []
    with DATA.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "month": row["month"],
                "monthly": float(row["monthly_model_mae"]),
                "seasonal": float(row["seasonal_model_mae"]),
                "annual": float(row["annual_model_mae"]),
            })
    return rows


def main():
    rows = load_data()
    months = np.arange(1, 13)
    monthly = np.array([r["monthly"] for r in rows])
    seasonal = np.array([r["seasonal"] for r in rows])
    annual = np.array([r["annual"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.6, 3.05), constrained_layout=True)

    ax.axvspan(5.55, 6.45, color="#BFE8F4", alpha=0.28, zorder=-5)
    ax.axvspan(6.55, 8.45, color="#D8C8EC", alpha=0.22, zorder=-5)

    # Small graphical dodge prevents near-identical points from hiding each other.
    ax.plot(months + 0.045, monthly, color=COLORS["monthly"], lw=1.25, marker="o",
            ms=3.2, mec="white", mew=0.35, label="Monthly equation", zorder=4)
    ax.plot(months, seasonal, color=COLORS["seasonal"], lw=1.05, ls=(0, (3.0, 1.8)),
            marker="s", ms=2.9, mec="white", mew=0.30, label="Seasonal equation", zorder=3)
    ax.plot(months - 0.045, annual, color=COLORS["annual"], lw=0.95, ls=(0, (1.0, 1.8)),
            marker="^", ms=3.0, mec="white", mew=0.30, label="Annual equation", zorder=2)

    ax.text(6, 0.0410, "best", ha="center", va="bottom", fontsize=6.0, color="#2B7F97")
    ax.text(7.5, 0.0692, "summer peak", ha="center", va="top", fontsize=6.0, color="#7A5DA6")

    ax.axhline(0.05, color="#767676", lw=0.65, ls=(0, (3, 2)), alpha=0.45, zorder=0)
    ax.grid(axis="y", color="#DADADA", lw=0.34, alpha=0.45, zorder=-1)

    ax.set_xlim(0.55, 12.45)
    ax.set_ylim(0.040, 0.071)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_yticks([0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070])
    ax.set_ylabel("MAE (degrees C)")
    ax.set_xlabel("Month")
    ax.set_title("Monthly MAE comparison across annual, seasonal and monthly equations",
                 fontsize=8.7, pad=6)
    ax.tick_params(axis="both", labelsize=7.0, width=0.65, length=3, colors="#4A4A4A")
    ax.spines["left"].set_color("#8A8A8A")
    ax.spines["bottom"].set_color("#8A8A8A")

    ax.legend(loc="upper right", ncol=3, fontsize=6.4, handlelength=1.7,
              columnspacing=0.95, handletextpad=0.45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)

    print(OUT_BASE.with_suffix(".svg"))
    print(OUT_BASE.with_suffix(".pdf"))
    print(OUT_BASE.with_suffix(".png"))


if __name__ == "__main__":
    main()
