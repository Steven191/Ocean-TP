from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.2,
    "axes.linewidth": 0.72,
    "axes.spines.top": False,
    "legend.frameon": False,
})


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "supporting" / "result2_annual_error_distribution_r2_stats.csv"
OUT_BASE = REPO_ROOT / "results" / "figures" / "supporting" / "FIG3_annual_metrics_combo"
OUT_BASE.parent.mkdir(parents=True, exist_ok=True)


def load_annual() -> list[dict]:
    df = pd.read_csv(DATA)
    df = df[df["year"].astype(str).str.fullmatch(r"\d{4}")].copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")
    annual = []
    for row in df.itertuples(index=False):
        annual.append(
            {
                "year": int(row.year),
                "mae": float(row.mae),
                "rmse": float(row.rmse),
                "r2": float(row.r2),
                "whislo": float(row.lower_whisker),
                "q1": float(row.q1),
                "median": float(row.median),
                "q3": float(row.q3),
                "whishi": float(row.upper_whisker),
            }
        )
    return annual


def add_boxplot(ax, annual):
    years = [row["year"] for row in annual]
    x = np.arange(1, len(annual) + 1)
    colors = ["#BEEAF4", "#B7DFF1", "#B8D7F0", "#C7D6F2", "#C9C4F4",
              "#C2B8F2", "#CBB5F1", "#D4B2EF", "#E1B7EC", "#EABBE8"]
    edge_colors = ["#5EC6E0", "#63B8DB", "#70A9D6", "#849ED4", "#8F88D8",
                   "#907ED9", "#A575DF", "#B86DE3", "#C966E2", "#D55FD6"]
    box_stats = [
        {
            "label": str(row["year"]),
            "whislo": row["whislo"],
            "q1": row["q1"],
            "med": row["median"],
            "mean": row["mae"],
            "q3": row["q3"],
            "whishi": row["whishi"],
            "fliers": [],
        }
        for row in annual
    ]
    bp = ax.bxp(
        box_stats,
        positions=x,
        widths=0.58,
        showmeans=True,
        showfliers=False,
        patch_artist=True,
        boxprops={"linewidth": 0.85},
        medianprops={"color": "#6E6E6E", "linewidth": 1.45},
        meanprops={
            "marker": "D",
            "markersize": 3.9,
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "linestyle": "none",
        },
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )
    for i, (face, edge) in enumerate(zip(colors, edge_colors)):
        bp["boxes"][i].set(facecolor=face, edgecolor="none", alpha=1.0)
        bp["means"][i].set(markeredgecolor=edge)
        for item in (bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1], bp["caps"][2 * i], bp["caps"][2 * i + 1]):
            item.set(color=edge, linewidth=0.9)

    ax.axhline(0, color="#8A8A8A", lw=0.58, ls=(0, (2.5, 2.2)), alpha=0.55)
    ax.grid(axis="y", color="#E2E2E2", lw=0.35, alpha=0.48)
    ax.set_xlim(0.35, len(annual) + 0.65)
    ax.set_ylim(-0.385, 0.425)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=6.4)
    ax.set_xlabel("Year", labelpad=2)
    ax.set_ylabel("Validation error", labelpad=3)
    ax.set_title("(a) Annual validation-error distribution (2015-2024)", fontsize=8.0, pad=5)
    ax.tick_params(axis="both", width=0.55, length=2.5, labelsize=6.5, colors="#4B4B4B")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6E6E6E", lw=1.45, label="Median"),
            Line2D([0], [0], marker="D", color="#6EAAD0", markerfacecolor="white", markeredgewidth=1.0, linestyle="none", label="Mean"),
        ],
        loc="upper right",
        fontsize=5.8,
        handlelength=1.2,
        borderpad=0.2,
        labelspacing=0.3,
    )


def add_metric_combo(ax, annual):
    years = np.array([row["year"] for row in annual])
    x = np.arange(len(years))
    mae = np.array([row["mae"] for row in annual])
    rmse = np.array([row["rmse"] for row in annual])
    r2 = np.array([row["r2"] for row in annual])

    mae_color = "#006DAA"
    rmse_color = "#5FA8D3"
    r2_color = "#BFD9EC"
    r2_edge = "#86B6D6"
    ax2 = ax.twinx()
    ax2.bar(
        x,
        r2 - 0.972,
        bottom=0.972,
        width=0.56,
        color=r2_color,
        edgecolor=r2_edge,
        linewidth=0.38,
        alpha=1.0,
        label=r"$R^2$",
        zorder=1,
    )
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(
        x,
        mae,
        color=mae_color,
        lw=1.35,
        marker="o",
        ms=3.5,
        mfc="white",
        mec=mae_color,
        mew=0.85,
        label="MAE",
        zorder=5,
    )
    ax.plot(
        x,
        rmse,
        color=rmse_color,
        lw=1.35,
        marker="s",
        ms=3.2,
        mfc="white",
        mec=rmse_color,
        mew=0.80,
        label="RMSE",
        zorder=5,
    )
    ax.axhline(mae.mean(), color=mae_color, lw=0.62, ls=(0, (2.2, 2.0)), alpha=0.38, zorder=2)
    ax.axhline(rmse.mean(), color=rmse_color, lw=0.62, ls=(0, (2.2, 2.0)), alpha=0.42, zorder=2)

    for idx, year in enumerate(years):
        if year in (2018, 2021, 2024):
            ax2.text(x[idx], r2[idx] + 0.0008, f"{r2[idx]:.3f}", ha="center", va="bottom", fontsize=5.3, color="#4F83A6")

    ax.set_ylim(0, 0.145)
    ax2.set_ylim(0.972, 0.995)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=6.0)
    ax.set_xlabel("Year", labelpad=1)
    ax.set_ylabel("Error (°C)", color="#355B74", labelpad=2)
    ax2.set_ylabel(r"$R^2$", color="#4F83A6", labelpad=3)
    ax.set_title("(b) Annual MAE, RMSE and $R^2$", fontsize=8.0, pad=5)
    ax.grid(axis="y", color="#E2E2E2", lw=0.35, alpha=0.50, zorder=0)
    ax.tick_params(axis="both", width=0.55, length=2.5, labelsize=6.2, colors="#4B4B4B")
    ax2.tick_params(axis="y", width=0.55, length=2.5, labelsize=6.2, colors="#4F83A6")
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#86B6D6")
    ax2.spines["right"].set_linewidth(0.65)

    handles = [
        Patch(facecolor=r2_color, edgecolor=r2_edge, linewidth=0.35, label=r"$R^2$"),
        Line2D([0], [0], color=mae_color, marker="o", mfc="white", mec=mae_color, lw=1.25, ms=3.2, label="MAE"),
        Line2D([0], [0], color=rmse_color, marker="s", mfc="white", mec=rmse_color, lw=1.25, ms=3.0, label="RMSE"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=5.8, ncol=3, handlelength=1.1, columnspacing=0.65, borderpad=0.2)


def main():
    annual = load_annual()
    fig = plt.figure(figsize=(8.1, 3.05), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.18], wspace=0.08)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_combo = fig.add_subplot(gs[0, 1])
    add_boxplot(ax_box, annual)
    add_metric_combo(ax_combo, annual)

    for path in [
        OUT_BASE.with_suffix(".pdf"),
        OUT_BASE.with_suffix(".png"),
        OUT_BASE.with_suffix(".svg"),
    ]:
        if path.suffix == ".png":
            fig.savefig(path, dpi=500, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
