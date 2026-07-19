from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "supporting" / "result2_fixed_equation_daily_stability_2023_2024.csv"
OUT_DIR = REPO_ROOT / "results" / "figures" / "supporting"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SOURCE = REPO_ROOT / "data" / "supporting" / "result2_fixed_equation_daily_stability_2023_2024_plot_source.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
    }
)


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["mae"] = df["mae"].astype(float)
    df["rmse"] = df["rmse"].astype(float)

    parts = []
    for year, sub in df.groupby("year", sort=True):
        sub = sub.sort_values("date").copy()
        sub["mae_roll7"] = sub["mae"].rolling(window=7, center=True, min_periods=3).mean()
        sub["rmse_roll7"] = sub["rmse"].rolling(window=7, center=True, min_periods=3).mean()
        parts.append(sub)
    plot_df = pd.concat(parts, ignore_index=True)
    plot_df[
        ["date", "year", "doy", "mae", "rmse", "mae_roll7", "rmse_roll7", "stable_day", "data_status"]
    ].to_csv(PLOT_SOURCE, index=False)

    colors = {
        2023: "#3f7f93",
        2024: "#c05a4d",
    }

    month_starts = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    month_ticks = [d.dayofyear for d in month_starts]
    month_labels = [d.strftime("%b") for d in month_starts]

    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    ax.set_facecolor("white")

    ax.axvspan(
        pd.Timestamp("2024-07-01").dayofyear,
        pd.Timestamp("2024-10-31").dayofyear,
        color="#d9c7ba",
        alpha=0.14,
        lw=0,
        zorder=0,
    )
    ax.axhline(0.10, color="#7b7b7b", lw=0.9, ls=(0, (4, 3)), zorder=1)

    for year in [2023, 2024]:
        sub = plot_df[plot_df["year"] == year].sort_values("doy")
        ax.plot(sub["doy"], sub["mae"], color=colors[year], lw=0.55, alpha=0.42, zorder=2)
        ax.plot(
            sub["doy"],
            sub["mae_roll7"],
            color=colors[year],
            lw=1.12,
            alpha=0.98,
            label=f"{year} daily MAE, 7-d mean",
            zorder=3,
        )

    ax.set_xlim(1, 366)
    ax.set_ylim(0.018, 0.108)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_yticks([0.025, 0.05, 0.075, 0.10])
    ax.set_ylabel("Daily MAE (\N{DEGREE SIGN}C)")
    ax.set_xlabel("Month, aligned by day of year")
    ax.grid(axis="y", color="#d8dde1", lw=0.45, alpha=0.75)
    ax.tick_params(axis="both", labelsize=7)

    title = "Fixed-equation daily prediction stability, 2023-2024"
    ax.set_title(title, loc="left", fontsize=9.6, weight="bold", pad=9)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(4, 0.0875),
        bbox_transform=ax.transData,
        fontsize=7.1,
        handlelength=2.8,
    )

    fig.tight_layout(pad=1.1)
    save_pub(fig, OUT_DIR / "fixed_equation_daily_mae_line_2023_2024")
    plt.close(fig)


if __name__ == "__main__":
    main()
