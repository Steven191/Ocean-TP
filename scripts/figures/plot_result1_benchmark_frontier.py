from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "legend.frameon": False,
})


PALETTE = {
    "ocean_tp": "#C23B34",
    "analytical": "#B64A7A",
    "kernel": "#3775BA",
    "ensemble": "#42949E",
    "statistical": "#767676",
    "neural": "#8A6FB5",
    "line": "#D8D8D8",
    "band": "#EAF0F8",
    "text": "#272727",
}


ALIASES = {
    "Ocean-TP optimized Mixed-B framework": "Ocean-TP (optimized Mixed-B)",
    "Mixed-B-only regression fit": "Mixed-B-only fit",
    "Equation-only Mixed-B form without Ocean-TP constraints": "Mixed-B-only fit",
    "GaussianProcess": "Gaussian process",
    "KernelRidge": "Kernel ridge",
    "ExtraTrees": "ExtraTrees",
    "StackingRegressor": "Stacking",
    "SVR": "SVR",
    "HistGradientBoosting": "Hist. gradient boosting",
    "VotingRegressor": "Voting ensemble",
    "KNN": "KNN",
    "NuSVR": "NuSVR",
    "Nystroem RBF + Ridge": "Nystroem RBF + ridge",
    "GradientBoosting": "Gradient boosting",
    "RandomForest": "Random forest",
    "Spline Regression": "Spline regression",
    "AdaBoost": "AdaBoost",
    "MLPRegressor": "MLP regressor",
    "NGBoost": "NGBoost",
    "DecisionTree": "Decision tree",
    "Polynomial Regression": "Polynomial regression",
    "Bagging": "Bagging",
    "RadiusNeighbors": "Radius neighbours",
    "LinearSVR": "Linear SVR",
    "HuberRegressor": "Huber regression",
    "TheilSenRegressor": "Theil-Sen",
    "RANSACRegressor": "RANSAC",
    "Linear Regression": "Linear regression",
    "BayesianRidge": "Bayesian ridge",
    "ElasticNet": "ElasticNet",
    "Lasso": "Lasso",
    "SGDRegressor": "SGD regressor",
    "FourierFeatures+Ridge": "Fourier features + ridge",
}


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "supporting" / "result1_baseline_metrics_10k_cleaned.csv"
OUT_DIR = REPO_ROOT / "results" / "figures" / "supporting"
SOURCE_OUT = REPO_ROOT / "data" / "supporting" / "result1_benchmark_frontier_source_data.csv"


def family_name(model: str) -> str:
    s = str(model).strip()
    if s.startswith("Ocean-TP"):
        return "Ocean-TP optimized Mixed-B framework"
    if s.startswith("Mixed-B-only"):
        return "Mixed-B-only regression fit"
    if s.startswith("Equation-only Mixed-B"):
        return "Mixed-B-only regression fit"
    rules = [
        ("Polynomial Regression", "Polynomial Regression"),
        ("SVR", "SVR"),
        ("Random Forest", "RandomForest"),
        ("RandomForest", "RandomForest"),
        ("Spline Regression", "Spline Regression"),
        ("Nystroem RBF", "Nystroem RBF + Ridge"),
        ("KNN", "KNN"),
        ("Decision Tree", "DecisionTree"),
        ("DecisionTree", "DecisionTree"),
        ("ExtraTrees", "ExtraTrees"),
        ("KernelRidge", "KernelRidge"),
        ("HistGradientBoosting", "HistGradientBoosting"),
        ("GradientBoosting", "GradientBoosting"),
        ("AdaBoost", "AdaBoost"),
        ("Bagging", "Bagging"),
        ("NuSVR", "NuSVR"),
        ("RadiusNeighbors", "RadiusNeighbors"),
        ("GaussianProcess", "GaussianProcess"),
        ("VotingRegressor", "VotingRegressor"),
        ("StackingRegressor", "StackingRegressor"),
        ("MLPRegressor", "MLPRegressor"),
        ("FourierFeatures+Ridge", "FourierFeatures+Ridge"),
        ("NGBoost", "NGBoost"),
        ("RANSACRegressor", "RANSACRegressor"),
        ("TheilSenRegressor", "TheilSenRegressor"),
        ("BayesianRidge", "BayesianRidge"),
        ("ElasticNet", "ElasticNet"),
        ("Lasso", "Lasso"),
        ("LinearSVR", "LinearSVR"),
        ("Linear Regression", "Linear Regression"),
        ("SGDRegressor", "SGDRegressor"),
        ("HuberRegressor", "HuberRegressor"),
    ]
    for prefix, family in rules:
        if s.startswith(prefix):
            return family
    return re.sub(r"\s*\(.*?\)\s*", "", s).strip()


def family_group(family: str) -> str:
    if family.startswith("Ocean-TP"):
        return "ocean_tp"
    if family.startswith("Mixed-B-only") or family.startswith("Equation-only Mixed-B"):
        return "analytical"
    if family in {
        "GaussianProcess", "KernelRidge", "SVR", "NuSVR",
        "Nystroem RBF + Ridge", "KNN", "RadiusNeighbors",
    }:
        return "kernel"
    if family in {
        "ExtraTrees", "StackingRegressor", "HistGradientBoosting",
        "VotingRegressor", "GradientBoosting", "RandomForest",
        "AdaBoost", "NGBoost", "DecisionTree", "Bagging",
    }:
        return "ensemble"
    if family in {"MLPRegressor", "FourierFeatures+Ridge"}:
        return "neural"
    return "statistical"


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["mae"] <= 1].copy()
    df["family"] = df["model"].map(family_name)
    rows = []
    for family, sub in df.groupby("family", sort=False):
        best = sub.loc[sub["mae"].idxmin()].copy()
        best["n_configs"] = len(sub)
        best["group"] = family_group(family)
        best["short_label"] = ALIASES.get(family, family)
        rows.append(best)
    out = pd.DataFrame(rows)
    ocean_tp = pd.DataFrame([{
        "train_count": np.nan,
        "test_count": np.nan,
        "category": "Ocean-TP",
        "model": "Ocean-TP optimized Mixed-B framework",
        "mae": 0.05107,
        "rmse": 0.11189,
        "r2": 0.98407,
        "family": "Ocean-TP optimized Mixed-B framework",
        "n_configs": 1,
        "group": "ocean_tp",
        "short_label": ALIASES["Ocean-TP optimized Mixed-B framework"],
    }])
    out = pd.concat([ocean_tp, out], ignore_index=True)
    out = out.sort_values("mae", ascending=True).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def plot_panel(ax, data, xmin, xlim, title, log_scale=False, show_values=False):
    y = np.arange(len(data))
    colors = [PALETTE[g] for g in data["group"]]
    markers = ["*" if g == "ocean_tp" else "D" if g == "analytical" else "o" for g in data["group"]]

    ax.hlines(y, xmin=xmin, xmax=data["mae"], color=PALETTE["line"], lw=0.9, zorder=1)
    for yi, x, color, marker, group in zip(y, data["mae"], colors, markers, data["group"]):
        size = 64 if group == "ocean_tp" else 36 if group == "analytical" else 22
        edge = PALETTE["text"] if group in {"ocean_tp", "analytical"} else "white"
        ax.scatter(x, yi, s=size, marker=marker, color=color, edgecolor=edge, lw=0.55, zorder=3)
        if show_values and group in {"ocean_tp", "analytical"}:
            ax.text(x + 0.00010, yi, f"{x:.5f}", va="center", ha="left",
                    fontsize=6.2, color=color, fontweight="bold")

    ocean_tp = data.loc[data["group"].eq("ocean_tp"), "mae"]
    if len(ocean_tp):
        ax.axvline(float(ocean_tp.iloc[0]), color=PALETTE["ocean_tp"],
                   lw=0.85, ls=(0, (1.2, 2.2)), alpha=0.85, zorder=0)
    mixb = data.loc[data["group"].eq("analytical"), "mae"]
    if len(mixb):
        ax.axvline(float(mixb.iloc[0]), color=PALETTE["analytical"],
                   lw=0.8, ls=(0, (2.5, 2.5)), alpha=0.75, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(data["short_label"], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    if log_scale:
        ax.set_xscale("log")
        ticks = [0.05, 0.1, 0.3, 0.7]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2f}" if t < 0.1 else f"{t:.1f}" for t in ticks])
    else:
        ticks = [0.051, 0.053, 0.055, 0.057, 0.059]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.3f}" for t in ticks])
        ax.axvspan(0.051, 0.0572, color=PALETTE["band"], zorder=-2)
    ax.set_xlabel("MAE (deg C), lower is better", labelpad=3)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=5)
    ax.tick_params(axis="x", labelsize=6.2, length=2.5, width=0.7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", color="#ECECEC", lw=0.5, zorder=-3)


def main():
    data = prepare_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(SOURCE_OUT, index=False, encoding="utf-8-sig")

    full = data.copy()
    frontier = data[data["mae"] <= 0.0605].copy()

    fig = plt.figure(figsize=(7.35, 6.55))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.35, 1.0],
        left=0.20, right=0.985, bottom=0.13, top=0.84, wspace=0.72
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_panel(
        ax_a, full, xmin=0.048, xlim=(0.048, 0.82),
        title="a  Best model in each algorithmic family", log_scale=True
    )
    plot_panel(
        ax_b, frontier, xmin=0.0505, xlim=(0.0505, 0.0607),
        title="b  Leading benchmark frontier", log_scale=False, show_values=True
    )

    fig.suptitle(
        "Ocean-TP separates optimized Mixed-B reconstruction from equation-only control",
        x=0.20, y=0.968, ha="left", fontsize=9.2, fontweight="bold"
    )
    fig.text(
        0.20, 0.928,
        "Best family-level configurations from 298 valid benchmarks; star marks full Ocean-TP and diamond marks the Mixed-B-only regression control.",
        ha="left", va="top", fontsize=6.5, color=PALETTE["text"]
    )

    handles = [
        Line2D([0], [0], marker="*", color="none", label="Ocean-TP (optimized Mixed-B)",
               markerfacecolor=PALETTE["ocean_tp"], markeredgecolor=PALETTE["text"], markersize=7.2),
        Line2D([0], [0], marker="D", color="none", label="Mixed-B-only regression fit",
               markerfacecolor=PALETTE["analytical"], markeredgecolor=PALETTE["text"], markersize=5.6),
        Line2D([0], [0], marker="o", color="none", label="Kernel / neighbour",
               markerfacecolor=PALETTE["kernel"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="o", color="none", label="Tree / ensemble",
               markerfacecolor=PALETTE["ensemble"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="o", color="none", label="Statistical / robust",
               markerfacecolor=PALETTE["statistical"], markeredgecolor="white", markersize=5.2),
        Line2D([0], [0], marker="o", color="none", label="Neural / hybrid",
               markerfacecolor=PALETTE["neural"], markeredgecolor="white", markersize=5.2),
    ]
    fig.legend(
        handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0.20, 0.025),
        fontsize=6.2, handletextpad=0.45, columnspacing=1.2
    )

    base = OUT_DIR / "benchmark_frontier"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(base.with_suffix(".png"))
    print(base.with_suffix(".pdf"))
    print(base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
