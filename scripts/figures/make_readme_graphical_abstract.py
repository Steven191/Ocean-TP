"""Create the README graphical abstract from repository support records."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "supporting"
OUT_DIR = REPO_ROOT / "docs" / "assets"
OUT = OUT_DIR / "ocean_tp_graphical_abstract"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


PALETTE = {
    "paper": "#F6F3EC",
    "panel": "#FFFFFF",
    "ink": "#17242E",
    "muted": "#66727C",
    "line": "#D7DDD9",
    "grid": "#E7E5DE",
    "ocean": "#0B5E78",
    "ocean_dark": "#093044",
    "teal": "#2E9C91",
    "blue": "#3A76AF",
    "coral": "#C8493E",
    "amber": "#D9A73D",
    "rose": "#A95275",
    "soft_teal": "#DCEDEA",
    "soft_blue": "#DDE8F2",
    "soft_coral": "#F2DED9",
    "soft_amber": "#F3E7C8",
}


GROUP_COLORS = {
    "ocean_tp": PALETTE["coral"],
    "analytical": PALETTE["rose"],
    "kernel": PALETTE["blue"],
    "ensemble": PALETTE["teal"],
    "statistical": "#878787",
    "neural": "#7F72B6",
}


LABEL_ALIASES = {
    "Ocean-TP optimized Mixed-B framework": "Ocean-TP",
    "Mixed-B-only regression fit": "Mixed-B-only",
    "GaussianProcess": "Gaussian process",
    "KernelRidge": "Kernel ridge",
    "ExtraTrees": "ExtraTrees",
    "StackingRegressor": "Stacking",
    "SVR": "SVR",
    "HistGradientBoosting": "Hist. gradient boosting",
}


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str = PALETTE["panel"],
    edgecolor: str = PALETTE["line"],
    linewidth: float = 1.0,
    radius: float = 0.02,
    alpha: float = 1.0,
    zorder: int = 2,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = PALETTE["ocean_dark"],
    rad: float = 0.0,
    lw: float = 1.5,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            transform=ax.transAxes,
            zorder=zorder,
        )
    )


def wrapped_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    width: int,
    *,
    size: float,
    color: str,
    weight: str = "normal",
    ha: str = "left",
    va: str = "top",
) -> None:
    ax.text(
        x,
        y,
        "\n".join(textwrap.wrap(text, width=width)),
        transform=ax.transAxes,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=1.22,
    )


def load_data() -> dict[str, object]:
    benchmark = pd.read_csv(DATA_DIR / "result1_benchmark_frontier_source_data.csv")
    benchmark["mae"] = pd.to_numeric(benchmark["mae"])
    benchmark["rmse"] = pd.to_numeric(benchmark["rmse"])
    benchmark["r2"] = pd.to_numeric(benchmark["r2"])

    ocean = benchmark.loc[
        benchmark["model"].eq("Ocean-TP optimized Mixed-B framework")
    ].iloc[0]
    mixed_b = benchmark.loc[benchmark["model"].eq("Mixed-B-only regression fit")].iloc[0]

    stability = pd.read_csv(
        DATA_DIR / "result2_fixed_equation_daily_stability_2023_2024_summary.csv"
    )
    year_stability = stability.loc[stability["level"].eq("year")].copy()

    daily = pd.read_csv(DATA_DIR / "result2_daily_prediction_2025_corrected.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["mae7"] = daily["mae"].rolling(7, center=True, min_periods=1).mean()

    depth = pd.read_csv(DATA_DIR / "result2_depth_accuracy_1m.csv")

    return {
        "benchmark": benchmark,
        "ocean": ocean,
        "mixed_b": mixed_b,
        "stability": year_stability,
        "daily": daily,
        "depth": depth,
    }


def clean_svg(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    path.write_text(cleaned, encoding="utf-8")


def draw_background(ax: plt.Axes) -> None:
    for i, y0 in enumerate(np.linspace(0.10, 0.88, 16)):
        xs = np.linspace(0.0, 1.0, 420)
        phase = i * 0.55
        ys = y0 + 0.007 * np.sin(xs * 14 + phase) + 0.004 * np.sin(xs * 29 - phase)
        ax.plot(
            xs,
            ys,
            color=PALETTE["line"],
            lw=0.55,
            alpha=0.35,
            transform=ax.transAxes,
            zorder=0,
        )

    ax.add_patch(
        Circle(
            (0.18, 0.82),
            0.23,
            transform=ax.transAxes,
            facecolor=PALETTE["soft_blue"],
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )
    ax.add_patch(
        Circle(
            (0.92, 0.12),
            0.22,
            transform=ax.transAxes,
            facecolor=PALETTE["soft_teal"],
            edgecolor="none",
            alpha=0.45,
            zorder=0,
        )
    )


def draw_method_panel(ax: plt.Axes) -> None:
    rounded_box(
        ax,
        0.035,
        0.145,
        0.575,
        0.71,
        facecolor="#FBFAF5",
        edgecolor="#D8D7CF",
        linewidth=1.0,
        radius=0.025,
        alpha=0.98,
    )
    ax.text(
        0.065,
        0.805,
        "Temperature-pressure profile reconstruction",
        transform=ax.transAxes,
        fontsize=15.5,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="top",
    )
    wrapped_text(
        ax,
        0.066,
        0.765,
        "Ocean-TP combines an analytical Mixed-B pressure basis with optimization constraints to reconstruct stable ocean T-P profiles from OOI records.",
        74,
        size=8.1,
        color=PALETTE["muted"],
    )

    # Input observation card.
    rounded_box(
        ax,
        0.068,
        0.300,
        0.145,
        0.350,
        facecolor=PALETTE["panel"],
        edgecolor="#CED6D3",
        radius=0.018,
    )
    ax.text(
        0.088,
        0.620,
        "OOI T-P\nobservations",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    x0, y0, w, h = 0.089, 0.335, 0.100, 0.225
    ax.plot([x0, x0], [y0, y0 + h], color=PALETTE["ocean_dark"], lw=0.85, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0, y0], color=PALETTE["ocean_dark"], lw=0.85, transform=ax.transAxes)
    depth_t = np.linspace(0, 1, 72)
    px = x0 + 0.018 + 0.062 * np.exp(-1.65 * depth_t) + 0.010 * np.sin(9.0 * depth_t)
    py = y0 + h * (1 - depth_t)
    ax.plot(px, py, color=PALETTE["teal"], lw=2.1, transform=ax.transAxes, zorder=4)
    ax.scatter(
        px[::7],
        py[::7],
        s=15,
        color=PALETTE["coral"],
        edgecolor="white",
        linewidth=0.45,
        transform=ax.transAxes,
        zorder=5,
    )
    ax.text(x0 + w - 0.003, y0 - 0.025, "T", transform=ax.transAxes, fontsize=7.2, color=PALETTE["muted"])
    ax.text(x0 - 0.025, y0 + h + 0.002, "P", transform=ax.transAxes, fontsize=7.2, color=PALETTE["muted"])

    # Analytical backbone.
    rounded_box(
        ax,
        0.270,
        0.335,
        0.185,
        0.285,
        facecolor=PALETTE["ocean_dark"],
        edgecolor=PALETTE["ocean_dark"],
        radius=0.020,
        linewidth=0.0,
    )
    ax.text(
        0.293,
        0.585,
        "Mixed-B analytical\nbackbone",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color="white",
        va="top",
    )
    formula = "T(P) = b0 + b1 P + b2 ln P + b3 sqrt(P) + ..."
    ax.text(
        0.292,
        0.522,
        formula,
        transform=ax.transAxes,
        fontsize=6.7,
        color="#D8F0F0",
        family="monospace",
        va="top",
    )
    tokens = ["P", "ln P", "sqrt(P)", "exp(-P/500)", "exp(-P/1200)", "(P+50)^-1"]
    tx, ty = 0.292, 0.472
    for i, token in enumerate(tokens):
        col = i % 2
        row = i // 2
        rounded_box(
            ax,
            tx + col * 0.081,
            ty - row * 0.044,
            0.070,
            0.026,
            facecolor="#164B62",
            edgecolor="#2E6F84",
            radius=0.009,
            linewidth=0.55,
            zorder=4,
        )
        ax.text(
            tx + col * 0.081 + 0.035,
            ty - row * 0.044 + 0.013,
            token,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.6,
            color="#EAF7F4",
            zorder=5,
        )

    # Ocean-TP optimization.
    rounded_box(
        ax,
        0.290,
        0.195,
        0.145,
        0.095,
        facecolor=PALETTE["soft_coral"],
        edgecolor="#DDA59C",
        radius=0.018,
    )
    ax.text(
        0.362,
        0.254,
        "Ocean-TP\noptimized constraints",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="center",
        va="center",
    )
    for cx, cy, fc in [
        (0.310, 0.222, PALETTE["coral"]),
        (0.332, 0.262, PALETTE["amber"]),
        (0.415, 0.222, PALETTE["teal"]),
    ]:
        ax.add_patch(
            Circle(
                (cx, cy),
                0.0085,
                transform=ax.transAxes,
                facecolor=fc,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
        )

    # Output card.
    rounded_box(
        ax,
        0.500,
        0.300,
        0.080,
        0.350,
        facecolor=PALETTE["panel"],
        edgecolor="#CED6D3",
        radius=0.018,
    )
    ax.text(
        0.540,
        0.620,
        "Stable\nprofile",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="center",
        va="top",
    )
    yy = np.linspace(0.345, 0.560, 90)
    zz = np.linspace(0, 1, len(yy))
    p1 = 0.527 + 0.028 * np.exp(-1.7 * zz) - 0.008 * np.sin(7 * zz)
    p2 = p1 + 0.010 + 0.002 * np.cos(8 * zz)
    ax.plot(p1, yy, color=PALETTE["coral"], lw=2.1, transform=ax.transAxes, zorder=5)
    ax.plot(p2, yy, color=PALETTE["teal"], lw=1.9, transform=ax.transAxes, zorder=5)
    for ytick in np.linspace(0.36, 0.55, 5):
        ax.plot([0.516, 0.568], [ytick, ytick], color=PALETTE["grid"], lw=0.55, transform=ax.transAxes, zorder=3)
    ax.text(0.541, 0.327, "T, P", transform=ax.transAxes, fontsize=6.8, color=PALETTE["muted"], ha="center")

    arrow(ax, (0.218, 0.475), (0.265, 0.475), color=PALETTE["ocean_dark"])
    arrow(ax, (0.362, 0.333), (0.362, 0.292), color=PALETTE["coral"], lw=1.35)
    arrow(ax, (0.456, 0.475), (0.496, 0.475), color=PALETTE["ocean_dark"])


def draw_benchmark_panel(fig: plt.Figure, data: dict[str, object]) -> None:
    ax = fig.add_axes([0.665, 0.555, 0.265, 0.155])
    benchmark = data["benchmark"].sort_values("mae").head(8).copy()
    labels = [LABEL_ALIASES.get(v, LABEL_ALIASES.get(str(v).split()[0], str(v))) for v in benchmark["family"]]
    labels = [label if len(label) <= 24 else label[:22] + "." for label in labels]
    y = np.arange(len(benchmark))
    colors = [GROUP_COLORS.get(g, "#8A8A8A") for g in benchmark["group"]]

    ax.hlines(y, benchmark["mae"].min() - 0.00045, benchmark["mae"], color=PALETTE["line"], lw=1.0)
    ax.scatter(benchmark["mae"], y, s=34, color=colors, edgecolor="white", linewidth=0.55, zorder=3)
    ax.scatter(
        benchmark["mae"].iloc[0],
        y[0],
        s=72,
        color=PALETTE["coral"],
        marker="*",
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlim(0.0506, 0.0570)
    ax.set_xticks([0.051, 0.053, 0.055, 0.057])
    ax.set_xticklabels(["0.051", "0.053", "0.055", "0.057"], fontsize=6.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, width=0.6)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.6)
    ax.set_xlabel("MAE (deg C), lower is better", fontsize=6.3, labelpad=2)
    ax.set_title("Benchmark frontier", loc="left", fontsize=8.7, fontweight="bold", color=PALETTE["ink"], pad=4)
    ax.set_facecolor("none")


def draw_daily_panel(fig: plt.Figure, data: dict[str, object]) -> None:
    ax = fig.add_axes([0.665, 0.295, 0.265, 0.145])
    daily = data["daily"]
    ax.plot(daily["date"], daily["mae"], color="#84AFC4", lw=0.75, marker="o", markersize=1.4, markeredgewidth=0)
    ax.plot(daily["date"], daily["mae7"], color=PALETTE["ocean"], lw=1.5)
    min_idx = daily["mae"].idxmin()
    max_idx = daily["mae"].idxmax()
    ax.scatter(daily.loc[min_idx, "date"], daily.loc[min_idx, "mae"], s=25, color=PALETTE["teal"], edgecolor="white", linewidth=0.5, zorder=4)
    ax.scatter(daily.loc[max_idx, "date"], daily.loc[max_idx, "mae"], s=25, color=PALETTE["coral"], edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_ylim(0, 0.112)
    ax.set_yticks([0.00, 0.05, 0.10])
    ax.set_yticklabels(["0", "0.05", "0.10"], fontsize=6.0)
    xticks = pd.to_datetime(["2025-01-01", "2025-03-01", "2025-05-01"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(["Jan", "Mar", "May"], fontsize=6.0)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.6)
    ax.tick_params(axis="both", length=2, width=0.6)
    ax.set_ylabel("MAE", fontsize=6.2, labelpad=2)
    ax.set_title("Daily 2025 prediction record", loc="left", fontsize=8.7, fontweight="bold", color=PALETTE["ink"], pad=4)
    ax.set_facecolor("none")


def draw_metric_stack(ax: plt.Axes, data: dict[str, object]) -> None:
    ocean = data["ocean"]
    mixed_b = data["mixed_b"]
    stability = data["stability"]
    daily = data["daily"]
    depth = data["depth"]

    improvement = (float(mixed_b["mae"]) - float(ocean["mae"])) / float(mixed_b["mae"]) * 100
    year_samples = int(stability["sample_count"].sum())
    year_days = int(stability["days"].sum())
    stable_days = int(stability["stable_days"].sum())
    weighted_mae = np.average(stability["weighted_mae"], weights=stability["sample_count"])
    mean_r2 = np.average(stability["mean_r2"], weights=stability["sample_count"])
    daily_mean = float(daily["mae"].mean())
    daily_min = float(daily["mae"].min())
    daily_min_date = daily.loc[daily["mae"].idxmin(), "date"].strftime("%b %d")
    total_2025_samples = int(daily["sample_count"].sum())
    depth_min = int(depth["depth_m"].min())
    depth_max = int(depth["depth_m"].max())

    rounded_box(
        ax,
        0.635,
        0.110,
        0.330,
        0.745,
        facecolor=PALETTE["panel"],
        edgecolor="#D8D7CF",
        linewidth=1.0,
        radius=0.025,
    )
    ax.text(
        0.665,
        0.815,
        "Evidence snapshot",
        transform=ax.transAxes,
        fontsize=15.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    wrapped_text(
        ax,
        0.666,
        0.777,
        "Compact README view generated from current support tables.",
        46,
        size=7.6,
        color=PALETTE["muted"],
    )

    rounded_box(
        ax,
        0.665,
        0.462,
        0.265,
        0.068,
        facecolor=PALETTE["soft_coral"],
        edgecolor="#E0B8B0",
        linewidth=0.8,
        radius=0.014,
    )
    ax.text(
        0.684,
        0.506,
        "Ocean-TP MAE",
        transform=ax.transAxes,
        fontsize=6.9,
        fontweight="bold",
        color=PALETTE["muted"],
        va="center",
    )
    ax.text(
        0.684,
        0.478,
        f"{float(ocean['mae']):.5f}",
        transform=ax.transAxes,
        fontsize=18.5,
        fontweight="bold",
        color=PALETTE["coral"],
        va="center",
    )
    ax.text(
        0.805,
        0.478,
        f"R^2 = {float(ocean['r2']):.5f}\n{improvement:.1f}% lower MAE than\nMixed-B-only control",
        transform=ax.transAxes,
        fontsize=6.7,
        color=PALETTE["ink"],
        va="center",
        linespacing=1.25,
    )

    rounded_box(
        ax,
        0.665,
        0.204,
        0.126,
        0.070,
        facecolor=PALETTE["soft_teal"],
        edgecolor="#B8D8D2",
        radius=0.013,
        linewidth=0.75,
    )
    rounded_box(
        ax,
        0.804,
        0.204,
        0.126,
        0.070,
        facecolor=PALETTE["soft_blue"],
        edgecolor="#BACDDE",
        radius=0.013,
        linewidth=0.75,
    )
    ax.text(0.680, 0.252, "2023-2024", transform=ax.transAxes, fontsize=6.5, color=PALETTE["muted"], fontweight="bold")
    ax.text(0.680, 0.225, f"{stable_days}/{year_days} stable days", transform=ax.transAxes, fontsize=8.4, color=PALETTE["ink"], fontweight="bold")
    ax.text(0.819, 0.252, "fixed equation", transform=ax.transAxes, fontsize=6.5, color=PALETTE["muted"], fontweight="bold")
    ax.text(0.819, 0.225, f"MAE {weighted_mae:.4f}", transform=ax.transAxes, fontsize=8.4, color=PALETTE["ink"], fontweight="bold")
    ax.text(0.819, 0.209, f"mean R^2 {mean_r2:.4f}", transform=ax.transAxes, fontsize=5.8, color=PALETTE["muted"])

    rounded_box(
        ax,
        0.665,
        0.144,
        0.265,
        0.042,
        facecolor="#FBFAF5",
        edgecolor="#E3E1D8",
        linewidth=0.65,
        radius=0.012,
    )
    ax.text(
        0.681,
        0.170,
        f"2025 daily mean MAE {daily_mean:.4f}; minimum {daily_min:.4f} on {daily_min_date}; {total_2025_samples:,} samples.",
        transform=ax.transAxes,
        fontsize=6.1,
        color=PALETTE["ink"],
        va="center",
    )
    ax.text(
        0.665,
        0.120,
        f"Depth support: {len(depth):,} one-meter bins spanning {depth_min}-{depth_max} m; 2023-2024 evaluation samples: {year_samples:,}.",
        transform=ax.transAxes,
        fontsize=6.15,
        color=PALETTE["muted"],
        va="center",
    )


def draw_header_footer(ax: plt.Axes) -> None:
    ax.text(
        0.040,
        0.935,
        "Ocean-TP",
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="top",
    )
    ax.text(
        0.218,
        0.927,
        "physics-guided ocean temperature-pressure modelling",
        transform=ax.transAxes,
        fontsize=11.5,
        color=PALETTE["muted"],
        ha="left",
        va="top",
    )
    badges = [
        ("OOI records", PALETTE["soft_blue"], PALETTE["blue"]),
        ("Mixed-B basis", PALETTE["soft_teal"], PALETTE["teal"]),
        ("Ocean-TP constraints", PALETTE["soft_coral"], PALETTE["coral"]),
    ]
    bx = 0.218
    for label, face, edge in badges:
        rounded_box(
            ax,
            bx,
            0.878,
            0.112 if label != "Ocean-TP constraints" else 0.145,
            0.035,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.7,
            radius=0.012,
            zorder=6,
        )
        ax.text(
            bx + (0.056 if label != "Ocean-TP constraints" else 0.0725),
            0.895,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.9,
            color=PALETTE["ink"],
            fontweight="bold",
            zorder=7,
        )
        bx += 0.124 if label != "Ocean-TP constraints" else 0.158

    ax.text(
        0.040,
        0.065,
        "README graphical abstract generated from the repository support records under data/supporting/.",
        transform=ax.transAxes,
        fontsize=6.3,
        color=PALETTE["muted"],
        ha="left",
        va="center",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor=PALETTE["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    draw_background(ax)
    draw_header_footer(ax)
    draw_method_panel(ax)
    draw_metric_stack(ax, data)
    draw_benchmark_panel(fig, data)
    draw_daily_panel(fig, data)

    svg_path = Path(f"{OUT}.svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    clean_svg(svg_path)
    fig.savefig(f"{OUT}.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"{OUT}.png")


if __name__ == "__main__":
    main()
