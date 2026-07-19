#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOI 数据分析脚本
- 扫描指定目录的 OOI CSV 文件（ctdpfl000），按小时聚合温度、按深度聚合剖面，生成可视化。
- 兼容两行表头格式：第一行字段名，第二行单位行将被跳过。
- 默认仅使用 QC=1 的数据（代表通过质量控制）。
输出：
- figures/temp_time_series_mean_hourly.png  小时均温时间序列
- figures/temp_vs_depth_profile_mean.png    全局平均温度-深度剖面（10m 分箱）
- figures/temp_depth_time_heatmap.png       时间-深度二维热图（小时×10m）
- README_ooi_report.md                      简要统计报告
"""
import os
import sys
import glob
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

# 字体设置，尽量保证中文正常显示
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

# 常量配置
COL_TIME = "time"
COL_TEMP = "sea_water_temperature_profiler_depth_enabled"
COL_QC = "sea_water_temperature_profiler_depth_enabled_qc_agg"
COL_Z = "z"  # 单位 m，向下为负
QC_GOOD_VALUES = {1}  # 仅保留 QC 聚合为 1 的数据
DEPTH_BIN_SIZE_M = 10  # 深度分箱宽度（米）


def list_csv_files(data_dir: Path) -> list:
    pattern = str(data_dir / "ooi-*.csv")
    files = sorted(glob.glob(pattern))
    return files


def read_ooi_chunked(path: str, chunksize: int = 200_000):
    """按块读取 OOI CSV，跳过第二行单位，解析时间为 UTC。"""
    return pd.read_csv(
        path,
        header=0,
        skiprows=[1],  # 跳过单位行
        usecols=[COL_TIME, COL_TEMP, COL_QC, COL_Z],
        parse_dates=[COL_TIME],
        infer_datetime_format=True,
        chunksize=chunksize,
    )


def ensure_figures_dir(out_dir: Path):
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)


def aggregate_hourly_temp(chunks):
    """对每个块进行：筛选 QC=1，按小时聚合温度均值。返回 DataFrame 列：time_hour, temp_mean, count。"""
    dfs = []
    for df in chunks:
        df = df[df[COL_QC].isin(QC_GOOD_VALUES)].copy()
        if df.empty:
            continue
        df["time_hour"] = df[COL_TIME].dt.floor("H")
        g = df.groupby("time_hour", as_index=False).agg(
            temp_mean=(COL_TEMP, "mean"),
            count=(COL_TEMP, "count"),
        )
        dfs.append(g)
    if not dfs:
        return pd.DataFrame(columns=["time_hour", "temp_mean", "count"])
    out = pd.concat(dfs, ignore_index=True)
    # 二次聚合（跨块合并），按加权平均合并
    agg = out.groupby("time_hour", as_index=False).apply(
        lambda x: pd.Series({
            "temp_mean": np.average(x["temp_mean"], weights=x["count"]),
            "count": int(x["count"].sum()),
        })
    ).reset_index()
    return agg


def aggregate_depth_profile(chunks, bin_size_m=DEPTH_BIN_SIZE_M):
    """聚合全局深度剖面（10m 分箱），返回 depth_bin_m, temp_mean, count。"""
    dfs = []
    for df in chunks:
        df = df[df[COL_QC].isin(QC_GOOD_VALUES)].copy()
        if df.empty:
            continue
        df["depth_pos_m"] = -df[COL_Z]  # z 为负值，转换为正的深度
        df["depth_bin_m"] = (df["depth_pos_m"] // bin_size_m).astype(int) * bin_size_m
        g = df.groupby("depth_bin_m", as_index=False).agg(
            temp_mean=(COL_TEMP, "mean"),
            count=(COL_TEMP, "count"),
        )
        dfs.append(g)
    if not dfs:
        return pd.DataFrame(columns=["depth_bin_m", "temp_mean", "count"])
    out = pd.concat(dfs, ignore_index=True)
    agg = out.groupby("depth_bin_m", as_index=False).apply(
        lambda x: pd.Series({
            "temp_mean": np.average(x["temp_mean"], weights=x["count"]),
            "count": int(x["count"].sum()),
        })
    ).reset_index()
    return agg.sort_values("depth_bin_m")


def aggregate_time_depth_heatmap(chunks, bin_size_m=DEPTH_BIN_SIZE_M):
    """生成时间(小时)×深度(10m)二维聚合：返回 DataFrame 含 time_hour, depth_bin_m, temp_mean, count。"""
    dfs = []
    for df in chunks:
        df = df[df[COL_QC].isin(QC_GOOD_VALUES)].copy()
        if df.empty:
            continue
        df["time_hour"] = df[COL_TIME].dt.floor("H")
        df["depth_pos_m"] = -df[COL_Z]
        df["depth_bin_m"] = (df["depth_pos_m"] // bin_size_m).astype(int) * bin_size_m
        g = df.groupby(["time_hour", "depth_bin_m"], as_index=False).agg(
            temp_mean=(COL_TEMP, "mean"),
            count=(COL_TEMP, "count"),
        )
        dfs.append(g)
    if not dfs:
        return pd.DataFrame(columns=["time_hour", "depth_bin_m", "temp_mean", "count"])
    out = pd.concat(dfs, ignore_index=True)
    # 合并同键，权重平均
    agg = out.groupby(["time_hour", "depth_bin_m"], as_index=False).apply(
        lambda x: pd.Series({
            "temp_mean": np.average(x["temp_mean"], weights=x["count"]),
            "count": int(x["count"].sum()),
        })
    ).reset_index()
    return agg


def plot_hourly_series(df_hourly: pd.DataFrame, out_path: Path):
    if df_hourly.empty:
        return
    plt.figure(figsize=(12, 4))
    plt.plot(df_hourly["time_hour"], df_hourly["temp_mean"], color="#1f77b4", lw=1.5)
    plt.xlabel("时间（小时）")
    plt.ylabel("温度（°C）")
    plt.title("小时均温时间序列（QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_depth_profile(df_depth: pd.DataFrame, out_path: Path):
    if df_depth.empty:
        return
    plt.figure(figsize=(5, 8))
    plt.plot(df_depth["temp_mean"], df_depth["depth_bin_m"], color="#d62728", lw=1.5)
    plt.gca().invert_yaxis()  # 深度向下增加
    plt.xlabel("温度（°C）")
    plt.ylabel("深度（m，10m 分箱）")
    plt.title("平均温度-深度剖面（QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap(df_td: pd.DataFrame, out_path: Path):
    if df_td.empty:
        return
    # 透视为二维矩阵
    pivot = df_td.pivot(index="depth_bin_m", columns="time_hour", values="temp_mean")
    plt.figure(figsize=(14, 6))
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(pivot, cmap=cmap)
    plt.xlabel("时间（小时）")
    plt.ylabel("深度（m，10m 分箱）")
    plt.title("温度时间-深度热图（QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def collect_samples(files, per_chunk_n=1000, max_total=100000):
    """抽样收集温度/深度/时间数据以绘制散点、直方图/核密度、箱线图。"""
    samples = []
    total = 0
    for f in files:
        for ch in read_ooi_chunked(f, chunksize=200_000):
            df = ch[ch[COL_QC].isin(QC_GOOD_VALUES)].copy()
            if df.empty:
                continue
            df["depth_pos_m"] = -df[COL_Z]
            df["time_hour"] = df[COL_TIME].dt.floor("H")
            n = min(per_chunk_n, len(df))
            samp = df.sample(n=n, random_state=42)
            samples.append(samp[[COL_TEMP, "depth_pos_m", "time_hour"]])
            total += n
            if total >= max_total:
                break
        if total >= max_total:
            break
    if not samples:
        return pd.DataFrame(columns=[COL_TEMP, "depth_pos_m", "time_hour"])
    return pd.concat(samples, ignore_index=True)


def plot_temp_pressure_scatter(df_samples: pd.DataFrame, out_path: Path):
    if df_samples.empty:
        return
    plt.figure(figsize=(6, 5))
    # 近似压力(dbar) ≈ 深度(m)
    plt.scatter(df_samples["depth_pos_m"], df_samples[COL_TEMP], s=5, alpha=0.5, color="#2ca02c")
    plt.xlabel("近似压力（dbar，≈深度m）")
    plt.ylabel("温度（°C）")
    plt.title("T–P 散点图（抽样，QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_temp_hist_kde(df_samples: pd.DataFrame, out_path_hist: Path, out_path_kde: Path):
    if df_samples.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(df_samples[COL_TEMP], bins=50, color="#1f77b4", alpha=0.8)
    plt.xlabel("温度（°C）")
    plt.ylabel("频数")
    plt.title("温度直方图（抽样，QC=1）")
    plt.tight_layout()
    plt.savefig(out_path_hist, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.kdeplot(df_samples[COL_TEMP], fill=True, color="#ff7f0e")
    plt.xlabel("温度（°C）")
    plt.ylabel("密度")
    plt.title("温度核密度（抽样，QC=1）")
    plt.tight_layout()
    plt.savefig(out_path_kde, dpi=150)
    plt.close()


def plot_boxplot_by_depth(df_samples: pd.DataFrame, out_path: Path, bin_size=100):
    if df_samples.empty:
        return
    df = df_samples.copy()
    df["depth_bin_m"] = (df["depth_pos_m"] // bin_size).astype(int) * bin_size
    # 仅保留有足够样本的分箱
    counts = df.groupby("depth_bin_m")[COL_TEMP].count()
    valid_bins = counts[counts >= 30].index
    df = df[df["depth_bin_m"].isin(valid_bins)]
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="depth_bin_m", y=COL_TEMP)
    plt.xlabel(f"深度分箱（m，{bin_size}m 间隔）")
    plt.ylabel("温度（°C）")
    plt.title("不同深度层温度分布（箱线图，抽样，QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tp_regression(df_samples: pd.DataFrame, out_path: Path):
    if df_samples.empty:
        return None
    plt.figure(figsize=(6, 5))
    sns.regplot(x="depth_pos_m", y=COL_TEMP, data=df_samples, scatter_kws={"s":5, "alpha":0.4}, line_kws={"color":"#d62728"})
    plt.xlabel("近似压力（dbar，≈深度m）")
    plt.ylabel("温度（°C）")
    plt.title("T–P 回归与相关性（抽样，QC=1）")
    # 计算皮尔逊相关系数
    r = df_samples["depth_pos_m"].corr(df_samples[COL_TEMP])
    plt.annotate(f"Pearson r = {r:.3f}", xy=(0.02, 0.95), xycoords="axes fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return float(r)


def plot_tp_hexbin(df_samples: pd.DataFrame, out_path: Path):
    if df_samples.empty:
        return
    plt.figure(figsize=(6, 5))
    hb = plt.hexbin(df_samples["depth_pos_m"], df_samples[COL_TEMP], gridsize=60, cmap="viridis", mincnt=1)
    plt.colorbar(hb, label="计数")
    plt.xlabel("近似压力（dbar，≈深度m）")
    plt.ylabel("温度（°C）")
    plt.title("T–P 二维密度（Hexbin，抽样，QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_temp_gradient(df_depth: pd.DataFrame, out_path: Path):
    if df_depth.empty:
        return
    df = df_depth.sort_values("depth_bin_m").copy()
    dz = np.gradient(df["depth_bin_m"])
    dt = np.gradient(df["temp_mean"])
    grad = dt / dz
    plt.figure(figsize=(5, 8))
    plt.plot(grad, df["depth_bin_m"], color="#9467bd", lw=1.5)
    plt.gca().invert_yaxis()
    plt.xlabel("dT/dz（°C/m）")
    plt.ylabel("深度（m，10m 分箱）")
    plt.title("温度梯度剖面（QC=1）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def aggregate_depth_stats(files, bin_size_m=DEPTH_BIN_SIZE_M):
    dfs = []
    for f in files:
        for df in read_ooi_chunked(f, chunksize=200_000):
            df = df[df[COL_QC].isin(QC_GOOD_VALUES)].copy()
            if df.empty:
                continue
            df["depth_pos_m"] = -df[COL_Z]
            df["depth_bin_m"] = (df["depth_pos_m"] // bin_size_m).astype(int) * bin_size_m
            df["temp_sq"] = df[COL_TEMP] ** 2
            g = df.groupby("depth_bin_m", as_index=False).agg(
                sum_temp=(COL_TEMP, "sum"),
                sum_temp_sq=("temp_sq", "sum"),
                count=(COL_TEMP, "count"),
            )
            dfs.append(g)
    if not dfs:
        return pd.DataFrame(columns=["depth_bin_m", "mean", "std", "count"])
    stats = pd.concat(dfs, ignore_index=True).groupby("depth_bin_m", as_index=False).sum()
    stats["mean"] = stats["sum_temp"] / stats["count"]
    stats["var"] = stats["sum_temp_sq"] / stats["count"] - stats["mean"] ** 2
    stats["std"] = np.sqrt(stats["var"].clip(lower=0))
    return stats[["depth_bin_m", "mean", "std", "count"]].sort_values("depth_bin_m")


def plot_errorbar_by_depth(df_stats: pd.DataFrame, out_path: Path):
    if df_stats.empty:
        return
    plt.figure(figsize=(6, 8))
    plt.errorbar(df_stats["mean"], df_stats["depth_bin_m"], xerr=df_stats["std"], fmt="o", ecolor="#7f7f7f", color="#1f77b4", ms=3, capsize=2)
    plt.gca().invert_yaxis()
    plt.xlabel("温度均值（°C） ± 标准差")
    plt.ylabel("深度（m，10m 分箱）")
    plt.title("按深度温度均值与标准差（误差棒）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report(report_path: Path, files: list, df_hourly: pd.DataFrame, df_depth: pd.DataFrame, extra_info: dict = None):
    # 简要统计
    lines = [
        "# OOI 数据分析简报\n",
        f"数据文件数量：{len(files)}\n",
    ]
    if not df_hourly.empty:
        lines.append(
            f"小时序列时间范围：{df_hourly['time_hour'].min()} 至 {df_hourly['time_hour'].max()}\n"
        )
        lines.append(
            f"小时均温（全局）范围：{df_hourly['temp_mean'].min():.3f} 至 {df_hourly['temp_mean'].max():.3f} °C\n"
        )
    if not df_depth.empty:
        lines.append(
            f"深度范围（分箱中心）：{int(df_depth['depth_bin_m'].min())} m 至 {int(df_depth['depth_bin_m'].max())} m\n"
        )
        lines.append(
            f"平均温度（剖面）范围：{df_depth['temp_mean'].min():.3f} 至 {df_depth['temp_mean'].max():.3f} °C\n"
        )
    lines.append("\n输出图表位于 figures/ 目录。\n")
    lines.append("- temp_time_series_mean_hourly.png\n")
    lines.append("- temp_vs_depth_profile_mean.png\n")
    lines.append("- temp_depth_time_heatmap.png\n")
    lines.append("- temp_vs_pressure_scatter.png\n")
    lines.append("- temp_histogram.png, temp_kde.png\n")
    lines.append("- temp_boxplot_by_depth_100m.png\n")
    lines.append("- temp_tp_regression.png（含 Pearson r）\n")
    lines.append("- temp_tp_hexbin.png\n")
    lines.append("- temp_gradient_vs_depth.png\n")
    lines.append("- temp_errorbar_by_depth.png\n")
    if extra_info and "pearson_r" in extra_info:
        lines.append(f"\nT–P Pearson 相关系数（抽样）：{extra_info['pearson_r']:.3f}\n")
    lines.append("\n注：近似压力(dbar)以深度(m)近似；字体设置尝试使用系统中文字体以避免中文乱码。\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    # 解析数据目录
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        data_dir = Path(__file__).parent.resolve()

    if not data_dir.exists():
        print(f"数据目录不存在：{data_dir}")
        sys.exit(1)

    files = list_csv_files(data_dir)
    if not files:
        print(f"未在 {data_dir} 找到匹配的 CSV：ooi-*.csv")
        sys.exit(1)

    print(f"发现 {len(files)} 个 CSV 文件，开始聚合…")

    # 累计聚合（跨文件整合）
    dfs_hourly = []
    dfs_depth = []
    dfs_td = []

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 处理：{Path(f).name}")
        chunks_h = read_ooi_chunked(f, chunksize=200_000)
        df_h = aggregate_hourly_temp(chunks_h)
        chunks_d = read_ooi_chunked(f, chunksize=200_000)
        df_d = aggregate_depth_profile(chunks_d)
        chunks_t = read_ooi_chunked(f, chunksize=200_000)
        df_t = aggregate_time_depth_heatmap(chunks_t)
        if not df_h.empty:
            dfs_hourly.append(df_h)
        if not df_d.empty:
            dfs_depth.append(df_d)
        if not df_t.empty:
            dfs_td.append(df_t)

    # 跨文件加权合并
    def combine_weighted(dfs, key_cols):
        if not dfs:
            return pd.DataFrame(columns=key_cols + ["temp_mean", "count"])
        df = pd.concat(dfs, ignore_index=True)
        df["weighted_sum"] = df["temp_mean"] * df["count"]
        agg = df.groupby(key_cols, as_index=False).agg(
            weighted_sum=("weighted_sum", "sum"),
            count=("count", "sum"),
        )
        agg["temp_mean"] = agg["weighted_sum"] / agg["count"]
        return agg[key_cols + ["temp_mean", "count"]]

    df_hourly_all = combine_weighted(dfs_hourly, ["time_hour"]).sort_values("time_hour")
    df_depth_all = combine_weighted(dfs_depth, ["depth_bin_m"]).sort_values("depth_bin_m")
    df_td_all = combine_weighted(dfs_td, ["time_hour", "depth_bin_m"]).sort_values(["time_hour", "depth_bin_m"])

    out_dir = data_dir / "figures"
    ensure_figures_dir(data_dir)
    print("生成图表…")
    plot_hourly_series(df_hourly_all, out_dir / "temp_time_series_mean_hourly.png")
    plot_depth_profile(df_depth_all, out_dir / "temp_vs_depth_profile_mean.png")
    plot_heatmap(df_td_all, out_dir / "temp_depth_time_heatmap.png")

    # 抽样并生成更多可视化
    print("抽样并生成 T–P 散点、直方图/KDE、箱线图…")
    df_samples = collect_samples(files, per_chunk_n=1000, max_total=100000)
    plot_temp_pressure_scatter(df_samples, out_dir / "temp_vs_pressure_scatter.png")
    plot_temp_hist_kde(df_samples, out_dir / "temp_histogram.png", out_dir / "temp_kde.png")
    plot_boxplot_by_depth(df_samples, out_dir / "temp_boxplot_by_depth_100m.png", bin_size=100)

    # T–P 回归与二维密度
    print("生成 T–P 回归与二维密度图…")
    pearson_r = plot_tp_regression(df_samples, out_dir / "temp_tp_regression.png")
    plot_tp_hexbin(df_samples, out_dir / "temp_tp_hexbin.png")

    # 温度梯度剖面
    print("生成温度梯度剖面图…")
    plot_temp_gradient(df_depth_all, out_dir / "temp_gradient_vs_depth.png")

    # 深度误差棒图（全量统计）
    print("聚合深度统计并生成误差棒图…")
    df_depth_stats = aggregate_depth_stats(files, bin_size_m=DEPTH_BIN_SIZE_M)
    plot_errorbar_by_depth(df_depth_stats, out_dir / "temp_errorbar_by_depth.png")

    print("输出报告…")
    write_report(data_dir / "README_ooi_report.md", files, df_hourly_all, df_depth_all, extra_info={"pearson_r": pearson_r if pearson_r is not None else np.nan})

    print("完成。图表请查看 figures/ 目录，报告 README_ooi_report.md。")


if __name__ == "__main__":
    main()