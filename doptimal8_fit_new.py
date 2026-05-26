#!/usr/bin/env python3
import os
import glob
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

random.seed(42)
np.random.seed(42)

def basis_functions(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 0.1)
    cols = [
        np.ones_like(P),
        P,
        np.log(P),
        np.sqrt(P),
        np.exp(-P / 500.0),
        np.exp(-P / 1200.0),
        1.0 / (P + 50.0),
        1.0 / (P + 500.0),
    ]
    return np.column_stack(cols)

GLOBAL_COEFFS = np.array([
    -356.873,
    0.012505,
    62.2025,
    -3.35576,
    20.0563,
    14.8832,
    10011.3,
    7576.48,
], dtype=float)

def load_ooi_files(data_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(data_dir, "ooi-*.csv")))
    files = [f for f in files if "synthetic" not in os.path.basename(f)]
    return files

def read_ooi_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1])
    if "sea_water_temperature_profiler_depth_enabled_qc_agg" in df.columns:
        df = df[df["sea_water_temperature_profiler_depth_enabled_qc_agg"] == 1]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sea_water_temperature_profiler_depth_enabled", "z"])
    return df

def build_candidate_pool(files: list, stride: int = 200, max_total: int = 60000) -> pd.DataFrame:
    parts = []
    for f in files:
        try:
            df = read_ooi_df(f)
            if len(df) == 0:
                continue
            sampled = df.iloc[::stride][["sea_water_temperature_profiler_depth_enabled", "z"]]
            sampled = sampled.rename(columns={"sea_water_temperature_profiler_depth_enabled": "T"})
            sampled["P"] = np.abs(sampled["z"])  # P ≈ |z|
            sampled = sampled[(sampled["P"] >= 100.0) & (sampled["P"] <= 4500.0)]
            parts.append(sampled[["P", "T"]])
        except Exception:
            continue
    if not parts:
        raise RuntimeError("No candidate data available")
    cat = pd.concat(parts, ignore_index=True)
    # 分层抽样：按P分成若干箱，每箱随机采样固定数量，提升覆盖
    bins = np.linspace(cat["P"].min(), cat["P"].max(), 200)
    cat["bin"] = np.digitize(cat["P"], bins) - 1
    selected_rows = []
    per_bin = max(1, int(max_total / max(1, cat["bin"].nunique())))
    for b, g in cat.groupby("bin"):
        if len(g) > per_bin:
            selected_rows.append(g.sample(n=per_bin, random_state=42))
        else:
            selected_rows.append(g)
    pool = pd.concat(selected_rows, ignore_index=True)
    if len(pool) > max_total:
        pool = pool.sample(n=max_total, random_state=42)
    return pool[["P", "T"]].reset_index(drop=True)

def logdet_xtx(X: np.ndarray, alpha: float = 1e-6) -> float:
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    # 使用Cholesky求logdet更稳定
    try:
        L = np.linalg.cholesky(XtX)
        return 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # 回退到SVD
        s = np.linalg.svd(XtX, compute_uv=False)
        return np.sum(np.log(s + 1e-12))

def fedorov_d_optimal(P_pool: np.ndarray, n_points: int = 8, max_iter: int = 10, candidate_sample: int = 1000) -> np.ndarray:
    X_pool = basis_functions(P_pool)
    # 列标准化，减少尺度影响
    X_pool = (X_pool - X_pool.mean(axis=0)) / (X_pool.std(axis=0) + 1e-12)
    # 初始化：按分位数均匀选取
    qs = np.linspace(0.05, 0.95, n_points)
    P_quantiles = np.quantile(P_pool, qs)
    sel_idx = [int(np.argmin(np.abs(P_pool - pq))) for pq in P_quantiles]
    sel_idx = np.array(sel_idx, dtype=int)
    best_logdet = logdet_xtx(X_pool[sel_idx])

    for it in range(max_iter):
        improved = False
        # 每轮对每个已选点尝试交换
        for k in range(n_points):
            # 采样候选子集以降低计算量
            cand_idx = np.random.choice(len(P_pool), size=min(candidate_sample, len(P_pool)), replace=False)
            current_rows = sel_idx.copy()
            for c in cand_idx:
                if c in current_rows:
                    continue
                trial_rows = current_rows.copy()
                trial_rows[k] = c
                trial_logdet = logdet_xtx(X_pool[trial_rows])
                if trial_logdet > best_logdet + 1e-8:
                    sel_idx = trial_rows
                    best_logdet = trial_logdet
                    improved = True
        if not improved:
            break
    return P_pool[sel_idx]

def gather_band_samples(cat: pd.DataFrame, targets: np.ndarray, band_width: float = 25.0, max_per_band: int = 8000):
    rows = []
    for pt in targets:
        band = cat[np.abs(cat["P"] - pt) <= band_width]
        if len(band) == 0:
            band = cat[np.abs(cat["P"] - pt) <= (band_width * 2)]
        if len(band) == 0:
            continue
        if len(band) > max_per_band:
            band = band.sample(n=max_per_band, random_state=42)
        rows.append(band)
    if not rows:
        raise RuntimeError("No samples gathered around targets")
    big = pd.concat(rows, ignore_index=True)
    P = big["P"].values
    T = big["T"].values
    X = basis_functions(P)
    return X, T

def solve_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 5e-3) -> np.ndarray:
    XtX = X.T @ X
    reg = XtX + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    return np.linalg.solve(reg, Xty)

def predict(P: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return basis_functions(P) @ coeffs

def eval_metrics(coeffs: np.ndarray, files: list) -> dict:
    sse_total = 0.0
    mae_total = 0.0
    n_total = 0
    sum_y = 0.0
    sum_y2 = 0.0
    for f in files:
        try:
            df = read_ooi_df(f)
            if len(df) == 0:
                continue
            T = df["sea_water_temperature_profiler_depth_enabled"].to_numpy()
            P = np.abs(df["z"].to_numpy())
            mask = (P >= 100.0) & (P <= 4500.0)
            T = T[mask]
            P = P[mask]
            y_pred = predict(P, coeffs)
            err = y_pred - T
            sse_total += float(np.sum(err ** 2))
            mae_total += float(np.sum(np.abs(err)))
            n_total += len(T)
            sum_y += float(np.sum(T))
            sum_y2 += float(np.sum(T ** 2))
        except Exception:
            continue
    mse = sse_total / max(n_total, 1)
    rmse = math.sqrt(mse)
    mae = mae_total / max(n_total, 1)
    y_mean = sum_y / max(n_total, 1)
    sst = sum_y2 - n_total * (y_mean ** 2)
    r2 = 1.0 - (sse_total / sst) if sst > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n_total}

def make_plot(pool: pd.DataFrame, coeffs_opt: np.ndarray, coeffs_uniform: np.ndarray, out_png: str, out_csv: Optional[str] = None):
    # 画观测均值与三条预测：全局、均匀-8、D-optimal-8
    data = pool.copy()
    bins = np.linspace(data["P"].min(), data["P"].max(), 80)
    data["bin"] = np.digitize(data["P"], bins) - 1
    gb = data.groupby("bin")
    P_bin = gb["P"].mean().to_numpy()
    T_bin = gb["T"].mean().to_numpy()
    pred_global = predict(P_bin, GLOBAL_COEFFS)
    pred_opt = predict(P_bin, coeffs_opt)
    pred_uniform = predict(P_bin, coeffs_uniform)
    err_global = np.abs(pred_global - T_bin)
    err_opt = np.abs(pred_opt - T_bin)
    err_uniform = np.abs(pred_uniform - T_bin)

    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(P_bin, T_bin, label="Observed mean", color="black")
    ax1.plot(P_bin, pred_global, label="Global coefficients", color="#1f77b4")
    ax1.plot(P_bin, pred_opt, label="D-optimal-8 fit", color="#2ca02c")
    ax1.plot(P_bin, pred_uniform, label="Uniform-8 fit", color="#ff7f0e")
    ax1.set_xlabel("Pressure P (dbar ≈ meters)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature curves: Global vs Uniform-8 vs D-optimal-8")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(P_bin, err_global, label="Global coefficient error", color="#1f77b4")
    ax2.plot(P_bin, err_uniform, label="Uniform-8 error", color="#ff7f0e")
    ax2.plot(P_bin, err_opt, label="D-optimal-8 error", color="#2ca02c")
    ax2.set_xlabel("Pressure P (dbar ≈ meters)")
    ax2.set_ylabel("Absolute error |ΔT| (°C)")
    ax2.set_title("Error curves: Global vs Uniform-8 vs D-optimal-8")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    # 可选导出曲线数据
    if out_csv is not None:
        try:
            import pandas as pd
            df = pd.DataFrame({
                "P": P_bin,
                "obs_mean": T_bin,
                "pred_global": pred_global,
                "pred_uniform": pred_uniform,
                "pred_dopt": pred_opt,
                "err_global": err_global,
                "err_uniform": err_uniform,
                "err_dopt": err_opt,
            })
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            df.to_csv(out_csv, index=False)
        except Exception:
            pass

def main():
    data_dir = "../data"
    files = load_ooi_files(data_dir)
    pool = build_candidate_pool(files, stride=200, max_total=60000)
    P_pool = pool["P"].values
    # Fedorov D-optimal selection for 8 points
    targets = fedorov_d_optimal(P_pool, n_points=8, max_iter=10, candidate_sample=1200)
    # Fit coefficients using band samples around targets
    X_band, y_band = gather_band_samples(pool, targets, band_width=25.0, max_per_band=8000)
    coeffs_opt = solve_ridge(X_band, y_band, alpha=5e-3)

    # Uniform-8: 按压力分位数均匀选取8点，并在各点±25 dbar 带宽采样 + 岭回归拟合
    qs = np.linspace(0.05, 0.95, 8)
    P_uniform = np.array([P_pool[int(np.argmin(np.abs(P_pool - np.quantile(P_pool, q))))] for q in qs])
    X_u, y_u = gather_band_samples(pool, P_uniform, band_width=25.0, max_per_band=8000)
    coeffs_uniform = solve_ridge(X_u, y_u, alpha=5e-3)

    # Only-8 fit: strictly use 8 nearest samples to targets (no bandwidth aggregation)
    only8_rows = []
    for pt in targets:
        idx = int(np.argmin(np.abs(pool["P"].values - pt)))
        only8_rows.append(pool.iloc[idx])
    only8_df = pd.DataFrame(only8_rows)
    P_8 = only8_df["P"].values
    T_8 = only8_df["T"].values
    X_8 = basis_functions(P_8)
    # Prefer direct solve, fallback to ridge for stability
    try:
        coeffs_only8 = np.linalg.solve(X_8, T_8)
    except np.linalg.LinAlgError:
        coeffs_only8 = solve_ridge(X_8, T_8, alpha=1e-2)

    # Evaluate on full files
    metrics_global = eval_metrics(GLOBAL_COEFFS, files)
    metrics_opt = eval_metrics(coeffs_opt, files)
    metrics_uniform = eval_metrics(coeffs_uniform, files)
    metrics_only8 = eval_metrics(coeffs_only8, files)

    report = {
        "targets_P": [float(x) for x in np.round(targets, 3)],
        "coeffs_opt": [float(x) for x in np.round(coeffs_opt, 6)],
        "uniform_P": [float(x) for x in np.round(P_uniform, 3)],
        "coeffs_uniform": [float(x) for x in np.round(coeffs_uniform, 6)],
        "coeffs_only8": [float(x) for x in np.round(coeffs_only8, 6)],
        "samples_P_only8": [float(x) for x in np.round(P_8, 3)],
        "metrics": {
            "global": metrics_global,
            "d_opt_8": metrics_opt,
            "uniform_8": metrics_uniform,
            "only8": metrics_only8,
        },
    }
    out_json = "../results/doptimal8_report.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    out_png = "../results/figures/doptimal8_accuracy.png"
    out_csv = "../results/doptimal8_curves.csv"
    make_plot(pool, coeffs_opt, coeffs_uniform, out_png, out_csv)
    # Basis functions visualization (English labels)
    out_basis_png = "../results/figures/basis_functions.png"
    plot_basis_functions(out_basis_png)
    print("Done. Report:", out_json)
    print("Figure:", out_png)
    print("Basis Functions Figure:", out_basis_png)
def plot_basis_functions(out_png: str):
    # Visualize basis functions over pressure range, normalized for comparability
    P = np.linspace(100.0, 4500.0, 500)
    X = basis_functions(P)
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
    names = [
        "1",
        "P",
        "log(P)",
        "sqrt(P)",
        "exp(-P/500)",
        "exp(-P/1200)",
        "1/(P+50)",
        "1/(P+500)",
    ]
    plt.figure(figsize=(12, 8))
    for i, nm in enumerate(names):
        plt.plot(P, Xn[:, i], label=nm)
    plt.xlabel("Pressure P (dbar ≈ meters)")
    plt.ylabel("Normalized basis value")
    plt.title("Basis functions (normalized to [0,1])")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()