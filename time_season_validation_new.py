import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 进度条（若不可用则回退为无操作）
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable
from numpy.typing import ArrayLike


"""
按季节进行时间序验证：
- 读取../data 下全部 OOI CSV（列：time, sea_water_temperature_profiler_depth_enabled, ... , z）
- 仅保留 QC=1 的样本（列 sea_water_temperature_profiler_depth_enabled_qc_agg == 1）
- 使用 derived/season_boundaries.txt 定义季节起点（4个日期：MM-DD），构造 4 个季节窗口
- 在每个季节内，按时间顺序切分训练/验证（默认 80% 训练、20% 验证），不随机
- 拟合 8 项可解释基函数的线性组合：
    T(P) = a0*1 + a1*P + a2*ln(P+) + a3*sqrt(P+) + a4*exp(-P/500) + a5*exp(-P/1200) + a6*1/(P+50) + a7*1/(P+500)
  其中 P+ 表示对非正值做最小偏移以避免无效（如 ln、sqrt）
- 计算 train MAE、train RMSE、val MAE、val RMSE、val R2
- 近似计算“物理残差均值/95%分位”占位：使用 |dT/dP| 的样本均值与95分位作为 proxy（真实 PDE 需物理方程）
- 输出 CSV 至 derived/TimeSeasonsEq_timeval.csv，并打印摘要
"""


@dataclass
class Config:
    data_dir: str = "../data"
    output_csv: str = "../results/TimeSeasonsEq_timeval.csv"
    boundaries_txt: str = "../results/season_boundaries.txt"
    train_fraction: float = 0.8
    max_rows_per_file: int = 0  # 0 表示不限制
    tz: str = "UTC"


def safe_features(p: ArrayLike) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    # 压力可能为负（数据中 z 为负），我们取深度的绝对值作为正压近似
    P = np.abs(p)
    P_pos = np.clip(P, 1e-6, None)
    X = np.column_stack([
        np.ones_like(P),
        P,
        np.log(P_pos),
        np.sqrt(P_pos),
        np.exp(-P / 500.0),
        np.exp(-P / 1200.0),
        1.0 / (P + 50.0),
        1.0 / (P + 500.0),
    ])
    return X


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 带微弱L2正则防数值不稳定
    reg = 1e-6
    A = X.T @ X + reg * np.eye(X.shape[1])
    b = X.T @ y
    coeffs = np.linalg.solve(A, b)
    return coeffs


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = np.var(y_true)
    if var <= 1e-12:
        return 1.0
    return float(1.0 - np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def physics_proxy(y_pred: np.ndarray, P: np.ndarray) -> Tuple[float, float]:
    # 简化物理指标：使用局部梯度近似 |dT/dP| 的统计
    # 实际 PDE 残差需要明确的物理方程，这里仅作占位验证
    order = np.argsort(P)
    yp = y_pred[order]
    pp = P[order]
    if len(pp) < 3:
        return 0.0, 0.0
    dP = np.diff(pp)
    dT = np.diff(yp)
    with np.errstate(divide='ignore', invalid='ignore'):
        grad = np.abs(dT / np.clip(dP, 1e-12, None))
    grad = grad[np.isfinite(grad)]
    if len(grad) == 0:
        return 0.0, 0.0
    return float(np.mean(grad)), float(np.quantile(grad, 0.95))


def read_boundaries(path: str) -> List[str]:
    # 文件包含四行 MM-DD
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines() if ln.strip()]
    if len(lines) != 4:
        raise ValueError(f"season_boundaries must have 4 lines, got {len(lines)}")
    return lines


def load_all_data(data_dir: str, max_rows_per_file: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "ooi-*.csv")))
    dfs = []
    for f in tqdm(files, desc="Loading CSV files", unit="file"):
        # 第一行为字段名，第二行为单位说明；应跳过第二行但保留字段名
        df = pd.read_csv(f, header=0, skiprows=[1])
        # 统一列名
        rename_map = {}
        if "sea_water_temperature_profiler_depth_enabled" in df.columns:
            rename_map["sea_water_temperature_profiler_depth_enabled"] = "temp"
        if "sea_water_temperature_profiler_depth_enabled_qc_agg" in df.columns:
            rename_map["sea_water_temperature_profiler_depth_enabled_qc_agg"] = "qc"
        if "z" in df.columns:
            rename_map["z"] = "depth"
        df = df.rename(columns=rename_map)
        # 若无 qc 列，则默认视为合格样本
        if "qc" not in df.columns:
            df["qc"] = 1
        if max_rows_per_file > 0:
            df = df.head(max_rows_per_file)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No data files found under {data_dir}")
    all_df = pd.concat(dfs, ignore_index=True)
    # 过滤 QC=1
    all_df["qc"] = pd.to_numeric(all_df["qc"], errors="coerce")
    all_df = all_df[all_df["qc"] == 1]
    # 解析时间
    all_df["time"] = pd.to_datetime(all_df["time"], utc=True, errors="coerce")
    all_df = all_df.dropna(subset=["time", "temp", "depth"])
    # 添加月日
    all_df["month"] = all_df["time"].dt.month
    all_df["day"] = all_df["time"].dt.day
    return all_df


def season_mask(month: int, day: int, start_md: str, end_md: str, wrap: bool) -> bool:
    sm, sd = map(int, start_md.split("-"))
    em, ed = map(int, end_md.split("-"))
    if not wrap:
        if sm == em:
            return (month == sm) and (day >= sd) and (day <= ed)
        return ((month == sm and day >= sd) or (month > sm and month < em) or (month == em and day <= ed))
    # 跨年季节：如 10-20 → 01-22
    return ((month == sm and day >= sd) or (month > sm) or (month < em) or (month == em and day <= ed))


def build_season_df(df: pd.DataFrame, boundaries: List[str]) -> Dict[str, pd.DataFrame]:
    s1, s2, s3, s4 = boundaries
    seasons = {
        "Season1": (s1, s2, False),
        "Season2": (s2, s3, False),
        "Season3": (s3, s4, False),
        "Season4": (s4, s1, True),
    }
    out = {}
    for name, (start_md, end_md, wrap) in tqdm(seasons.items(), desc="Building seasonal subsets"):
        mask = df.apply(lambda r: season_mask(int(r["month"]), int(r["day"]), start_md, end_md, wrap), axis=1)
        sdf = df[mask].copy()
        sdf = sdf.sort_values("time")
        out[name] = sdf
    return out


def train_validate_time_split(sdf: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(sdf) == 0:
        return sdf, sdf
    n_train = int(len(sdf) * frac)
    train_df = sdf.iloc[:n_train]
    val_df = sdf.iloc[n_train:]
    return train_df, val_df


def run_one_season(name: str, sdf: pd.DataFrame) -> Dict:
    # 用深度的绝对值作为压力近似
    P_train = np.abs(sdf["depth"].to_numpy())
    T_train = sdf["temp"].to_numpy()
    # 时间序切分
    train_df, val_df = train_validate_time_split(sdf, config.train_fraction)
    P_tr = np.abs(train_df["depth"].to_numpy())
    T_tr = train_df["temp"].to_numpy()
    P_va = np.abs(val_df["depth"].to_numpy())
    T_va = val_df["temp"].to_numpy()

    # 设计矩阵
    X_tr = safe_features(P_tr)
    coeffs = fit_least_squares(X_tr, T_tr)
    # 预测
    y_tr = X_tr @ coeffs
    X_va = safe_features(P_va)
    y_va = X_va @ coeffs

    # 指标
    out = {
        "season": name,
        "date_range": None,
        "total_observations": int(len(sdf)),
        "train_MAE": mae(T_tr, y_tr),
        "train_RMSE": rmse(T_tr, y_tr),
        "val_MAE": mae(T_va, y_va),
        "val_RMSE": rmse(T_va, y_va),
        "val_R2": r2(T_va, y_va),
        "physics_residual_mean": None,
        "physics_residual_q95": None,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "coef_1": float(coeffs[0]) if len(coeffs) >= 1 else None,
        "coef_P": float(coeffs[1]) if len(coeffs) >= 2 else None,
        "coef_ln(P)": float(coeffs[2]) if len(coeffs) >= 3 else None,
        "coef_sqrt(P)": float(coeffs[3]) if len(coeffs) >= 4 else None,
        "coef_exp(-P/500)": float(coeffs[4]) if len(coeffs) >= 5 else None,
        "coef_exp(-P/1200)": float(coeffs[5]) if len(coeffs) >= 6 else None,
        "coef_1/(P+50)": float(coeffs[6]) if len(coeffs) >= 7 else None,
        "coef_1/(P+500)": float(coeffs[7]) if len(coeffs) >= 8 else None,
    }

    # 近似物理指标
    m, q95 = physics_proxy(np.concatenate([y_tr, y_va]), np.concatenate([P_tr, P_va]))
    out["physics_residual_mean"] = m
    out["physics_residual_q95"] = q95

    # 日期范围（格式 MM-DD → MM-DD）
    if len(sdf) > 0:
        start_md = f"{int(sdf.iloc[0]['month']):02d}-{int(sdf.iloc[0]['day']):02d}"
        end_md = f"{int(sdf.iloc[-1]['month']):02d}-{int(sdf.iloc[-1]['day']):02d}"
        out["date_range"] = f"{start_md} → {end_md}"
    return out


def main(config: Config):
    boundaries = read_boundaries(config.boundaries_txt)
    df = load_all_data(config.data_dir, max_rows_per_file=config.max_rows_per_file)
    seasons = build_season_df(df, boundaries)
    rows = []
    for name, sdf in tqdm(seasons.items(), desc="Processing seasons"):
        res = run_one_season(name, sdf)
        rows.append(res)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(config.output_csv), exist_ok=True)
    out_df.to_csv(config.output_csv, index=False)
    print(out_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-csv", default="../results/TimeSeasonsEq_timeval.csv")
    parser.add_argument("--boundaries-txt", default="../results/season_boundaries.txt")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    args = parser.parse_args()

    config = Config(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        boundaries_txt=args.boundaries_txt,
        train_fraction=args.train_fraction,
        max_rows_per_file=args.max_rows_per_file,
    )
    main(config)