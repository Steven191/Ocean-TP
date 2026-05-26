#!/usr/bin/env python3
"""Compute depth layering with configurable regression models."""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as cheb
from pinn_tp_fit import load_ooi_samples


# Default parameters (can be overridden via CLI)
BIN_WIDTH = 50.0
MIN_THICKNESS = 500.0
MIN_SAMPLES_PER_LAYER = 1500
NUM_LAYERS = 3
VAL_FRACTION = 0.2
RNG_SEED = 42
MAX_FIT_SAMPLES_PER_LAYER = 200_000


@dataclass
class SegmentData:
    start: float
    end: float
    count: int
    indices: np.ndarray


@dataclass
class LayerMetrics:
    layer: str
    depth_range: Tuple[float, float]
    sample_size: int
    train_size: int
    val_size: int
    mae_val: float
    rmse_val: float
    mape_val: float
    r2_val: float
    mae_train: float
    rmse_train: float
    r2_train: float
    coeffs: dict
    bias_model: dict


def prepare_samples(data_dir: Path, sample_n: int) -> pd.DataFrame:
    P, z, T = load_ooi_samples(data_dir, sample_n=sample_n, seed=RNG_SEED)
    df = pd.DataFrame(
        {
            "pressure": P.numpy().ravel(),
            "depth": z.numpy().ravel(),
            "temp": T.numpy().ravel(),
        }
    )
    df["depth_pos"] = df["depth"].abs()
    return df


def aggregate_bins(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    min_depth = math.floor(df["depth_pos"].min() / BIN_WIDTH) * BIN_WIDTH
    max_depth = math.ceil(df["depth_pos"].max() / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH
    edges = np.arange(min_depth, max_depth + BIN_WIDTH, BIN_WIDTH)
    df["bin"] = pd.cut(
        df["depth_pos"],
        bins=edges,
        right=False,
        include_lowest=True,
        labels=False,
    )
    n_bins = len(edges) - 1
    grouped = df.groupby("bin").agg(
        count=("temp", "count"),
        temp_sum=("temp", "sum"),
        temp_sq=("temp", lambda x: np.sum(np.square(x))),
    )
    grouped = grouped.reindex(range(n_bins), fill_value=0.0)
    grouped["count"] = grouped["count"].astype(int)
    return edges, grouped


def compute_prefix_arrays(grouped: pd.DataFrame):
    counts = grouped["count"].to_numpy(dtype=np.int64)
    temp_sum = grouped["temp_sum"].to_numpy(dtype=np.float64)
    temp_sq = grouped["temp_sq"].to_numpy(dtype=np.float64)
    prefix_count = np.concatenate([[0], np.cumsum(counts)])
    prefix_sum = np.concatenate([[0.0], np.cumsum(temp_sum)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(temp_sq)])
    return prefix_count, prefix_sum, prefix_sq


def segment_cost(i: int, j: int, prefix_count, prefix_sum, prefix_sq):
    count = prefix_count[j] - prefix_count[i]
    if count < MIN_SAMPLES_PER_LAYER:
        return None
    total = prefix_sum[j] - prefix_sum[i]
    total_sq = prefix_sq[j] - prefix_sq[i]
    mean = total / count
    variance = max(total_sq / count - mean**2, 0.0)
    return variance


def dynamic_partition(edges: np.ndarray, prefix_count, prefix_sum, prefix_sq) -> List[float]:
    n_bins = len(edges) - 1
    dp = np.full((NUM_LAYERS + 1, n_bins + 1), np.inf)
    choice = np.full((NUM_LAYERS + 1, n_bins + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, NUM_LAYERS + 1):
        for j in range(1, n_bins + 1):
            depth_j = edges[j]
            for i in range(k - 1, j):
                depth_i = edges[i]
                thickness = depth_j - depth_i
                if thickness < MIN_THICKNESS:
                    continue
                cost = segment_cost(i, j, prefix_count, prefix_sum, prefix_sq)
                if cost is None:
                    continue
                prev = dp[k - 1, i]
                if not np.isfinite(prev):
                    continue
                val = prev + cost
                if val < dp[k, j]:
                    dp[k, j] = val
                    choice[k, j] = i
    if not np.isfinite(dp[NUM_LAYERS, n_bins]):
        raise RuntimeError("无法满足分层约束，请调整参数。")

    boundaries = []
    k = NUM_LAYERS
    j = n_bins
    while k > 0:
        i = choice[k, j]
        if i < 0:
            raise RuntimeError("分层回溯失败。")
        boundaries.append(edges[j])
        j = i
        k -= 1
    boundaries.append(edges[0])
    return sorted(boundaries)


def layer_masks(df: pd.DataFrame, boundaries: List[float]) -> List[SegmentData]:
    depth = df["depth_pos"].to_numpy()
    segments = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if idx == len(boundaries) - 2:
            mask = (depth >= start) & (depth <= end)
        else:
            mask = (depth >= start) & (depth < end)
        segments.append(SegmentData(start, end, int(mask.sum()), mask))
    return segments


def build_eight_term_features(p: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    p = p.astype(np.float64)
    features = [
        np.ones_like(p),
        p,
        p ** 2,
        np.sqrt(np.clip(p, 1.0, None)),
        np.log(np.clip(p, 1.0, None)),
        np.exp(-p / 400.0),
        np.exp(-p / 1200.0),
        1.0 / (p + 75.0),
    ]
    names = [
        "1",
        "P",
        "P^2",
        "sqrt(P)",
        "ln(P)",
        "exp(-P/400)",
        "exp(-P/1200)",
        "1/(P+75)",
    ]
    return np.column_stack(features), names


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mape = float(np.mean(np.abs(resid / np.clip(y_true, 0.1, None))))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, mape, r2


def build_multi_term_features(p: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    p = np.asarray(p, dtype=np.float64)
    p_clip = np.clip(p, 1.0, None)
    sqrt_p = np.sqrt(p_clip)
    ln_p = np.log(p_clip)
    features = [
        np.ones_like(p),
        p,
        p ** 2,
        p ** 3,
        sqrt_p,
        ln_p,
        ln_p ** 2,
        np.exp(-p / 400.0),
        np.exp(-p / 800.0),
        1.0 / (p + 50.0),
        1.0 / (p + 200.0),
        np.sin(p / 500.0),
        np.cos(p / 700.0),
    ]
    names = [
        "1",
        "P",
        "P^2",
        "P^3",
        "sqrt(P)",
        "ln(P)",
        "(lnP)^2",
        "exp(-P/400)",
        "exp(-P/800)",
        "1/(P+50)",
        "1/(P+200)",
        "sin(P/500)",
        "cos(P/700)",
    ]
    return np.column_stack(features), names


def build_multi_term_derivatives(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p_clip = np.clip(p, 1.0, None)
    sqrt_p = np.sqrt(p_clip)
    ln_p = np.log(p_clip)
    derivs = [
        np.zeros_like(p),
        np.ones_like(p),
        2.0 * p,
        3.0 * (p ** 2),
        0.5 / sqrt_p,
        1.0 / p_clip,
        2.0 * ln_p / p_clip,
        -1.0 / 400.0 * np.exp(-p / 400.0),
        -1.0 / 800.0 * np.exp(-p / 800.0),
        -1.0 / ((p + 50.0) ** 2),
        -1.0 / ((p + 200.0) ** 2),
        (1.0 / 500.0) * np.cos(p / 500.0),
        -(1.0 / 700.0) * np.sin(p / 700.0),
    ]
    return np.column_stack(derivs)


def fit_layer(df: pd.DataFrame, seg: SegmentData, layer_idx: int, model: str, deriv_weight: float) -> LayerMetrics:
    layer_df = df.loc[seg.indices].copy()
    if layer_df.empty:
        raise RuntimeError("层数据为空，无法拟合。")

    full_pressure = layer_df["pressure"].to_numpy()
    full_temp = layer_df["temp"].to_numpy()
    original_n = len(layer_df)

    rng = np.random.default_rng(RNG_SEED + layer_idx)
    if len(layer_df) > MAX_FIT_SAMPLES_PER_LAYER:
        layer_df = layer_df.sample(n=MAX_FIT_SAMPLES_PER_LAYER, random_state=rng.integers(0, 2**32))

    pressure = layer_df["pressure"].to_numpy()
    temp = layer_df["temp"].to_numpy()
    n = len(layer_df)
    val_mask = rng.random(n) < VAL_FRACTION
    if val_mask.sum() == 0 or val_mask.sum() == n:
        val_mask[0] = True
        val_mask[-1] = False
    train_mask = ~val_mask

    span = max(seg.end - seg.start, 1.0)
    df_layer = layer_df.copy()
    df_layer["depth_bin"] = df_layer["pressure"].round().astype(int)
    avg = df_layer.groupby("depth_bin", as_index=False)["temp"].mean()
    p_avg = avg["depth_bin"].to_numpy(dtype=np.float64)
    t_avg = avg["temp"].to_numpy(dtype=np.float64)

    if model == "chebyshev":
        degree = 8 if len(p_avg) > 5000 else 4
        norm = lambda P: (2.0 * (np.asarray(P) - seg.start) / span) - 1.0
        coeffs = cheb.chebfit(norm(p_avg), t_avg, degree)

        def predict(P):
            return cheb.chebval(norm(P), coeffs)

        coeff_map = {
            "basis": "chebyshev",
            "degree": int(len(coeffs) - 1),
            "domain": [float(seg.start), float(seg.end)],
            "coefficients": [float(c) for c in coeffs],
        }
    elif model == "eight_term":
        X_avg, names = build_eight_term_features(p_avg)
        ridge = 1e-6
        A = X_avg.T @ X_avg + ridge * np.eye(X_avg.shape[1])
        b = X_avg.T @ t_avg
        coeffs = np.linalg.solve(A, b)

        def predict(P):
            X, _ = build_eight_term_features(np.asarray(P))
            return X @ coeffs

        coeff_map = {
            "basis": "eight_term",
            "features": names,
            "coefficients": [float(c) for c in coeffs],
        }
    else:
        X_avg, names = build_multi_term_features(p_avg)
        X_full, _ = build_multi_term_features(pressure)
        X_deriv = build_multi_term_derivatives(p_avg)
        grad_t = np.gradient(t_avg, p_avg, edge_order=2)
        lw = 1.0
        dw = max(deriv_weight, 0.0)
        A = np.vstack(
            [
                np.sqrt(lw) * X_avg,
                np.sqrt(dw) * X_deriv,
            ]
        )
        b = np.concatenate(
            [
                np.sqrt(lw) * t_avg,
                np.sqrt(dw) * grad_t,
            ]
        )
        ridge = 1e-6
        coeffs = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)

        def predict(P):
            X, _ = build_multi_term_features(np.asarray(P))
            return X @ coeffs

        coeff_map = {
            "basis": "multi_term",
            "features": names,
            "deriv_weight": dw,
            "coefficients": [float(c) for c in coeffs],
        }

    mae_tr, rmse_tr, _, r2_tr = compute_error_metrics(temp[train_mask], predict(pressure[train_mask]))
    mae_val, rmse_val, mape_val, r2_val = compute_error_metrics(temp[val_mask], predict(pressure[val_mask]))

    preds_val = predict(pressure[val_mask])
    bias_X = np.column_stack([np.ones_like(pressure[val_mask]), pressure[val_mask]])
    bias_coef, *_ = np.linalg.lstsq(bias_X, preds_val - temp[val_mask], rcond=None)
    bias = {"c0": float(bias_coef[0]), "c1": float(bias_coef[1])}

    return LayerMetrics(
        layer=f"L{layer_idx + 1}",
        depth_range=(seg.start, seg.end),
        sample_size=original_n,
        train_size=int(train_mask.sum()),
        val_size=int(val_mask.sum()),
        mae_val=mae_val,
        rmse_val=rmse_val,
        mape_val=mape_val,
        r2_val=r2_val,
        mae_train=mae_tr,
        rmse_train=rmse_tr,
        r2_train=r2_tr,
        coeffs=coeff_map,
        bias_model=bias,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute optimized depth layering metrics.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing ooi-*.csv files.")
    parser.add_argument("--samples", type=int, default=60000, help="Number of QC=1 samples to draw.")
    parser.add_argument("--layers", type=int, default=3, help="Number of depth layers to derive.")
    parser.add_argument("--bin-width", type=float, default=50.0, help="Depth bin width in meters.")
    parser.add_argument("--min-thickness", type=float, default=500.0, help="Minimum layer thickness in meters.")
    parser.add_argument("--min-samples", type=int, default=1500, help="Minimum samples required per layer.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction per layer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model",
        type=str,
        default="chebyshev",
        choices=["chebyshev", "eight_term", "multi_term"],
        help="Projection model for layer fits.",
    )
    parser.add_argument(
        "--deriv-weight",
        type=float,
        default=0.05,
        help="Derivative constraint weight for multi_term model.",
    )
    args = parser.parse_args()

    global BIN_WIDTH, MIN_THICKNESS, MIN_SAMPLES_PER_LAYER, NUM_LAYERS, VAL_FRACTION, RNG_SEED
    BIN_WIDTH = float(args.bin_width)
    MIN_THICKNESS = float(args.min_thickness)
    MIN_SAMPLES_PER_LAYER = int(args.min_samples)
    NUM_LAYERS = int(args.layers)
    VAL_FRACTION = float(args.val_fraction)
    RNG_SEED = int(args.seed)

    data_dir = args.data_dir.resolve()
    df = prepare_samples(data_dir, sample_n=args.samples)
    edges, grouped = aggregate_bins(df)
    prefix_count, prefix_sum, prefix_sq = compute_prefix_arrays(grouped)
    boundaries = dynamic_partition(edges, prefix_count, prefix_sum, prefix_sq)
    segments = layer_masks(df, boundaries)
    metrics = [fit_layer(df, seg, idx, args.model, args.deriv_weight) for idx, seg in enumerate(segments)]

    result = {
        "boundaries": boundaries,
        "layers": [
            {
                "layer": m.layer,
                "depth_range": list(m.depth_range),
                "sample_size": m.sample_size,
                "train_size": m.train_size,
                "val_size": m.val_size,
                "mae_val": m.mae_val,
                "rmse_val": m.rmse_val,
                "mape_val": m.mape_val,
                "r2_val": m.r2_val,
                "mae_train": m.mae_train,
                "rmse_train": m.rmse_train,
                "r2_train": m.r2_train,
                "coefficients": m.coeffs,
                "bias_model": m.bias_model,
            }
            for m in metrics
        ],
        "bin_width": BIN_WIDTH,
        "min_thickness": MIN_THICKNESS,
        "min_samples_per_layer": MIN_SAMPLES_PER_LAYER,
        "sample_size": int(len(df)),
        "val_fraction": VAL_FRACTION,
        "model": args.model,
    }

    out_path = (data_dir / "derived" / "depth_layer_results.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
