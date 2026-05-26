#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PINN trainer for deep-sea Temperature–Pressure modelling.

Key features:
  • Multi-frequency Fourier embeddings (log-spaced) with optional input normalisation.
  • SIREN-style sine activation or configurable activation stack (tanh / gelu / hybrid).
  • Hybrid analytical base (8-term) so the network focuses on residual learning.
  • Multiple physics losses: first/second derivative, pseudo-PDE, Laplacian-style smoothing.
  • Adaptive physics weights, cosine LR scheduler, optional multi-head outputs.
  • CSV export of prediction curves for post-hoc analysis.

Backward compatible with the original CLI (default options preserve prior behaviour).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Data utilities (copied / adapted from the previous implementation for compatibility)
# --------------------------------------------------------------------------------------
COL_TIME = "time"
COL_TEMP = "sea_water_temperature_profiler_depth_enabled"
COL_QC = "sea_water_temperature_profiler_depth_enabled_qc_agg"
COL_Z = "z"  # negative depth (m)
QC_GOOD_VALUES = {1}


def list_csv_files(data_dir: Path) -> List[Path]:
    return sorted(f for f in data_dir.glob("ooi-*.csv") if f.is_file())


def read_ooi_chunked(path: Path, chunksize: int = 200_000):
    return pd.read_csv(
        path,
        header=0,
        skiprows=[1],
        usecols=[COL_TIME, COL_TEMP, COL_QC, COL_Z],
        parse_dates=[COL_TIME],
        infer_datetime_format=True,
        chunksize=chunksize,
    )


def load_ooi_samples(
    data_dir: Path,
    sample_n: int = 50_000,
    chunksize: int = 200_000,
    seed: int = 42,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    files = list_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 未找到 ooi-*.csv 文件。")

    rng = np.random.default_rng(seed)
    samples = []
    total = 0

    for path in files:
        for chunk in read_ooi_chunked(path, chunksize=chunksize):
            if hasattr(chunk[COL_TIME].dtype, "tz") and chunk[COL_TIME].dtype.tz is not None:
                chunk[COL_TIME] = chunk[COL_TIME].dt.tz_convert(None)
            if start is not None:
                chunk = chunk[chunk[COL_TIME] >= start]
            if end is not None:
                chunk = chunk[chunk[COL_TIME] < end]
            chunk = chunk[chunk[COL_QC].isin(QC_GOOD_VALUES)].copy()
            if chunk.empty:
                continue

            chunk["depth_m"] = -chunk[COL_Z]
            chunk["pressure_dbar"] = chunk["depth_m"]
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=[COL_TEMP, "pressure_dbar"])
            chunk = chunk[(chunk[COL_TEMP] > -2.5) & (chunk[COL_TEMP] < 40.0)]
            if chunk.empty:
                continue

            remaining = sample_n - total
            if remaining <= 0:
                break
            n = min(len(chunk), remaining)
            idx = rng.choice(len(chunk), size=n, replace=False)
            sampled = chunk.iloc[idx]
            samples.append(sampled[["pressure_dbar", "depth_m", COL_TEMP]])
            total += n
        if total >= sample_n:
            break

    if not samples:
        raise RuntimeError("未能采样到有效的 QC=1 温度数据。")

    df = pd.concat(samples, ignore_index=True)
    pressure = torch.tensor(df["pressure_dbar"].astype(np.float32).to_numpy()).view(-1, 1)
    depth = torch.tensor(df["depth_m"].astype(np.float32).to_numpy()).view(-1, 1)
    temp = torch.tensor(df[COL_TEMP].astype(np.float32).to_numpy()).view(-1, 1)
    return pressure, depth, temp


# --------------------------------------------------------------------------------------
# Embeddings & activations
# --------------------------------------------------------------------------------------
class FourierEmbedding(nn.Module):
    def __init__(self, freqs: Iterable[float], include_input: bool = True):
        super().__init__()
        self.register_buffer("freqs", torch.tensor(list(freqs), dtype=torch.float32))
        self.include_input = include_input

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (..., 1)
        phases = inputs * self.freqs  # broadcast
        sin = torch.sin(2.0 * math.pi * phases)
        cos = torch.cos(2.0 * math.pi * phases)
        out = [sin, cos]
        if self.include_input:
            out.insert(0, inputs)
        return torch.cat(out, dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first = is_first
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


def make_activation(name: str, in_dim: int, out_dim: int, layer_idx: int, sine_w0: float):
    name = name.lower()
    if name == "sine":
        return SineLayer(in_dim, out_dim, w0=sine_w0, is_first=(layer_idx == 0))
    if name == "gelu":
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())
    if name == "tanh":
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())
    if name == "hybrid":
        # tanh first half, gelu second half
        hidden = out_dim
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
    raise ValueError(f"Unsupported activation: {name}")


# --------------------------------------------------------------------------------------
# PINN model
# --------------------------------------------------------------------------------------
class ResidualPINN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        num_heads: int,
        activation: str,
        sine_w0: float,
        hybrid_base: str,
        feature_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.activation = activation.lower()
        self.feature_dim = feature_dim
        self.hybrid_base = hybrid_base.lower()

        layers: List[nn.Module] = []
        last_dim = in_dim
        for idx in range(depth):
            act = make_activation(self.activation, last_dim, hidden, idx, sine_w0)
            layers.append(act)
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, num_heads if num_heads > 1 else 1)

        if self.hybrid_base == "eight_term":
            self.base_linear = nn.Linear(feature_dim, num_heads if num_heads > 1 else 1, bias=True)
        elif self.hybrid_base == "none":
            self.base_linear = None
        else:
            raise ValueError(f"Unknown hybrid base: {hybrid_base}")

    def forward(self, features: torch.Tensor, base_feats: torch.Tensor | None = None) -> torch.Tensor:
        resid = self.head(self.backbone(features))
        if self.base_linear is not None and base_feats is not None:
            base = self.base_linear(base_feats)
            output = base + resid
        else:
            output = resid
        return output


# --------------------------------------------------------------------------------------
# Analytical bases / helpers
# --------------------------------------------------------------------------------------
def eight_term_features(P: torch.Tensor) -> torch.Tensor:
    P_clip = torch.clamp(P, min=1.0)
    feats = [
        torch.ones_like(P),
        P,
        torch.log(P_clip),
        torch.sqrt(P_clip),
        torch.exp(-P / 500.0),
        torch.exp(-P / 1200.0),
        1.0 / (P + 50.0),
        1.0 / (P + 500.0),
    ]
    return torch.cat(feats, dim=-1)


# --------------------------------------------------------------------------------------
# Physics helper
# --------------------------------------------------------------------------------------
def physics_rhs(T_pred: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
    # Simple pseudo-PDE: dT/dP + k(P) * T = 0  (k linear in pressure)
    k = 1e-3 + 2e-6 * pressure
    return -k * T_pred


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train_pinn(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.to_datetime(args.start).tz_localize(None) if args.start else None
    end_ts = pd.to_datetime(args.end).tz_localize(None) if args.end else None

    print("加载数据…")
    P, z, T = load_ooi_samples(
        data_dir,
        sample_n=args.samples,
        seed=args.seed,
        start=start_ts,
        end=end_ts,
    )
    n_samples = len(P)
    print(f"样本数：{n_samples}")

    pressure_raw = P.clone()

    if args.multi_freq:
        freqs = [float(f.strip()) for f in args.multi_freq.split(",") if f.strip()]
    else:
        freqs = [1e-3]
    fourier = FourierEmbedding(freqs, include_input=True)

    depth_tensor = z.to(device)

    if args.normalize_input:
        p_min = float(pressure_raw.min())
        p_max = float(pressure_raw.max())
        scale = torch.tensor((p_max - p_min) * 0.5, device=device)
        offset = torch.tensor((p_max + p_min) * 0.5, device=device)
    else:
        scale = torch.tensor(1.0, device=device)
        offset = torch.tensor(0.0, device=device)

    def build_inputs(pressure_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        P_norm = (pressure_tensor - offset) / scale
        lnP = torch.log(torch.clamp(pressure_tensor, min=1.0))
        sqrtP = torch.sqrt(torch.clamp(pressure_tensor, min=0.0))
        fourier_feats = fourier(P_norm)
        feats = [P_norm, lnP, sqrtP, fourier_feats]
        if args.use_depth:
            depth_norm = (depth_tensor - depth_tensor.mean()) / (depth_tensor.std() + 1e-6)
            feats.append(depth_norm)
        net = torch.cat(feats, dim=-1)
        base = eight_term_features(pressure_tensor.detach())
        return net, base

    pressure = pressure_raw.clone().to(device)
    pressure.requires_grad_(True)
    net_input, base_feats = build_inputs(pressure)
    targets = T.to(device)
    feature_dim = net_input.shape[-1]

    # Multi-head assignment (by pressure quantiles)
    if args.num_heads > 1:
        boundaries = np.quantile(P.numpy(), np.linspace(0, 1, args.num_heads + 1))
        head_idx = np.digitize(P.numpy().ravel(), boundaries[1:-1], right=False)
    else:
        head_idx = np.zeros((n_samples,), dtype=np.int64)

    heads_tensor = torch.tensor(head_idx, dtype=torch.long, device=device)

    model = ResidualPINN(
        in_dim=feature_dim,
        hidden=args.hidden,
        depth=args.layers,
        num_heads=args.num_heads,
        activation=args.activation,
        sine_w0=args.sine_w0,
        hybrid_base=args.hybrid_base,
        feature_dim=base_feats.shape[-1],
    ).to(device)

    if args.hybrid_base != "none":
        with torch.no_grad():
            base_layer = model.base_linear
            X_np = base_feats.detach().cpu().numpy()
            y_np = targets.detach().cpu().numpy()
            if args.num_heads > 1:
                for head in range(args.num_heads):
                    mask = (head_idx == head)
                    if mask.sum() < base_feats.shape[-1]:
                        continue
                    X_h = X_np[mask]
                    y_h = y_np[mask]
                    coef, residuals, _, _ = np.linalg.lstsq(X_h, y_h, rcond=None)
                    base_layer.weight.data[head:head+1] = torch.tensor(coef.T, device=device, dtype=base_layer.weight.dtype)
                    base_layer.bias.data[head] = 0.0
            else:
                coef, residuals, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
                base_layer.weight.data = torch.tensor(coef.T, device=device, dtype=base_layer.weight.dtype)
                base_layer.bias.data.zero_()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    targets = T.to(device)

    lambda_phys = args.lambda_phys
    lambda_grad1 = args.grad1_weight
    lambda_grad2 = args.grad2_penalty
    lambda_pseudo = args.pseudo_weight
    history_total, history_data, history_phys = [], [], []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        preds_all = model(net_input.detach(), base_feats if args.hybrid_base != "none" else None)
        if args.num_heads > 1:
            preds = preds_all.gather(1, heads_tensor.view(-1, 1))
        else:
            preds = preds_all

        data_loss = torch.mean((preds - targets) ** 2)

        # Physics losses
        pressure_phys = pressure.detach().clone().requires_grad_(True)
        net_phys, base_phys = build_inputs(pressure_phys)
        preds_phys_all = model(net_phys, base_phys if args.hybrid_base != "none" else None)
        if args.num_heads > 1:
            preds_phys = preds_phys_all[:, 0:1]  # approximate using primary head
        else:
            preds_phys = preds_phys_all

        grad_outputs = torch.ones_like(preds_phys, device=device)
        dT_dP = autograd.grad(
            preds_phys,
            pressure_phys,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        phys_rhs = physics_rhs(preds_phys, pressure_phys)
        phys_loss = torch.mean((dT_dP - phys_rhs) ** 2)

        grad1_loss = torch.mean(dT_dP ** 2)

        # Second derivative
        d2T_dP2 = autograd.grad(
            dT_dP,
            pressure_phys,
            grad_outputs=torch.ones_like(dT_dP, device=device),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad2_loss = torch.mean(d2T_dP2 ** 2)

        # Smooth penalty
        smooth_penalty = torch.mean((preds_phys - preds_phys.mean()) ** 2)

        total_loss = (
            data_loss
            + lambda_phys * phys_loss
            + lambda_grad1 * grad1_loss
            + lambda_grad2 * grad2_loss
            + args.smooth_weight * smooth_penalty
        )
        total_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Adaptive lambda schedule
        if args.lambda_schedule == "adaptive" and epoch % args.adapt_every == 0:
            with torch.no_grad():
                lambda_phys = float(lambda_phys * (data_loss.item() / (phys_loss.item() + 1e-6)))
                lambda_phys = float(np.clip(lambda_phys, 1e-4, 10.0))

        history_total.append(total_loss.item())
        history_data.append(data_loss.item())
        history_phys.append(phys_loss.item())

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[{epoch:05d}/{args.epochs}] total={total_loss.item():.3e} data={data_loss.item():.3e} "
                f"phys={phys_loss.item():.3e} grad1={grad1_loss.item():.3e} grad2={grad2_loss.item():.3e} "
                f"λ_phys={lambda_phys:.3f}"
            )

    elapsed = time.time() - t0
    print(f"训练完成，用时 {elapsed:.1f}s")

    model.eval()
    with torch.no_grad():
        preds_final = model(net_input, base_feats if args.hybrid_base != "none" else None)
        preds_final = preds_final.gather(1, heads_tensor.view(-1, 1)) if args.num_heads > 1 else preds_final
        residual = preds_final - targets
        mae = torch.mean(torch.abs(residual)).item()
        rmse = torch.sqrt(torch.mean(residual ** 2)).item()
        r2 = 1.0 - torch.sum(residual ** 2).item() / (torch.sum((targets - targets.mean()) ** 2).item() + 1e-9)
        print(f"R²={r2:.5f}  MAE={mae:.5f}  RMSE={rmse:.5f}")

    # Plot losses
    steps = np.arange(1, len(history_total) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, history_total, label="total")
    plt.plot(steps, history_data, label="data")
    plt.plot(steps, history_phys, label="physics")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(fig_dir / "pinn_losses.png", dpi=150)
    plt.close()

    if args.save_curve:
        df_out = pd.DataFrame(
            {
                "pressure": P.numpy().ravel(),
                "depth": z.numpy().ravel(),
                "temp_obs": T.numpy().ravel(),
                "temp_pred": preds_final.cpu().numpy().ravel(),
            }
        )
        df_out.to_csv(data_dir / "pinn_curve_compare.csv", index=False)

    torch.save({"state_dict": model.state_dict()}, data_dir / "pinn_tp.pt")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "samples": n_samples,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "lambda_phys_final": lambda_phys,
        "config": cfg,
    }
    (data_dir / "README_pinn_tp.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="训练增强版 PINN 拟合深海 T–P 关系")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="数据目录")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--activation", type=str, default="sine")
    parser.add_argument("--sine-w0", type=float, default=30.0)
    parser.add_argument("--multi-freq", type=str, default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--use-depth", action="store_true", help="包含深度特征")
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--hybrid-base", type=str, default="eight_term", choices=["eight_term", "none"])
    parser.add_argument("--lambda-phys", type=float, default=0.1)
    parser.add_argument("--grad1-weight", type=float, default=0.01)
    parser.add_argument("--grad2-penalty", type=float, default=0.1)
    parser.add_argument("--pseudo-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=1e-3)
    parser.add_argument("--lambda-schedule", type=str, default="adaptive", choices=["adaptive", "none"])
    parser.add_argument("--adapt-every", type=int, default=200)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-curve", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_pinn(args)


if __name__ == "__main__":
    main()
