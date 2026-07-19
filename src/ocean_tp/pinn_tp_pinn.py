#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN 训练脚本：拟合深海温度-压力（近似深度）关系，同时融入物理约束。

核心功能：
1. 加载 OOI CTD CSV 数据（同 analyze_ooi_data.py），抽样构建观测点与物理约束点。
2. 定义可扩展 PINN 模型：
   - 支持 Fourier 特征编码；
   - 支持物理先验层（经验公式 + 网络残差）；
   - 提供物理约束函数（经验型或小型 MLP）。
3. 训练循环包含自适应损失权重（基于梯度范数比值）与早停逻辑。
4. 输出训练日志、保存模型参数，并可选生成验证曲线与残差可视化。

运行示例：
    python pinn_tp_pinn.py --data-dir . --epochs 4000 --batch-size 2048 \
        --max-samples 80000 --fourier-freqs 5 --use-prior
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


COL_TIME = "time"
COL_TEMP = "sea_water_temperature_profiler_depth_enabled"
COL_QC = "sea_water_temperature_profiler_depth_enabled_qc_agg"
COL_Z = "z"  # 单位 m，向下为负
QC_GOOD_VALUES = {1}


# ---------------------------------------------------------------------------
# 数据加载与抽样
# ---------------------------------------------------------------------------

def read_ooi_chunked(path: Path, chunksize: int = 200_000):
    """与 analyze_ooi_data.py 保持一致的分块读取方式。"""
    return pd.read_csv(
        path,
        header=0,
        skiprows=[1],
        usecols=[COL_TIME, COL_TEMP, COL_QC, COL_Z],
        parse_dates=[COL_TIME],
        infer_datetime_format=True,
        chunksize=chunksize,
    )


def list_csv_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.glob("ooi-*.csv"))
    return list(files)


@dataclass
class ObservationBatch:
    features: torch.Tensor  # shape (N, input_dim)
    targets: torch.Tensor  # shape (N, 1)
    depth_range: Tuple[float, float]


def load_observations(
    files: List[Path],
    max_samples: int = 100_000,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> ObservationBatch:
    """
    抽样 QC=1 的观测数据，构建 PINN 的数据损失样本。
    输入特征默认为压力（近似深度，正值）与归一化深度。
    """
    rng = np.random.default_rng(seed)
    rows = []
    total = 0
    depth_min, depth_max = math.inf, -math.inf

    for path in files:
        for chunk in read_ooi_chunked(path):
            chunk = chunk[chunk[COL_QC].isin(QC_GOOD_VALUES)].copy()
            if chunk.empty:
                continue
            chunk["depth_pos_m"] = -chunk[COL_Z]
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if chunk.empty:
                continue

            depth_min = min(depth_min, chunk["depth_pos_m"].min())
            depth_max = max(depth_max, chunk["depth_pos_m"].max())

            remaining = max_samples - total
            if remaining <= 0:
                break
            n = min(len(chunk), remaining)
            indices = rng.choice(len(chunk), size=n, replace=False)
            sampled = chunk.iloc[indices][["depth_pos_m", COL_TEMP]].copy()
            rows.append(sampled)
            total += n
        if total >= max_samples:
            break

    if not rows:
        raise RuntimeError("未能加载任何 QC=1 的观测样本。")

    df = pd.concat(rows, ignore_index=True)
    depth = df["depth_pos_m"].to_numpy(dtype=np.float32)
    temp = df[COL_TEMP].to_numpy(dtype=np.float32)

    # 输入特征：压力约等于深度（单位 dbar），以及归一化深度
    pressure = depth.copy()  # 约等于深度
    depth_norm = (depth - depth.mean()) / (depth.std() + 1e-6)
    features = np.stack([pressure, depth_norm], axis=1)

    features_tensor = torch.from_numpy(features).to(device)
    targets_tensor = torch.from_numpy(temp[:, None]).to(device)

    return ObservationBatch(
        features=features_tensor,
        targets=targets_tensor,
        depth_range=(float(depth.min()), float(depth.max())),
    )


def sample_collocation_points(
    depth_range: Tuple[float, float],
    n_samples: int,
    device: torch.device | str = "cpu",
    method: str = "uniform",
) -> torch.Tensor:
    """采样物理约束点（仅需深度），当前使用一维深度坐标。"""
    z_min, z_max = depth_range
    if method == "uniform":
        depths = torch.rand(n_samples, device=device) * (z_max - z_min) + z_min
    elif method == "latin":
        # 拉丁超立方采样（简化版）
        cut = torch.linspace(0.0, 1.0, steps=n_samples + 1, device=device)
        u = torch.rand(n_samples, device=device)
        pts = cut[:-1] + (cut[1:] - cut[:-1]) * u
        depths = pts * (z_max - z_min) + z_min
    else:
        raise ValueError(f"未知采样方法: {method}")

    pressure = depths.clone()
    depth_norm = (depths - depths.mean()) / (depths.std() + 1e-6)
    return torch.stack([pressure, depth_norm], dim=1)


# ---------------------------------------------------------------------------
# 特征编码与模型组件
# ---------------------------------------------------------------------------

class FourierFeatureEncoding(nn.Module):
    """对输入进行傅里叶特征映射，以捕获高频结构。"""

    def __init__(self, in_dim: int, num_frequencies: int = 4, sigma: float = 1.0):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies 必须为正整数。")
        # 固定随机频率，可视为常量参数
        B = torch.randn(in_dim, num_frequencies, dtype=torch.float32) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)，输出 sin/cos 展开
        proj = 2 * math.pi * x @ self.B  # (N, num_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class EmpiricalPrior(nn.Module):
    """
    海水温度经验先验层：T_prior = a + b * depth + c * log(depth + d)
    其中 depth 为压力近似（正值）。参数可选是否训练。
    """

    def __init__(self, init_params: Tuple[float, float, float, float], trainable: bool = True):
        super().__init__()
        a, b, c, d = init_params
        if trainable:
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
            self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor(d, dtype=torch.float32))
        else:
            self.register_buffer("a", torch.tensor(a, dtype=torch.float32))
            self.register_buffer("b", torch.tensor(b, dtype=torch.float32))
            self.register_buffer("c", torch.tensor(c, dtype=torch.float32))
            self.register_buffer("d", torch.tensor(d, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[..., 0] 视为压力/深度（正值）
        depth = torch.clamp(x[..., 0], min=1.0)
        offset = torch.abs(self.d) if isinstance(self.d, torch.Tensor) else abs(self.d)
        return self.a + self.b * depth + self.c * torch.log(depth + offset)


class PhysicsConstraint(nn.Module):
    """
    提供 dT/dz 的物理期望值。

    模式：
    - "empirical": 经验公式 dT/dz = -alpha * tanh(depth/scale) - beta * (T - T_ref)
      其中 alpha/beta/scale/T_ref 为可学习参数。
    - "mlp": 小型网络从 (T, depth_norm, pressure) 预测 dT/dz。
    """

    def __init__(self, mode: str = "empirical"):
        super().__init__()
        self.mode = mode
        if mode == "empirical":
            self.alpha = nn.Parameter(torch.tensor(1e-3))
            self.beta = nn.Parameter(torch.tensor(5e-4))
            self.scale = nn.Parameter(torch.tensor(500.0))
            self.t_ref = nn.Parameter(torch.tensor(2.0))
        elif mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(3, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
        else:
            raise ValueError(f"未知 PhysicsConstraint 模式: {mode}")

    def forward(self, T: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        depth = torch.clamp(features[..., 0], min=0.0)
        depth_norm = features[..., 1]
        if self.mode == "empirical":
            grad = -self.alpha * torch.tanh(depth / (torch.abs(self.scale) + 1e-3))
            grad -= self.beta * (T.squeeze(-1) - self.t_ref)
            return grad.unsqueeze(-1)
        else:
            inp = torch.cat([T, depth.unsqueeze(-1), depth_norm.unsqueeze(-1)], dim=-1)
            return self.net(inp)


class GradientEnhancedBlock(nn.Module):
    """
    梯度增强模块：并行分支学习隐层梯度信息并融合。
    参考文献中常用于提升敏感区域拟合能力。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        grad_pressure: torch.Tensor,
        grad_depth_norm: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([h, grad_pressure, grad_depth_norm], dim=-1)
        return self.block(concat)


class PINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 5,
        use_fourier: bool = False,
        fourier_freqs: int = 4,
        fourier_sigma: float = 0.5,
        use_prior: bool = False,
        prior_params: Tuple[float, float, float, float] = (4.0, -1e-3, -0.2, 10.0),
        use_gradient_block: bool = False,
    ):
        super().__init__()
        self.use_prior = use_prior
        self.use_gradient_block = use_gradient_block

        self.fourier = (
            FourierFeatureEncoding(in_dim, fourier_freqs, fourier_sigma)
            if use_fourier
            else None
        )
        fourier_dim = 0 if self.fourier is None else fourier_freqs * 2
        input_dim = in_dim + fourier_dim

        feature_layers: List[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            feature_layers.append(nn.Linear(dims[i], dims[i + 1]))
            feature_layers.append(nn.Tanh())
        # 移除末尾多余的 Tanh
        feature_layers = feature_layers[:-1]
        self.feature_net = nn.Sequential(*feature_layers)
        self.output_layer = nn.Linear(dims[-1], 1)

        self.prior = EmpiricalPrior(prior_params, trainable=True) if use_prior else None
        if use_gradient_block:
            self.grad_block = GradientEnhancedBlock(hidden_dim)
        else:
            self.grad_block = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        features = x
        if self.fourier is not None:
            features = torch.cat([features, self.fourier(x)], dim=-1)

        hidden = self.feature_net(features)
        out = self.output_layer(hidden)
        if self.use_gradient_block:
            grads = torch.autograd.grad(
                out,
                x,
                torch.ones_like(out),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grads is None:
                grads = torch.zeros_like(x)
            grad_pressure = grads[..., 0:1]
            grad_depth_norm = grads[..., 1:2]
            enhanced = self.grad_block(hidden, grad_pressure, grad_depth_norm)
            out = out + enhanced

        if self.prior is not None:
            out = self.prior(x) + out

        return out


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    data_dir: Path
    device: torch.device
    epochs: int = 5000
    batch_size: int = 2048
    collocation_ratio: float = 2.0
    max_samples: int = 80_000
    lr: float = 1e-4
    optimizer: str = "adam"
    lambda_data: float = 0.5
    lambda_phys: float = 0.5
    adapt_rate: float = 0.5
    adapt_every: int = 50
    patience: int = 300
    fourier_freqs: int = 4
    fourier_sigma: float = 0.5
    use_fourier: bool = False
    use_prior: bool = False
    use_gradient_block: bool = False
    physics_mode: str = "empirical"
    collocation_method: str = "latin"
    seed: int = 42


class AdaptiveLambda:
    """自适应调整数据/物理损失权重。"""

    def __init__(self, lambda_data: float, lambda_phys: float, rate: float, eps: float = 1e-6):
        total = lambda_data + lambda_phys
        self.lambda_data = max(lambda_data / total, 1e-3)
        self.lambda_phys = max(lambda_phys / total, 1e-3)
        self.rate = rate
        self.eps = eps

    def update(self, grad_data_norm: float, grad_phys_norm: float):
        ratio = (grad_data_norm + self.eps) / (grad_phys_norm + self.eps)
        self.lambda_phys *= ratio ** self.rate
        self.lambda_phys = float(np.clip(self.lambda_phys, 0.05, 0.95))
        self.lambda_data = 1.0 - self.lambda_phys


class PINNTrainer:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = config.device

        files = list_csv_files(config.data_dir)
        if not files:
            raise FileNotFoundError(f"在 {config.data_dir} 未找到 ooi-*.csv 数据文件。")

        obs = load_observations(
            files,
            max_samples=config.max_samples,
            seed=config.seed,
            device=self.device,
        )
        self.depth_range = obs.depth_range
        self.obs_dataset = TensorDataset(obs.features, obs.targets)
        self.data_loader = DataLoader(
            self.obs_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.model = PINN(
            in_dim=obs.features.shape[1],
            hidden_dim=64,
            num_layers=5,
            use_fourier=config.use_fourier,
            fourier_freqs=config.fourier_freqs,
            fourier_sigma=config.fourier_sigma,
            use_prior=config.use_prior,
            use_gradient_block=config.use_gradient_block,
        ).to(self.device)
        self.physics = PhysicsConstraint(mode=config.physics_mode).to(self.device)
        self.lambda_adaptor = AdaptiveLambda(
            lambda_data=config.lambda_data,
            lambda_phys=config.lambda_phys,
            rate=config.adapt_rate,
        )
        if config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        elif config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        else:
            raise ValueError(f"不支持的优化器：{config.optimizer}")

        self.best_loss = math.inf
        self.best_state = None
        self.no_improve_steps = 0

    def physics_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clone().detach().requires_grad_(True)
        preds = self.model(inputs)
        grads = torch.autograd.grad(
            preds,
            inputs,
            torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
        )[0]
        dT_dz = grads[:, 0:1]  # 压力近似深度
        physics_expect = self.physics(preds, inputs)
        return dT_dz - physics_expect

    def train(self):
        cfg = self.cfg
        collocation_n = int(cfg.batch_size * cfg.collocation_ratio)
        data_iter = iter(self.data_loader)

        for step in range(1, cfg.epochs + 1):
            try:
                features, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                features, targets = next(data_iter)
            features = features.to(self.device).requires_grad_(True)
            targets = targets.to(self.device)

            collocation = sample_collocation_points(
                self.depth_range,
                collocation_n,
                device=self.device,
                method=cfg.collocation_method,
            ).requires_grad_(True)

            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(features)
            loss_data = torch.mean((preds - targets) ** 2)

            residual = self.physics_residual(collocation)
            loss_phys = torch.mean(residual ** 2)

            # 计算梯度范数以自适应权重
            first_param = next(self.model.parameters())
            grad_data = torch.autograd.grad(
                loss_data,
                first_param,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grad_phys = torch.autograd.grad(
                loss_phys,
                first_param,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grad_data_norm = grad_data.norm().item() if grad_data is not None else 0.0
            grad_phys_norm = grad_phys.norm().item() if grad_phys is not None else 0.0

            if cfg.adapt_every > 0 and step % cfg.adapt_every == 0:
                self.lambda_adaptor.update(grad_data_norm, grad_phys_norm)

            λd = self.lambda_adaptor.lambda_data
            λp = self.lambda_adaptor.lambda_phys
            loss = λd * loss_data + λp * loss_phys

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                total_loss = loss.item()
                if total_loss + 1e-6 < self.best_loss:
                    self.best_loss = total_loss
                    self.best_state = {
                        "model": self.model.state_dict(),
                        "physics": self.physics.state_dict(),
                        "step": step,
                        "loss": total_loss,
                    }
                    self.no_improve_steps = 0
                else:
                    self.no_improve_steps += 1

            if step % 100 == 0 or step == 1:
                print(
                    f"[{step:05d}/{cfg.epochs}] "
                    f"loss={loss.item():.5f} "
                    f"data={loss_data.item():.5f} "
                    f"phys={loss_phys.item():.5f} "
                    f"λd={λd:.3f} λp={λp:.3f} "
                    f"|grad_d|={grad_data_norm:.4f} |grad_p|={grad_phys_norm:.4f}"
                )

            if cfg.patience > 0 and self.no_improve_steps >= cfg.patience:
                print(
                    f"早停触发：{cfg.patience} 步无改进，最佳 loss={self.best_loss:.6f} "
                    f"@ step {self.best_state['step'] if self.best_state else 'N/A'}"
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model"])
            self.physics.load_state_dict(self.best_state["physics"])
        return self.best_state

    def evaluate(self, n_eval: int = 5000) -> dict:
        self.model.eval()
        with torch.no_grad():
            dataset = self.obs_dataset
            idx = torch.randperm(len(dataset))[:n_eval]
            features = dataset.tensors[0][idx].to(self.device)
            targets = dataset.tensors[1][idx].to(self.device)
            preds = self.model(features)
            mse = torch.mean((preds - targets) ** 2).item()
            mae = torch.mean((preds - targets).abs()).item()
        self.model.train()
        return {"mse": mse, "mae": mae}

    def save(self, out_dir: Path, extra: Optional[dict] = None):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "pinn_model.pt")
        torch.save(self.physics.state_dict(), out_dir / "physics_module.pt")
        meta = {
            "config": self.cfg.__dict__,
            "best_loss": self.best_loss,
            "best_state": self.best_state["step"] if self.best_state else None,
        }
        if extra:
            meta.update(extra)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="训练 PINN 拟合深海 T-P 关系")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="包含 ooi-*.csv 的目录")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=80_000)
    parser.add_argument("--collocation-ratio", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lambda-data", type=float, default=0.5)
    parser.add_argument("--lambda-phys", type=float, default=0.5)
    parser.add_argument("--adapt-rate", type=float, default=0.5)
    parser.add_argument("--adapt-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--fourier-freqs", type=int, default=4)
    parser.add_argument("--fourier-sigma", type=float, default=0.5)
    parser.add_argument("--use-fourier", action="store_true")
    parser.add_argument("--use-prior", action="store_true")
    parser.add_argument("--use-gradient-block", action="store_true")
    parser.add_argument("--physics-mode", choices=["empirical", "mlp"], default="empirical")
    parser.add_argument("--collocation-method", choices=["uniform", "latin"], default="latin")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("pinn_outputs"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    return TrainingConfig(
        data_dir=args.data_dir.resolve(),
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        collocation_ratio=args.collocation_ratio,
        max_samples=args.max_samples,
        lr=args.lr,
        optimizer=args.optimizer,
        lambda_data=args.lambda_data,
        lambda_phys=args.lambda_phys,
        adapt_rate=args.adapt_rate,
        adapt_every=args.adapt_every,
        patience=args.patience,
        fourier_freqs=args.fourier_freqs,
        fourier_sigma=args.fourier_sigma,
        use_fourier=args.use_fourier,
        use_prior=args.use_prior,
        use_gradient_block=args.use_gradient_block,
        physics_mode=args.physics_mode,
        collocation_method=args.collocation_method,
        seed=args.seed,
    )


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    trainer = PINNTrainer(cfg)
    best_state = trainer.train()
    metrics = trainer.evaluate()
    print(f"评估指标：MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
    trainer.save(cfg.data_dir / cfg.output_dir, extra={"metrics": metrics, "best_state": best_state})


if __name__ == "__main__":
    main()
