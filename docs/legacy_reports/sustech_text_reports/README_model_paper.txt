# 混合基函数 + PINN 残差：论文式结构化说明

## 摘要
- 目标：针对深海温度-压力（近似深度）关系，构建“混合基函数 + PINN 残差”的可解释且具表达力的模型，并引入物理梯度约束 `dT/dz` 以提升物理一致性。
- 方法：以 8 项混合基函数线性组合给出主趋势 `B(P; c)`，残差网络 `R(P, z; θ)` 捕获未解释的非线性细节，最终输出 `T̂ = B + R`。使用 collocation 采样和期望梯度（经验或小 MLP）形成物理残差损失，与数据损失联合优化。权重由自适应机制动态调整。
- 结果：在快速权重网格与 Top-K 搜索下，获得候选系数集；在增强训练（warmup+cosine、深层重加权、基层冻结、早停）下稳定拟合并输出模型及日志。

## 引言
- PINN 结合数据与物理约束，但常缺乏可解释先验。我们的方案在可解释的基函数上叠加 PINN 残差，兼顾低频趋势与高频细节，同时以物理残差稳住梯度行为。
- 数据来自 OOI CTD，输入包括压力 `P` 与标准化深度 `depth_norm`，以 `P≈depth_pos_m` 作为近似。

## 方法总览与系统架构
- 总输出：`T̂(P, z) = B(P; c) + R(P, z; θ)`
- 损失：`loss = λd·MSE(T̂, T) + λp·MSE((∂T̂/∂z) - physics_expect(T̂, x), 0)`
- 权重自适应：基于梯度范数比调整 `λp∈[0.05, 0.95]`，令 `λd=1-λp`。

```text
数据(OI CSV, QC=1) → 特征 x=[P, depth_norm], 目标 T
           │
           ├─ 混合基搜索(search_mix.py) → derived/mix_search.json(Top-K: basis_set+coeffs)
           │
           └─ 训练(pinn_tp_fit.py)
               ├─ 构建 B(P;c)（来自候选系数）
               ├─ 构建 R(P,z;θ)（MLP+可选Fourier/梯度增强）
               ├─ 预测: T̂ = B + R
               ├─ 物理点采样 + 自动微分: ∂T̂/∂z
               ├─ 物理期望: physics_expect(T̂, x)
               ├─ 损失: λd·MSE(T̂,T) + λp·MSE((∂T̂/∂z)-expect,0)
               ├─ 自适应权重更新 + 优化器步进
               └─ 日志/图表/权重/元数据输出
```

## 模块连接与数据传输
- 输入构造：从 CSV 读取，过滤 QC=1；`P=depth_pos_m`，`depth_norm=(depth-mean)/std`；得到 `x=[P, depth_norm]` 与目标 `T`。
- 基函数部分 `B(P;c)`：从 `derived/mix_search.json` 读取 Top-K，选择候选的 8 项基函数集合 `basis_set` 与系数 `coeffs`，输入仅依赖 `P`；输出 `(N,1)`。
- 残差网络 `R(P,z;θ)`：输入 `x`（可拼接 Fourier），输出 `(N,1)`；在启用梯度增强时，并行使用 `∂T̂/∂P` 与 `∂T̂/∂depth_norm` 校正隐藏表示。
- 组合输出：`T̂ = B + R`，与目标 `T` 计算数据损失。
- 物理约束：collocation 采样得到 `x_col`，对 `T̂(x_col)` 自动微分获取 `∂T̂/∂z`，经 `PhysicsConstraint(T̂, x_col)` 得到期望梯度，形成残差并计算物理损失。
- 权重与优化：`AdaptiveLambda` 周期性更新 `λd/λp`；优化器步进；学习率 warmup+cosine；早停保存并回载最佳状态；可选深层重加权与基层冻结增强稳定性。

## 模块说明
- 混合基函数库与线性组合 `B(P;c)`：
  - 形式：`B(P;c)=Σ_i c_i·φ_i(P)`，其中 `φ_i` 来自库的 8 项组合。
  - 典型项：常数、线性、多项式、`log(P+δ)`、`exp(-P/β)`（如 `14.8832·exp(-P/1200)`）、分式 `1/(P+γ)`、`tanh(P/κ)` 等。
  - 解释性：分项刻画压力响应、曲率与渐近行为，系数由 Top-K 搜索确定。
- 残差网络 `R(P,z;θ)`（MLP）：
  - 结构：宽度 64，层数 5，激活 `Tanh`，输出 `(N,1)`；可选 Fourier 与梯度增强。
  - 梯度增强：并行使用输入梯度分量修正隐藏表示，提升梯度敏感区拟合。
- Fourier 特征编码：
  - 将输入 `x` 映射到 `sin/cos` 展开（固定频率矩阵作为 buffer），与原始输入拼接以提升高频表达力。
- 物理约束 `PhysicsConstraint`：
  - `empirical`：`dT/dz=-α·tanh(P/scale)-β·(T-t_ref)`，参数可学习；
  - `mlp`：从 `(T, P, depth_norm)` 直接预测 `dT/dz`。
- 自适应权重 `AdaptiveLambda`：
  - 比值：`ratio=||∇loss_data||/||∇loss_phys||`；
  - 更新：`λp←clip(λp·ratio^rate,0.05,0.95)`，`λd=1-λp`。
- collocation 采样：
  - `uniform/latin` 两种方式；数量 `batch_size·collocation_ratio`；在 `depth_range` 内采样得到 `x_col`。
- 训练器（增强）：
  - warmup+cosine 调度；早停；深层重加权；基层冻结；并行候选/种子训练。

## 训练目标与优化流程
- 数据损失：`loss_data=MSE(T̂, T)`；可对深层样本重加权。
- 物理损失：`loss_phys=MSE((∂T̂/∂z)-physics_expect, 0)`。
- 总损失：`loss=λd·loss_data+λp·loss_phys`。
- 步骤：前向 → 自动微分 → 权重更新（每 `adapt_every` 步）→ 反向与优化器步进 → 调度与早停。

## 复现配置与超参数
- 混合搜索（快速示例）：
```bash
python search_mix.py --limit-files 5 --max-per-band 4000 --topk 5 \
  --objective weighted_mae --out derived/mix_search.json
```
- 训练（Top-1 + 增强）：
```bash
python pinn_tp_fit.py --mix-json derived/mix_search.json --candidate-index 0 \
  --epochs 4000 --batch-size 2048 --max-samples 80000 --collocation-ratio 2.0 \
  --optimizer adam --lr 1e-4 --use-fourier --fourier-freqs 4 --use-gradient-block \
  --physics-mode empirical --collocation-method latin --warmup cosine \
  --freeze-base true --deep-weighting true --patience 400 --seed 42 --device cpu
```

## 实验记录与输出
- 日志：`pinn_train.log`（损失、权重、梯度范数等）。
- 模型与元数据：`data/pinn_tp.pt` 与相关 JSON。
- 图表：拟合曲线与残差分层等输出至 `figures/`。
- 搜索结果：`derived/mix_search.json`（Top-K 候选、分箱权重与指标）。

## 讨论与局限
- 搜索在快速参数下不同候选的 8 点集合与指标可能相近，需全量搜索细分；
- 经验物理约束简洁但可能忽略局地异常，可与 MLP 约束融合；
- 深层重加权提升稳定性但需防止过度拟合深层样本；
- 冻结策略有助收敛，但需结合 warmup 与调度谨慎调节窗口。

## 结论
- 我们以混合基函数承载可解释主趋势，并以 PINN 残差补足复杂细节，通过物理残差与自适应权重实现数据与物理的协同优化，在增强训练与合理搜索配置下获得稳定且具可解释性的拟合结果。