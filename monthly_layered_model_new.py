
#!/usr/bin/env python3
"""
月份分层模型：结合月份建模和深度分层来提高精度

核心思想：
1. 根据月份划分数据 (1-12月)
2. 为每个月份训练独立的分层模型
3. 使用集成方法结合预测结果
4. 针对浅层采用更精细的建模策略
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as cheb

from time_season_validation_new import load_all_data

from compute_depth_layers import dynamic_partition, prepare_samples, aggregate_bins, compute_prefix_arrays

warnings.filterwarnings("ignore")

def mean_absolute_error(y_true, y_pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """计算R²分数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

class SimpleRandomForest:
    """简单的随机森林实现"""
    
    def __init__(self, n_estimators=10, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        np.random.seed(random_state)
    
    def fit(self, X, y):
        """训练随机森林"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            n_samples = len(X)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 训练决策树（简化为多项式拟合）
            try:
                degree = min(3, len(np.unique(X_boot)) - 1)
                if degree > 0:
                    coeffs = np.polyfit(X_boot.flatten(), y_boot, degree)
                    self.trees.append({'type': 'poly', 'coeffs': coeffs})
                else:
                    # 如果数据点太少，使用平均值
                    self.trees.append({'type': 'mean', 'value': np.mean(y_boot)})
            except:
                self.trees.append({'type': 'mean', 'value': np.mean(y_boot)})
    
    def predict(self, X):
        """预测"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        predictions = []
        
        for tree in self.trees:
            if tree['type'] == 'poly':
                pred = np.polyval(tree['coeffs'], X.flatten())
            else:
                pred = np.full(len(X), tree['value'])
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)

@dataclass
class SeasonalLayerConfig:
    """分层配置 (复用 SeasonalLayerConfig 名称)"""
    layers: int = 4
    min_thickness: float = 300.0
    min_samples_per_layer: int = 1000
    bin_width: float = 1.0
    val_fraction: float = 0.2
    ensemble_method: str = 'weighted'  # 'simple', 'weighted', 'stacking'
    use_depth_weighting: bool = True
    shallow_layer_boost: float = 1.5  # 浅层权重提升

class MonthlyLayeredModel:
    """月份分层模型"""
    
    def __init__(self, config: SeasonalLayerConfig):
        self.config = config
        self.monthly_models = {}
        self.global_stats = {}
        
    def extract_monthly_data(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """提取月份数据"""
        # 确保时间列正确解析
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
            
        df['month'] = df['time'].dt.month
        
        monthly_data = {}
        for month in range(1, 13):
            monthly_data[month] = df[df['month'] == month]
            # print(f"Month {month}: {len(monthly_data[month])} samples")
            
        return monthly_data
    
    def compute_optimal_boundaries(self, pressure: np.ndarray, temperature: np.ndarray,
                                 num_layers: int, min_thickness: float, 
                                 min_samples: int, bin_width: float) -> List[float]:
        """计算最优分层边界"""
        # 创建临时DataFrame
        df = pd.DataFrame({
            'depth_pos': np.abs(pressure),
            'temp': temperature
        })
        
        # 聚合到bins
        min_depth = math.floor(df["depth_pos"].min() / bin_width) * bin_width
        max_depth = math.ceil(df["depth_pos"].max() / bin_width) * bin_width + bin_width
        edges = np.arange(min_depth, max_depth + bin_width, bin_width)
        
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
        
        # 计算前缀数组
        counts = grouped["count"].to_numpy(dtype=np.int64)
        temp_sum = grouped["temp_sum"].to_numpy(dtype=np.float64)
        temp_sq = grouped["temp_sq"].to_numpy(dtype=np.float64)
        prefix_count = np.concatenate([[0], np.cumsum(counts)])
        prefix_sum = np.concatenate([[0.0], np.cumsum(temp_sum)])
        prefix_sq = np.concatenate([[0.0], np.cumsum(temp_sq)])
        
        # 使用动态规划计算边界
        boundaries = dynamic_partition(edges, prefix_count, prefix_sum, prefix_sq)
        
        return boundaries
    
    def fit_chebyshev_layer(self, pressure: np.ndarray, temperature: np.ndarray,
                          depth_min: float, depth_max: float, degree: int = 5):
        """拟合Chebyshev多项式"""
        # 归一化到[-1, 1]
        p_norm = 2 * (pressure - depth_min) / (depth_max - depth_min) - 1
        
        # 拟合Chebyshev多项式
        coeffs = cheb.chebfit(p_norm, temperature, degree)
        
        def predict_func(p):
            p_norm_new = 2 * (p - depth_min) / (depth_max - depth_min) - 1
            return cheb.chebval(p_norm_new, coeffs)
            
        return predict_func

    def train_monthly_layer_model(self, month_df: pd.DataFrame, month: int) -> Dict:
        """训练单个月份的分层模型"""
        print(f"\n训练 Month {month} 分层模型...")
        
        # 准备数据
        pressure = month_df['depth'].values
        temperature = month_df['temp'].values
        
        # 计算最优分层边界
        boundaries = self.compute_optimal_boundaries(
            pressure, temperature,
            num_layers=self.config.layers,
            min_thickness=self.config.min_thickness,
            min_samples=self.config.min_samples_per_layer,
            bin_width=self.config.bin_width
        )
        
        layer_models = {}
        layer_metrics = {}
        
        for i in range(len(boundaries) - 1):
            layer_name = f"L{i+1}"
            depth_min, depth_max = boundaries[i], boundaries[i+1]
            
            # 提取层内数据
            layer_mask = (np.abs(pressure) >= depth_min) & (np.abs(pressure) < depth_max)
            layer_p = np.abs(pressure[layer_mask])
            layer_t = temperature[layer_mask]
            
            if len(layer_p) < self.config.min_samples_per_layer:
                print(f"警告: Month {month} {layer_name} 样本数不足: {len(layer_p)}")
                continue
            
            # 训练/验证分割
            n_val = int(len(layer_p) * self.config.val_fraction)
            indices = np.random.permutation(len(layer_p))
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            
            # 多模型集成
            models = {}
            
            # 1. Chebyshev拟合
            try:
                cheb_model = self.fit_chebyshev_layer(
                    layer_p[train_idx], layer_t[train_idx],
                    depth_min, depth_max
                )
                models['chebyshev'] = cheb_model
            except Exception as e:
                print(f"Chebyshev拟合失败: {e}")
            
            # 2. 随机森林（处理非线性）
            try:
                rf_model = SimpleRandomForest(
                    n_estimators=20, 
                    max_depth=5,
                    random_state=42
                )
                rf_model.fit(layer_p[train_idx], layer_t[train_idx])
                models['random_forest'] = rf_model
            except Exception as e:
                print(f"随机森林拟合失败: {e}")
            
            # 3. 多项式拟合（浅层使用更高阶）
            try:
                degree = 5 if i == 0 else 3  # 浅层使用更高阶多项式
                poly_coeffs = np.polyfit(layer_p[train_idx], layer_t[train_idx], degree)
                models['polynomial'] = {'coeffs': poly_coeffs, 'degree': degree}
            except Exception as e:
                print(f"多项式拟合失败: {e}")
            
            # 验证并选择最佳模型
            best_model = None
            best_mae = float('inf')
            model_scores = {}
            
            # 如果没有成功训练任何模型，设置默认值
            if not models:
                # 添加一个简单的平均值模型作为兜底
                try:
                    mean_val = np.mean(layer_t[train_idx])
                    models['mean'] = {'value': mean_val}
                except:
                    models['mean'] = {'value': 2.0} # 绝对兜底

            for model_name, model in models.items():
                try:
                    if model_name == 'chebyshev':
                        pred = np.array([model(p) for p in layer_p[val_idx]])
                    elif model_name == 'random_forest':
                        pred = model.predict(layer_p[val_idx])
                    elif model_name == 'polynomial':
                        pred = np.polyval(model['coeffs'], layer_p[val_idx])
                    elif model_name == 'mean':
                        pred = np.full_like(layer_p[val_idx], model['value'])
                    
                    mae = mean_absolute_error(layer_t[val_idx], pred)
                    rmse = np.sqrt(mean_squared_error(layer_t[val_idx], pred))
                    r2 = r2_score(layer_t[val_idx], pred)
                    
                    model_scores[model_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_model = (model_name, model)
                        
                except Exception as e:
                    print(f"模型 {model_name} 验证失败: {e}")
            
            layer_models[layer_name] = {
                'best_model': best_model,
                'all_models': models,
                'scores': model_scores,
                'boundaries': (depth_min, depth_max),
                'sample_size': len(layer_p)
            }
            
            layer_metrics[layer_name] = {
                'depth_range': (depth_min, depth_max),
                'sample_size': len(layer_p),
                'best_mae': best_mae,
                'model_scores': model_scores
            }
            
            print(f"  {layer_name} ({depth_min:.0f}-{depth_max:.0f}m): "
                  f"最佳模型={best_model[0] if best_model else 'None'}, "
                  f"MAE={best_mae:.4f}°C")
        
        return {
            'boundaries': boundaries,
            'layer_models': layer_models,
            'layer_metrics': layer_metrics,
            'month': month
        }
    
    def train(self, data_dir: Path, samples: int = 100000):
        """训练月份分层模型"""
        print(f"加载数据 (limit {samples})...")
        df = load_all_data(str(data_dir))
        if len(df) > samples:
            df = df.sample(n=samples, random_state=42)
        
        print("提取月份数据...")
        monthly_data = self.extract_monthly_data(df)
        
        # 为每个月份训练模型
        for month, month_df in monthly_data.items():
            if len(month_df) < 1000:
                print(f"跳过 Month {month}：样本数不足")
                continue
                
            self.monthly_models[month] = self.train_monthly_layer_model(month_df, month)
        
        # 计算全局统计
        self.global_stats = self._compute_global_stats(monthly_data)
        
    def predict(self, pressure: np.ndarray, month: int = None, 
                datetime_info: pd.DataFrame = None) -> np.ndarray:
        """预测温度"""
        if month is None and datetime_info is not None:
            # 自动判断月份
            month = datetime_info['month'].iloc[0] if len(datetime_info) > 0 else 6
        
        if month not in self.monthly_models:
            # 使用集成预测
            return self._ensemble_predict(pressure)
        
        model_info = self.monthly_models[month]
        
        # 确保 model_info 是字典且不为 None
        if not model_info or not isinstance(model_info, dict):
            return self._ensemble_predict(pressure)
            
        predictions = np.zeros_like(pressure, dtype=float)
        
        # 按层预测
        boundaries = model_info['boundaries']
        layer_models = model_info['layer_models']
        
        for i, (layer_name, layer_info) in enumerate(layer_models.items()):
            depth_min, depth_max = layer_info['boundaries']
            layer_mask = (np.abs(pressure) >= depth_min) & (np.abs(pressure) < depth_max)
            
            if not np.any(layer_mask):
                continue
            
            layer_p = np.abs(pressure[layer_mask])
            best_model_name, best_model = layer_info['best_model']
            
            # 如果最佳模型为 None，跳过
            if best_model is None:
                continue

            try:
                if best_model_name == 'chebyshev':
                    layer_pred = np.array([best_model(p) for p in layer_p])
                elif best_model_name == 'random_forest':
                    layer_pred = best_model.predict(layer_p)
                elif best_model_name == 'polynomial':
                    layer_pred = np.polyval(best_model['coeffs'], layer_p)
                elif best_model_name == 'mean':
                    layer_pred = np.full_like(layer_p, best_model['value'])
                
                predictions[layer_mask] = layer_pred
                
            except Exception as e:
                print(f"预测失败 Month {month} {layer_name}: {e}")
                # 使用简单线性插值作为后备
                predictions[layer_mask] = np.interp(layer_p, [depth_min, depth_max], [4.0, 1.5])
        
        return predictions

    def _ensemble_predict(self, pressure: np.ndarray) -> np.ndarray:
        """集成所有月份的预测结果"""
        all_predictions = []
        weights = []
        
        for month, model_info in self.monthly_models.items():
            # 跳过无效模型
            if not model_info or not isinstance(model_info, dict):
                continue
                
            try:
                # 直接调用内部预测逻辑
                pred = np.zeros_like(pressure, dtype=float)
                boundaries = model_info['boundaries']
                layer_models = model_info['layer_models']
                
                for layer_name, layer_info in layer_models.items():
                    depth_min, depth_max = layer_info['boundaries']
                    layer_mask = (np.abs(pressure) >= depth_min) & (np.abs(pressure) < depth_max)
                    
                    if not np.any(layer_mask):
                        continue
                        
                    layer_p = np.abs(pressure[layer_mask])
                    best_model_name, best_model = layer_info['best_model']
                    
                    if best_model is None:
                        continue
                        
                    if best_model_name == 'chebyshev':
                        layer_pred = np.array([best_model(p) for p in layer_p])
                    elif best_model_name == 'random_forest':
                        layer_pred = best_model.predict(layer_p)
                    elif best_model_name == 'polynomial':
                        layer_pred = np.polyval(best_model['coeffs'], layer_p)
                    elif best_model_name == 'mean':
                        layer_pred = np.full_like(layer_p, best_model['value'])
                    else:
                        layer_pred = np.interp(layer_p, [depth_min, depth_max], [4.0, 1.5])
                        
                    pred[layer_mask] = layer_pred
                
                all_predictions.append(pred)
                
                # 基于模型质量计算权重
                avg_mae = np.mean([
                    layer['best_mae'] for layer in model_info['layer_metrics'].values()
                ])
                weights.append(1.0 / (avg_mae + 1e-6))
                
            except Exception as e:
                print(f"集成预测失败 Month {month}: {e}")
        
        if not all_predictions:
            return np.full_like(pressure, 2.5)  # 默认值
        
        # 加权平均
        all_predictions = np.array(all_predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(all_predictions, axis=0, weights=weights)

    def _compute_global_stats(self, monthly_data: Dict[int, pd.DataFrame]) -> Dict:
        """计算全局统计信息"""
        all_temps = []
        all_pressures = []
        
        for month_df in monthly_data.values():
            all_temps.extend(month_df['temp'].values)
            all_pressures.extend(month_df['depth'].values)
        
        return {
            'temp_mean': np.mean(all_temps),
            'temp_std': np.std(all_temps),
            'pressure_mean': np.mean(all_pressures),
            'pressure_std': np.std(all_pressures),
            'total_samples': len(all_temps)
        }
    
    def evaluate(self, data_dir: Path, test_samples: int = 10000) -> Dict:
        """评估模型性能"""
        print("加载测试数据...")
        df = load_all_data(str(data_dir))
        if len(df) > test_samples:
            df = df.sample(n=test_samples, random_state=42)
        
        monthly_test_data = self.extract_monthly_data(df)
        
        results = {}
        
        for month, month_df in monthly_test_data.items():
            if month not in self.monthly_models or len(month_df) < 100:
                continue
            
            pressure = month_df['depth'].values
            true_temp = month_df['temp'].values
            
            # 预测
            pred_temp = self.predict(pressure, month)
            
            # 计算指标
            mae = mean_absolute_error(true_temp, pred_temp)
            rmse = np.sqrt(mean_squared_error(true_temp, pred_temp))
            r2 = r2_score(true_temp, pred_temp)
            
            results[month] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'sample_size': len(true_temp)
            }
            
            print(f"Month {month}: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C, R²={r2:.4f}")
        
        return results
    
    def save_model(self, filepath: Path):
        """保存模型"""
        model_data = {
            'config': self.config.__dict__,
            'monthly_models': self.monthly_models,
            'global_stats': self.global_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        print(f"模型已保存到: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="训练月份分层模型")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--min-thickness", type=float, default=300.0)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("monthly_layered_model.json"))
    
    args = parser.parse_args()
    
    # 配置
    config = SeasonalLayerConfig(
        layers=args.layers,
        min_thickness=args.min_thickness,
        min_samples_per_layer=args.min_samples
    )
    
    # 训练模型
    model = MonthlyLayeredModel(config)
    model.train(args.data_dir, args.samples)
    
    # 评估模型
    results = model.evaluate(args.data_dir, test_samples=20000)
    
    # 保存模型
    model.save_model(args.output)
    
    # 输出总结
    print("\n=== 月份分层模型训练完成 ===")
    print(f"训练样本数: {args.samples}")
    print(f"分层数: {args.layers}")
    print(f"模型已保存: {args.output}")

if __name__ == "__main__":
    main()
