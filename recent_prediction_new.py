
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_all_data(data_dir: str) -> pd.DataFrame:
    """加载目录下所有 CSV 文件"""
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # print(f"Found {len(all_files)} CSV files")
    
    dfs = []
    from tqdm import tqdm
    for f in tqdm(all_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(f, skiprows=[1])
            # 统一列名
            if 'z' in df.columns:
                df = df.rename(columns={'z': 'depth'})
            
            # 查找温度列
            temp_col = next((c for c in df.columns if 'sea_water_temperature' in c), None)
            if temp_col:
                df = df.rename(columns={temp_col: 'temp'})
            
            # 统一时间列
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        raise ValueError("No valid data loaded")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def evaluate_recent_years(data_dir: Path, model_paths: dict, start_year: int = 2020):
    """评估模型在最近几年的表现"""
    print(f"加载数据 (filter year >= {start_year})...")
    df = load_all_data(str(data_dir))
    
    # 提取年份和月份
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    recent_df = df[df['year'] >= start_year].copy()
    
    if len(recent_df) == 0:
        print(f"警告: 没有找到 {start_year} 年以后的数据")
        return {}
        
    print(f"近期数据样本数: {len(recent_df)}")
    
    # 按年份分组评估
    results = {}
    
    # 加载模型
    loaded_models = {}
    # 这里我们只加载之前生成的 JSON 结果作为"模型"（实际上我们需要加载模型对象或者使用保存的参数）
    # 为了简化，我们假设我们有办法利用保存的参数重建预测
    # 由于之前的脚本只保存了 metrics 和 config，没有保存模型对象本身(pickle)，
    # 我们这里主要关注 "D-Optimal" 模型（简单的公式）和 "Seasonal/Monthly"（需要重建）
    # 但为演示目的，我们可能需要重新训练或者使用简化策略。
    
    # 鉴于上下文，我们最好重新训练模型或者使用之前脚本中定义的类来加载参数。
    # 简单起见，我们在这里实例化类并重新训练（使用全量数据或大量数据），然后评估近期数据。
    # 这是一个"评估"脚本，所以理想情况下应该加载预训练模型。
    # 但之前的脚本只输出了 JSON 结果。
    
    # 我们修改策略：
    # 1. 重新训练各个模型（使用 2020 年之前的数据作为训练集，或者全量数据如果目的是评估当前模型的时效性）
    # 通常"近期预测准确度"意味着测试集是近期数据。
    # 所以我们将数据分为：训练集 (< start_year) 和 测试集 (>= start_year)
    
    train_df = df[df['year'] < start_year]
    test_df = recent_df
    
    print(f"训练集样本 (<{start_year}): {len(train_df)}")
    print(f"测试集样本 (>={start_year}): {len(test_df)}")
    
    if len(train_df) < 1000:
        print("警告: 训练数据不足，使用随机采样划分")
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        # 再次过滤测试集为近期
        test_df = test_df[test_df['year'] >= start_year]
    
    # 1. D-Optimal 8参数模型
    print("训练 D-Optimal 8参数模型...")
    # from doptimal8_fit_new import fit_model_doptimal
    # 注意：doptimal8_fit_new.py 里的 fit_model_doptimal 需要适配
    # 我们直接在这里实现简单的 8 参数拟合
    
    def basis_functions(P):
        # 预处理 P，避免无效值
        P = np.array(P)
        # 确保非负
        P_safe = np.abs(P)
        
        return np.column_stack([
            np.ones_like(P_safe),
            P_safe,
            np.log(P_safe + 1e-6),
            np.sqrt(P_safe),
            np.exp(-P_safe/500),
            np.exp(-P_safe/1200),
            1/(P_safe+50),
            1/(P_safe+500)
        ])
    
    # 过滤无效值
    valid_mask = ~np.isnan(train_df['depth']) & ~np.isnan(train_df['temp'])
    P_train = train_df[valid_mask]['depth'].values
    T_train = train_df[valid_mask]['temp'].values
    
    # 简单最小二乘拟合 (模拟 D-Optimal 的结果，实际 D-Optimal 是选点策略，最终还是回归)
    # 为了公平对比，我们用全量训练数据的最小二乘作为基准
    H = basis_functions(P_train)
    coeffs, _, _, _ = np.linalg.lstsq(H, T_train, rcond=None)
    
    # 预测
    P_test = test_df['depth'].values
    T_test = test_df['temp'].values
    H_test = basis_functions(P_test)
    T_pred_dopt = H_test @ coeffs
    
    results['D-Optimal'] = {
        'mae': mean_absolute_error(T_test, T_pred_dopt),
        'rmse': np.sqrt(mean_squared_error(T_test, T_pred_dopt)),
        'r2': r2_score(T_test, T_pred_dopt)
    }
    
    # 2. 季节性分层模型
    print("训练季节性分层模型...")
    from seasonal_layered_model_new import SeasonalLayeredModel, SeasonalLayerConfig
    # 需要临时修改 train 方法接受 dataframe 或者我们手动构造
    # 由于 seasonal_layered_model_new.py 设计为从文件读取，我们这里稍微变通一下
    # 我们实例化模型，手动填充数据
    
    seasonal_config = SeasonalLayerConfig(layers=4, min_thickness=200.0, min_samples_per_layer=500)
    seasonal_model = SeasonalLayeredModel(seasonal_config)
    
    # 手动训练
    # 提取季节数据
    seasonal_data = seasonal_model.extract_seasonal_data(train_df)
    for season, season_df in seasonal_data.items():
        if len(season_df) > 100:
            seasonal_model.seasonal_models[season] = seasonal_model.train_seasonal_layer_model(season_df, season)
            
    # 预测
    T_pred_seasonal = seasonal_model.predict(P_test, datetime_info=test_df)
    
    results['Seasonal_Layered'] = {
        'mae': mean_absolute_error(T_test, T_pred_seasonal),
        'rmse': np.sqrt(mean_squared_error(T_test, T_pred_seasonal)),
        'r2': r2_score(T_test, T_pred_seasonal)
    }
    
    # 3. 月份分层模型
    print("训练月份分层模型...")
    from monthly_layered_model_new import MonthlyLayeredModel
    # 配置复用
    monthly_model = MonthlyLayeredModel(seasonal_config)
    
    # 手动训练
    monthly_data = monthly_model.extract_monthly_data(train_df)
    for month, month_df in monthly_data.items():
        if len(month_df) > 100:
            monthly_model.monthly_models[month] = monthly_model.train_monthly_layer_model(month_df, month)
            
    # 预测
    T_pred_monthly = monthly_model.predict(P_test, datetime_info=test_df)
    
    results['Monthly_Layered'] = {
        'mae': mean_absolute_error(T_test, T_pred_monthly),
        'rmse': np.sqrt(mean_squared_error(T_test, T_pred_monthly)),
        'r2': r2_score(T_test, T_pred_monthly)
    }

    # 按年份细分评估
    years = sorted(test_df['year'].unique())
    year_details = {}
    
    for year in years:
        mask = test_df['year'] == year
        if not mask.any():
            continue
            
        y_true = T_test[mask]
        y_depth = P_test[mask]
        
        # D-Optimal
        y_pred_dopt = T_pred_dopt[mask]
        mae_dopt = mean_absolute_error(y_true, y_pred_dopt)
        
        # Seasonal
        y_pred_sea = T_pred_seasonal[mask]
        mae_sea = mean_absolute_error(y_true, y_pred_sea)
        
        # Monthly
        y_pred_mon = T_pred_monthly[mask]
        mae_mon = mean_absolute_error(y_true, y_pred_mon)
        
        year_details[int(year)] = {
            'D-Optimal_MAE': mae_dopt,
            'Seasonal_MAE': mae_sea,
            'Monthly_MAE': mae_mon,
            'Samples': int(mask.sum())
        }
        print(f"Year {year}: D-Opt MAE={mae_dopt:.4f}, Seasonal MAE={mae_sea:.4f}, Monthly MAE={mae_mon:.4f}")

    return {
        'overall': results,
        'by_year': year_details
    }

def main():
    parser = argparse.ArgumentParser(description='近期预测准确度评估')
    parser.add_argument('--data-dir', type=Path, required=True, help='数据目录')
    parser.add_argument('--output', type=Path, required=True, help='输出结果路径')
    parser.add_argument('--start-year', type=int, default=2020, help='近期开始年份')
    
    args = parser.parse_args()
    
    results = evaluate_recent_years(args.data_dir, {}, args.start_year)
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"评估完成，结果已保存到: {args.output}")

if __name__ == '__main__':
    main()
