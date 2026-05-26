import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_data(data_path):
    """加载数据并进行预处理"""
    print(f"正在加载数据: {data_path}")
    # 读取CSV，跳过单位行
    df = pd.read_csv(data_path, skiprows=[1])
    
    # 重命名列以匹配之前的代码习惯
    rename_map = {}
    if 'z' in df.columns:
        rename_map['z'] = 'pressure'
    if 'sea_water_temperature_profiler_depth_enabled' in df.columns:
        rename_map['sea_water_temperature_profiler_depth_enabled'] = 'temperature'
    elif 't' in df.columns:
        rename_map['t'] = 'temperature'
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # 确保pressure是正值 (深度)
    if 'pressure' in df.columns:
        df['pressure'] = df['pressure'].abs()

        
    # 转换时间列
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        
    # 移除无效数据
    df = df.dropna(subset=['pressure', 'temperature'])
    
    print(f"数据加载完成: {len(df)} 行")
    return df

def set_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    # 尝试设置中文字体，如果不行则回退
    plt.rcParams['axes.unicode_minus'] = False

def plot_temp_pressure_profile(df, output_dir):
    """绘制温深剖面图 (Scatter & Hexbin)"""
    print("绘制温深剖面图...")
    
    # 1. 散点图 (下采样以避免过慢)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    plt.figure(figsize=(8, 10))
    plt.scatter(df_sample['temperature'], df_sample['pressure'], 
                alpha=0.5, s=10, c=df_sample['time'].dt.year, cmap='viridis')
    plt.colorbar(label='Year')
    plt.gca().invert_yaxis()  # 深度向下增加
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (dbar)')
    plt.title(f'Temperature vs Pressure Profile (Sample N={sample_size})')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'dataset_tp_profile_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Hexbin 图 (全量数据密度)
    plt.figure(figsize=(8, 10))
    plt.hexbin(df['temperature'], df['pressure'], gridsize=100, cmap='inferno', bins='log')
    plt.colorbar(label='Log(Count)')
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (dbar)')
    plt.title('Temperature vs Pressure Density (Hexbin)')
    plt.savefig(output_dir / 'dataset_tp_profile_hexbin.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series(df, output_dir):
    """绘制时间序列图"""
    print("绘制时间序列图...")
    
    # 下采样
    df_sorted = df.sort_values('time')
    sample_size = min(50000, len(df))
    # 使用等间隔采样保留趋势
    indices = np.linspace(0, len(df_sorted)-1, sample_size, dtype=int)
    df_sample = df_sorted.iloc[indices]
    
    plt.figure(figsize=(15, 6))
    plt.scatter(df_sample['time'], df_sample['temperature'], 
                c=df_sample['pressure'], cmap='viridis_r', s=5, alpha=0.6)
    plt.colorbar(label='Pressure (dbar)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Time Series (Colored by Pressure)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'dataset_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_distributions(df, output_dir):
    """绘制分布直方图"""
    print("绘制分布直方图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 压力分布
    sns.histplot(data=df, x='pressure', bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Pressure Distribution')
    axes[0].set_xlabel('Pressure (dbar)')
    
    # 温度分布
    sns.histplot(data=df, x='temperature', bins=50, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Temperature Distribution')
    axes[1].set_xlabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_variation(df, output_dir):
    """绘制季节性变化 (Boxplot by Month)"""
    print("绘制季节性变化图...")
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='month', y='temperature', palette='coolwarm')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Distribution by Month')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(output_dir / 'dataset_seasonal_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_depth_time_heatmap(df, output_dir):
    """绘制 深度-时间 热力图 (需要分箱)"""
    print("绘制深度-时间热力图...")
    
    # 创建年月列
    df['ym'] = df['time'].dt.to_period('M')
    
    # 深度分箱
    df['pressure_bin'] = pd.cut(df['pressure'], bins=np.arange(0, df['pressure'].max()+100, 50))
    
    # 计算每个月、每个深度层的平均温度
    pivot_table = df.pivot_table(index='pressure_bin', columns='ym', values='temperature', aggfunc='mean')
    
    # 转换索引为中点值以便绘图
    pivot_table.index = [interval.mid for interval in pivot_table.index]
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_table, cmap='RdYlBu_r', cbar_kws={'label': 'Temperature (°C)'})
    plt.xlabel('Time (Year-Month)')
    plt.ylabel('Pressure (dbar)')
    plt.title('Monthly Mean Temperature by Depth')
    plt.gca().invert_yaxis() # heatmap 默认是从上到下index增加，这里pressure增加就是深度增加，符合直觉，不用invert? 
    # 不，heatmap 0在顶部。index是pressure从小到大。0 pressure (surface) should be at top.
    # pivot_table index: 25, 75, 125...
    # heatmap plots index 0 at top. So small pressure at top. Correct.
    
    plt.savefig(output_dir / 'dataset_depth_time_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='数据集可视化生成工具')
    parser.add_argument('--data-path', type=Path, default=Path('../data/ooi-new.csv'), help='数据文件路径')
    parser.add_argument('--output-dir', type=Path, default=Path('../results/figures'), help='输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    set_style()
    
    df = load_data(args.data_path)
    
    plot_temp_pressure_profile(df, args.output_dir)
    plot_time_series(df, args.output_dir)
    plot_distributions(df, args.output_dir)
    plot_seasonal_variation(df, args.output_dir)
    
    # 这个图比较耗时，且依赖于数据跨度，如果数据跨度太小可能报错，加个try-except
    try:
        plot_depth_time_heatmap(df, args.output_dir)
    except Exception as e:
        print(f"无法绘制深度-时间热力图: {e}")
        
    print(f"所有图表已生成至: {args.output_dir}")

if __name__ == '__main__':
    main()
