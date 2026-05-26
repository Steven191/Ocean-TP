#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

def basis_functions(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 0.1) # Protect against 0/negative
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

def load_data(data_dir: str, limit: int = 100000):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    dfs = []
    for f in files:
        if "synthetic" in f: continue
        try:
            # Try loading with skipping unit row
            df = pd.read_csv(f, skiprows=[1])
            # Check for required columns
            if 'sea_water_temperature_profiler_depth_enabled' in df.columns and 'z' in df.columns:
                temp_col = 'sea_water_temperature_profiler_depth_enabled'
                depth_col = 'z'
            elif 'temp' in df.columns and 'depth' in df.columns:
                temp_col = 'temp'
                depth_col = 'depth'
            else:
                continue
                
            df = df[[temp_col, depth_col]].dropna()
            df = df.rename(columns={temp_col: 'temp', depth_col: 'depth'})
            df['depth'] = np.abs(df['depth'])
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        raise ValueError("No valid data found")
        
    full_df = pd.concat(dfs, ignore_index=True)
    if len(full_df) > limit:
        full_df = full_df.sample(n=limit, random_state=42)
    
    return full_df.sort_values('depth')

def train_and_evaluate(df):
    P = df['depth'].values
    T = df['temp'].values
    
    # Split train/test
    indices = np.random.permutation(len(df))
    split = int(len(df) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    
    P_train, T_train = P[train_idx], T[train_idx]
    P_test, T_test = P[test_idx], T[test_idx]
    
    X_full_train = basis_functions(P_train)
    X_full_test = basis_functions(P_test)
    
    results = []
    
    print(f"Evaluating term increase (1 to 8 terms)...")
    for n_terms in range(1, 9):
        X_train = X_full_train[:, :n_terms]
        X_test = X_full_test[:, :n_terms]
        
        # Fit
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_train, T_train, rcond=None)
            
            # Predict
            pred_train = X_train @ coeffs
            pred_test = X_test @ coeffs
            
            mae_train = mean_absolute_error(T_train, pred_train)
            mae_test = mean_absolute_error(T_test, pred_test)
            rmse_test = np.sqrt(mean_squared_error(T_test, pred_test))
            
            print(f"Terms: {n_terms}, Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")
            
            results.append({
                "n_terms": n_terms,
                "mae_train": mae_train,
                "mae_test": mae_test,
                "rmse_test": rmse_test,
                "coeffs": coeffs.tolist()
            })
        except Exception as e:
            print(f"Error fitting {n_terms} terms: {e}")
            
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output", default="../results/term_increase_results.json")
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_data(args.data_dir)
    print(f"Loaded {len(df)} samples")
    
    results = train_and_evaluate(df)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.output}")
    
    # Plotting
    terms = [r['n_terms'] for r in results]
    maes = [r['mae_test'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(terms, maes, 'o-', linewidth=2)
    plt.xlabel('Number of Basis Terms')
    plt.ylabel('Test MAE (°C)')
    plt.title('Model Performance vs Number of Terms')
    plt.grid(True)
    plt.savefig(os.path.join('../results/figures', 'term_increase_mae.png'))
    print("Plot saved.")

if __name__ == "__main__":
    main()
