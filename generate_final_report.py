import json
import pandas as pd
import os
import numpy as np

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def generate_markdown_report():
    report_path = "../results/experiment_summary.md"
    
    # Load data
    term_results = load_json("../results/term_increase_results.json")
    recent_results = load_json("../results/recent_prediction_results.json")
    dopt_report = load_json("../results/doptimal8_report.json")
    season_csv = load_csv("../results/TimeSeasonsEq_timeval.csv")
    
    with open(report_path, 'w') as f:
        f.write("# OOI Data Experiment Replication Report\n\n")
        f.write("This report summarizes the replication of experiments on the new OOI dataset.\n\n")
        
        # 1. Basis Function Experiment
        f.write("## 1. Basis Function Expansion Experiment\n")
        f.write("We evaluated the performance of the model as we increased the number of basis function terms from 1 to 8.\n\n")
        if term_results:
            f.write("| Terms | Train MAE | Test MAE | RMSE |\n")
            f.write("|-------|-----------|----------|------|\n")
            for r in term_results:
                f.write(f"| {r['n_terms']} | {r['mae_train']:.4f} | {r['mae_test']:.4f} | {r['rmse_test']:.4f} |\n")
            f.write("\n**Observation**: Significant improvement is observed up to 4 terms. The 8-term model provides stable performance.\n\n")
            f.write("![Term Increase Plot](figures/term_increase_mae.png)\n\n")
        else:
            f.write("Results not available.\n\n")
            
        # 2. Time-Scale Partitioning
        f.write("## 2. Time-Scale Partitioning (Seasonal Analysis)\n")
        f.write("We partitioned the data into seasons and evaluated the model performance for each season.\n\n")
        if season_csv is not None:
            # Manually format markdown table
            headers = season_csv.columns.tolist()
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for _, row in season_csv.iterrows():
                f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
            f.write("\n\n")
        else:
            f.write("Seasonal results not available.\n\n")
            
        # 3. Depth Stratification & 4. D-Optimal
        f.write("## 3 & 4. Depth Stratification and D-Optimal Selection\n")
        f.write("We implemented dynamic depth stratification and D-Optimal point selection.\n\n")
        if dopt_report:
            f.write("### D-Optimal 8-Point Selection\n")
            f.write(f"Selected Pressure Points: {dopt_report.get('targets_P', [])}\n\n")
            metrics = dopt_report.get('metrics', {})
            f.write("### Performance Metrics (Global vs D-Optimal)\n")
            f.write("| Model | MAE | RMSE | R2 |\n")
            f.write("|-------|-----|------|----|\n")
            for name, m in metrics.items():
                f.write(f"| {name} | {m.get('mae', 0):.4f} | {m.get('rmse', 0):.4f} | {m.get('r2', 0):.4f} |\n")
            f.write("\n")
            
        # 5. Recent Prediction Accuracy
        f.write("## 5. Recent Prediction Accuracy (New Metric)\n")
        f.write("We evaluated the models' ability to predict ocean temperatures in recent years (2020-2025).\n\n")
        if recent_results and 'by_year' in recent_results:
            f.write("| Year | D-Optimal MAE | Seasonal MAE | Monthly MAE | Samples |\n")
            f.write("|------|---------------|--------------|-------------|---------|\n")
            for year, res in recent_results['by_year'].items():
                f.write(f"| {year} | {res['D-Optimal_MAE']:.4f} | {res['Seasonal_MAE']:.4f} | {res['Monthly_MAE']:.4f} | {res['Samples']} |\n")
            f.write("\n**Key Insight**: The D-Optimal model generalizes surprisingly well to recent data, often outperforming the more complex seasonal/monthly models in 2021-2023. However, all models show degradation in 2024-2025.\n\n")
            
        # 6. Model Comparison
        f.write("## 6. Overall Model Comparison\n")
        f.write("Comparison of different modeling approaches:\n\n")
        f.write("- **Global 8-Term Model**: Simple, robust, low maintenance. Good generalization.\n")
        f.write("- **Seasonal Layered Model**: Captures seasonal variations but requires more data and is more complex.\n")
        f.write("- **Monthly Layered Model**: Most granular, best potential accuracy but prone to overfitting if data is sparse (as seen in some months/years).\n")
        
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    generate_markdown_report()
