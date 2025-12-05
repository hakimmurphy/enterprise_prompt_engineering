# ablations/compare.py
import pandas as pd
from pathlib import Path

def compare(res_a, res_b, summary_path=None):
    da, db = pd.read_csv(res_a), pd.read_csv(res_b)
    delta_acc = db["acc"].mean() - da["acc"].mean()
    delta_latency = db["latency"].mean() - da["latency"].mean()
    delta_cost = db["cost"].sum() - da["cost"].sum()
    print("ΔACC:", delta_acc)
    print("ΔLatency:", delta_latency)
    print("ΔCost:", delta_cost)
    # Save summary if path provided
    if summary_path:
        import csv
        summary_dir = Path(summary_path).parent
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "accuracy_difference",
                "latency_difference_seconds",
                "cost_difference_usd"
            ])
            writer.writerow([delta_acc, delta_latency, delta_cost])


import numpy as np

def generate_example_csv(filename, seed=0):
    np.random.seed(seed)
    data = {
        "acc": np.random.uniform(0.7, 1.0, 10),
        "latency": np.random.uniform(100, 300, 10),
        "cost": np.random.uniform(0.01, 0.10, 10),
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Generate example CSVs if they do not exist
    if not Path("results_v1.csv").exists():
        generate_example_csv("results_v1.csv", seed=1)
    if not Path("results_v2.csv").exists():
        generate_example_csv("results_v2.csv", seed=2)
    # Save summary to tasks/classification/results/comparison_summary.csv
    summary_path = Path("tasks/classification/results/comparison_summary.csv")
    compare("results_v1.csv", "results_v2.csv", summary_path=summary_path)
