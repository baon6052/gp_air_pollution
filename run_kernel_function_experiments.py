from pathlib import Path

import numpy as np
import pandas as pd

from dataset import OUTPUTS_DIR
from gaussian_process import run_model

kernel_functions = [
    "RBF",
    "Matern52",
    "Linear",
    "Exponential",
    "Custom1",
    "Custom2",
    "Custom3",
]

results = {k: {} for k in kernel_functions}
n_of_runs = 5
for kernel_name in kernel_functions:
    results[kernel_name]["mae_list"] = []
    for i in range(n_of_runs):
        mae, mse, rmse, _, _ = run_model(
            climate_variables=[],
            plotting=False,
            num_samples=5,
            acquisition_function="ModelVariance",
            kernel_name=kernel_name,
        )
        results[kernel_name]["mae_list"].append(mae)
        results[kernel_name]["mse_list"].append(mse)
        results[kernel_name]["rmse_list"].append(mse)

    for metric in ["mae", "mse", "rmse"]:
        results[kernel_name][f"mean_{metric}"] = np.round(
            np.mean(results[kernel_name][f"{metric}_list"]), 4
        )
        results[kernel_name][f"std_{metric}"] = np.round(
            np.std(results[kernel_name][f"{metric}_list"]), 4
        )
    print(f"Finished: {kernel_name}")

df = pd.DataFrame(results).T.drop(columns=["mae_list", "mse_list", "rmse_list"])
df.to_csv(Path(OUTPUTS_DIR, "kernel_function_performance.csv"))
