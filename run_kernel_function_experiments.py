from pathlib import Path

from gaussian_process import main
from dataset import OUTPUTS_DIR
import numpy as np
import pandas as pd

kernel_functions = ["Custom2", "Custom1", "RBF", "Matern52", "Linear"]

results = {k: {} for k in kernel_functions}
n_of_runs = 5
for kernel_name in kernel_functions:
    results[kernel_name]["mae_list"] = []
    for i in range(n_of_runs):
        mae = main(
            climate_variables=[],
            plotting=False,
            num_samples=5,
            acquisition_function="ModelVariance",
            kernel_name=kernel_name,
        )
        results[kernel_name]["mae_list"].append(mae)
    results[kernel_name]["mean_mae"] = np.round(
        np.mean(results[kernel_name]["mae_list"]), 4
    )
    results[kernel_name]["std_mae"] = np.round(
        np.std(results[kernel_name]["mae_list"]), 4
    )
    print(f"Finished: {kernel_name}")

df = pd.DataFrame(results).T
df.to_csv(Path(OUTPUTS_DIR, "kernel_function_performance.csv"))
