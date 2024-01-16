import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import OUTPUTS_DIR
from gaussian_process import main, run_model

num_samples = [5, 10, 15, 20, 25, 30]

n_of_runs = 20
results = {}
for j, (n_sample) in enumerate(num_samples):
    results[n_sample] = {}
    results[n_sample]["mae_list"] = []
    results[n_sample]["mse_list"] = []
    results[n_sample]["rmse_list"] = []
    for i in range(n_of_runs):
        mae, mse, rmse, _, _ = run_model(
            climate_variables=[],
            plotting=False,
            num_samples=n_sample,
            acquisition_function="ModelVariance",
            kernel_name="Matern52",
            regenerate_cache=False,
        )

        results[n_sample][f"mae_list"].append(mae)
        results[n_sample][f"mse_list"].append(mse)
        results[n_sample][f"rmse_list"].append(rmse)
    for metric in ["mae", "mse", "rmse"]:
        results[n_sample][f"mean_{metric}"] = np.round(
            np.mean(results[n_sample][f"{metric}_list"]), 4
        )
        results[n_sample][f"std_{metric}"] = np.round(
            np.std(results[n_sample][f"{metric}_list"]), 4
        )
    print(f"Finished: {n_sample}")

df = pd.DataFrame(results).T

df = df.drop(columns=["mae_list", "mse_list", "rmse_list"])

df.to_csv(Path(OUTPUTS_DIR, "n_samples_large_area_rev4.csv"))
