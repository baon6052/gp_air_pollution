import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import OUTPUTS_DIR

from multifidelity_analysis import run_multifidelity_analysis


kernel = "Exponential"

noise_multipliers = [0.1, 0.5, 1.0, 1.5]

high_fidelity_percentages = [5, 10, 20, 50, 100]
num_low_fidelity_samples = [50, 150, 200, 400]

metrics = itertools.product(
    noise_multipliers,
    high_fidelity_percentages,
    num_low_fidelity_samples,
)

n_of_runs = 5
results = {}
for j, (
    noise_multiplier,
    high_fidelity_percentage,
    num_low_fidelity_sample,
) in enumerate(metrics):
    d_key = f"({noise_multiplier},{high_fidelity_percentage},{num_low_fidelity_sample})"
    results[d_key] = {}
    results[d_key]["mae_list"] = []
    results[d_key]["mse_list"] = []
    results[d_key]["rmse_list"] = []

    num_high_fidelity_sample = int(
        num_low_fidelity_sample * high_fidelity_percentage / 100
    )

    for i in range(n_of_runs):
        mae, mse, rmse = run_multifidelity_analysis(
            climate_variables=[],
            num_high_fidelity_samples=num_high_fidelity_sample,
            num_low_fidelity_samples=num_low_fidelity_sample,
            plotting=False,
            kernel_name=kernel,
            noise_multiplier=noise_multiplier,
        )

        results[d_key][f"mae_list"].append(mae)
        results[d_key][f"mse_list"].append(mse)
        results[d_key][f"rmse_list"].append(rmse)
    for metric in ["mae", "mse", "rmse"]:
        results[d_key][f"mean_{metric}"] = np.round(
            np.mean(results[d_key][f"{metric}_list"]),
            4,
        )
        results[d_key][f"std_{metric}"] = np.round(
            np.std(results[d_key][f"{metric}_list"]),
            4,
        )
    print(
        f"Finished: \nnoise_multiplier {noise_multiplier}\high_fidelity_percentage {high_fidelity_percentage}\num_low_fidelity_sample {num_low_fidelity_sample}"
    )

df = pd.DataFrame(results).T

df.to_csv(Path("exp_data", "multifidelity_experiments.csv"))
df.to_hdf(
    Path("exp_data", "multifidelity_experiments.h5"),
    key="multifidelity_experiments",
)
