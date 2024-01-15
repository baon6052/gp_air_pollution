from pathlib import Path

from gaussian_process import main, run_model
from dataset import OUTPUTS_DIR
import numpy as np
import pandas as pd


climate_variables = [
    "feels_like",
    "temp",
    "temp_min",
    "temp_max",
    "humidity",
    "clouds_all",
    "wind_deg",
    "pressure",
    "wind_speed",
]

n_samples = 5
n_runs = 10
results = {"mae_list": []}
for i in range(n_runs):
    mae, _, _ = run_model(
        climate_variables=climate_variables,
        plotting=False,
        num_samples=n_samples,
        acquisition_function="ModelVariance",
        kernel_name="Custom2",
    )
    results["mae_list"].append(mae)

results["average_mae"] = np.mean(results["mae_list"])
results["std_mae"] = np.std(results["mae_list"])


print(results)
