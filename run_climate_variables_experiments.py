import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import OUTPUTS_DIR
from gaussian_process import main, run_model

climate_variables = [
    "feels_like",
    "temp_max",
    "temp",
    "temp_min",
    "wind_speed",
    "wind_deg",
    "clouds_all",
    "humidity",
    "pressure",
]

kernel_names = [
    "RBF",
    "Matern52",
    "Linear",
    "Exponential",
    "Custom1",
    "Custom2",
    "Custom3",
]
climate_variables_iterator = []
for i in range(len(climate_variables) + 1):
    climate_variables_iterator.append(climate_variables[:i])

metrics = itertools.product(climate_variables_iterator, kernel_names)
n_of_runs = 8
results = {str(cv): {} for cv in climate_variables_iterator}
for j, (climate_variables, kernel_name) in enumerate(metrics):
    results[str(climate_variables)][kernel_name] = {}
    results[str(climate_variables)][kernel_name]["mae_list"] = []
    results[str(climate_variables)][kernel_name]["mse_list"] = []
    results[str(climate_variables)][kernel_name]["rmse_list"] = []
    for i in range(n_of_runs):
        mae, mse, rmse, _, _ = run_model(
            climate_variables=climate_variables,
            plotting=False,
            num_samples=5,
            acquisition_function="ModelVariance",
            kernel_name=kernel_name,
        )

        results[str(climate_variables)][kernel_name][f"mae_list"].append(mae)
        results[str(climate_variables)][kernel_name][f"mse_list"].append(mse)
        results[str(climate_variables)][kernel_name][f"rmse_list"].append(rmse)
    for metric in ["mae", "mse", "rmse"]:
        results[str(climate_variables)][kernel_name][
            f"mean_{metric}"
        ] = np.round(
            np.mean(
                results[str(climate_variables)][kernel_name][f"{metric}_list"]
            ),
            4,
        )
        results[str(climate_variables)][kernel_name][
            f"std_{metric}"
        ] = np.round(
            np.std(
                results[str(climate_variables)][kernel_name][f"{metric}_list"]
            ),
            4,
        )
    print(f"Finished: {climate_variables}-{kernel_name}")

df = pd.DataFrame(results).T

df.to_csv(Path(OUTPUTS_DIR, "climate_variables_performance_large_area_7.csv"))
df.to_hdf(
    Path(OUTPUTS_DIR, "climate_variables_performance_large_area_7.h5"),
    key="climate_variables_performance_large_area_7",
)
