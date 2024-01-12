from pathlib import Path

from gaussian_process import main
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


climate_variables_iterator = [
    climate_variables[:i] for i in range(len(climate_variables))
]
n_of_runs = 5
results = {k: {} for k in range(1, len(climate_variables_iterator) + 1)}
for j, climate_variables in enumerate(climate_variables_iterator):
    results[j + 1]["climate_variables"] = climate_variables
    results[j + 1]["mae_list"] = []
    for i in range(n_of_runs):
        mae = main(
            climate_variables=climate_variables,
            plotting=False,
            num_samples=5,
            acquisition_function="ModelVariance",
        )
        results[j + 1]["mae_list"].append(mae)
    results[j + 1]["mean_mae"] = np.round(np.mean(results[j + 1]["mae_list"]), 4)
    results[j + 1]["std_mae"] = np.round(np.std(results[j + 1]["mae_list"]), 4)
    print(f"Finished: {climate_variables}")

df = pd.DataFrame(results).T
df.to_csv(Path(OUTPUTS_DIR, "climate_variables_performance3.csv"))
