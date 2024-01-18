from pathlib import Path

import pandas as pd

from gaussian_process import run_model

acquisition_functions = ["ModelVariance", "IVR"]
results = {}
raw_data = {
    "acquisition_function": [],
    "mae": [],
    "mse": [],
    "rmse": []
}
n_runs = 25

for acq_func in acquisition_functions:
    for i in range(n_runs):
        mae, mse, rmse, _, _ = run_model(
            climate_variables=[],
            plotting=False,
            num_samples=5,
            acquisition_function=acq_func,
            kernel_name="Exponential",
            regenerate_cache=False,
        )

        raw_data["acquisition_function"].append(acq_func)
        raw_data[f"mae"].append(mae)
        raw_data[f"mse"].append(mse)
        raw_data[f"rmse"].append(rmse)

df = pd.DataFrame(raw_data)

df.to_csv(Path('exp_data', "acquisition_functions.csv"))
