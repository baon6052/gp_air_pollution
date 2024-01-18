import pandas as pd

from custom_monte_carlo_sensitivity import CustomMonteCarloSensitivity
from gaussian_process import run_model


def run_sensitivity_analysis():
    climate_variables = [
        "temp",
        "pressure",
        "humidity",
        "clouds_all",
        "wind_deg",
        "wind_speed",
    ]
    mae, model, parameter_space = run_model(
        climate_variables=climate_variables,
        num_samples=5,
        plotting=False,
        acquisition_function="ModelVariance",
        kernel_name="Custom2",
    )

    senstivity = CustomMonteCarloSensitivity(
        model=model, input_domain=parameter_space
    )
    main_effects, total_effects, _ = senstivity.compute_effects(
        num_monte_carlo_points=200_000, climate_variables=climate_variables
    )

    df = pd.DataFrame(
        data={
            "Variable": main_effects.keys(),
            "Main Effects": map(lambda x: x[0], main_effects.values()),
            "Total Effects": map(lambda x: x[0], total_effects.values()),
        }
    )
    df.to_csv("exp_data/sensitivity_analysis.csv", index=False)


if __name__ == "__main__":
    run_sensitivity_analysis()
