# from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from gaussian_process import run_model


def run_sensitivity_analysis():
    mae, model, parameter_space = run_model(
        climate_variables=[
            "feels_like",
            "temp",
            "temp_min",
            "temp_max",
            "humidity",
            "clouds_all",
            "wind_deg",
            "pressure",
            "wind_speed",
        ],
        num_samples=5,
        plotting=False,
        acquisition_function="ModelVariance",
        kernel_name="Matern52",
    )

    senstivity = MonteCarloSensitivity(model=model, input_domain=parameter_space)
    main_effects, total_effects, _ = senstivity.compute_effects(
        num_monte_carlo_points=10
    )
    print()


if __name__ == "__main__":
    run_sensitivity_analysis()
