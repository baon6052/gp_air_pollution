import math
from pathlib import Path
from typing import Literal

import GPy
from GPy.util.normalizer import Standardize
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from GPy.models import GPRegression
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.interfaces import IModel
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import GPyModelWrapper
from sklearn.metrics import mean_squared_error, mean_absolute_error

from custom_experimental_design_loop import CustomExperimentalDesignLoop
from dataset import (
    extend_train_data,
    get_air_pollutant_level,
    get_batch_air_pollutant_levels, INPUTS_DIR, get_cached_air_pollution_data,
)
from dataset import get_cached_openweather_data


def get_model(train_x: npt.ArrayLike, train_y: npt.ArrayLike):
    num_input_parameters = train_x.shape[1]
    kernel = GPy.kern.Matern52(num_input_parameters)
    model_gpy = GPRegression(train_x, train_y, kernel, normalizer=Standardize())
    return GPyModelWrapper(model_gpy)


def get_parameter_space(
        input_bounds: dict[str, tuple[float, float]], climate_variables: list[str]
):
    parameter_spaces = [
        ContinuousParameter(name, min_bound, max_bound)
        for name, (min_bound, max_bound) in input_bounds.items()
    ]

    constant_spaces = [DiscreteParameter(name, [1]) for name in climate_variables]
    parameter_spaces.extend(constant_spaces)

    return ParameterSpace(parameter_spaces)


def read_sample_locations_air_pollution(path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_bayes_optimization(
        model: IModel,
        parameter_space: ParameterSpace,
        acquisition_func: ModelVariance,
        batch_size=1,
        max_iterations: int = 30,
        climate_variables: list[str] = [],
):
    expdesign_loop = CustomExperimentalDesignLoop(
        model=model,
        space=parameter_space,
        acquisition=acquisition_func,
        batch_size=batch_size,
        climate_variables=climate_variables,
    )

    expdesign_loop.run_loop(lambda x: get_air_pollutant_level(x[0]), max_iterations)
    return expdesign_loop


def evaluate_model(
        y_pred: npt.ArrayLike,
        y_true: npt.ArrayLike,
        metric: Literal["MAE", "MSE", "RMSE"] = "MAE",
) -> float:
    if metric == "MAE":
        return mean_absolute_error(y_true, y_pred)
    if metric == "MSE":
        return mean_squared_error(y_true, y_pred)
    if metric == "RMSE":
        return mean_squared_error(y_true, y_pred, squared=False)


def plot_results(mean, uncertainty, coords, observations):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, figsize=(12, 6), constrained_layout=True
    )

    num_samples = math.isqrt(coords.shape[0])
    print(num_samples)

    mean = mean.reshape((num_samples, num_samples))
    uncertainty = uncertainty.reshape((num_samples, num_samples))

    ax1.set_title("Mean PM2.5 Concentrations")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    cs1 = ax1.contourf(coords[:, 0].reshape((num_samples, num_samples)),
                       coords[:, 1].reshape((num_samples, num_samples)), mean)
    ax1.scatter(observations[:, 0], observations[:, 1], c="red", marker="o")
    plt.colorbar(cs1, ax=ax1)

    ax2.set_title("Uncertainty in estimation")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    cs2 = ax2.contourf(coords[:, 0].reshape((num_samples, num_samples)),
                       coords[:, 1].reshape((num_samples, num_samples)), uncertainty)
    ax2.scatter(observations[:, 0], observations[:, 1], c="red", marker="o")
    plt.colorbar(cs2, ax=ax2)
    fig.suptitle("Simple Gaussian Process Model for PM2.5 Concentrations")
    fig.savefig("gaussian_process.png")
    plt.show()


def process_results(model: IModel, observations: np.ndarray, num_samples: int,
                    climate_variables: list[str] = []):
    model_inputs = get_cached_openweather_data(num_samples ** 2, climate_variables)
    print(model_inputs[:, :2].shape)
    ground_truth = get_cached_air_pollution_data(num_samples ** 2)

    mean, uncertainty = model.predict(model_inputs)

    # plot the results
    plot_results(mean, uncertainty, model_inputs[:, :2], observations)

    # evaluate model performance
    mae = evaluate_model(mean, ground_truth)
    print(f"Mean absolute error: {mae:.2f}")


def run_basic_gp_regression(sample_locations_air_pollution_df: pd.DataFrame):
    matplotlib.use("Agg")
    GPy.plotting.change_plotting_library("matplotlib")

    filtered_df = sample_locations_air_pollution_df[
        sample_locations_air_pollution_df["datetime"] == "2023-12-01 17:00:00+00:00"
        ]
    train_x = filtered_df[["latitude", "longitude"]].to_numpy()
    train_y = np.expand_dims(filtered_df["pm2_5"].to_numpy(), axis=1)

    kernel = GPy.kern.Matern52(2)
    model = GPy.models.GPRegression(train_x, train_y, kernel)
    model.optimize(messages=True, max_iters=1_000)

    ax = model.plot()
    dataplot = ax["gpmean"][0]
    dataplot.figure.savefig("model.png")


def process_items(ctx, param, value):
    if value:
        items = value.split(",")
        items = [item.strip() for item in items]
        return items


@click.command()
@click.option("--climate_variables", callback=process_items, required=False)
def main(climate_variables):
    if not climate_variables:
        climate_variables = []

    sample_locations_air_pollution_df = read_sample_locations_air_pollution(
        Path(INPUTS_DIR, "air_pollution_per_lat_lng.csv")
    )

    bounds = {}
    # for parameter in input_parameters:
    #     min_bound = sample_locations_air_pollution_df.get(parameter).min()
    #     max_bound = sample_locations_air_pollution_df.get(parameter).max()
    #
    #     bounds[parameter] = (min_bound, max_bound)

    latitude_bounds = (
        sample_locations_air_pollution_df.latitude.min(),
        sample_locations_air_pollution_df.latitude.max(),
    )
    longitude_bounds = (
        sample_locations_air_pollution_df.longitude.min(),
        sample_locations_air_pollution_df.longitude.max(),
    )

    bounds["longitude"] = longitude_bounds
    bounds["latitude"] = latitude_bounds

    parameter_space = get_parameter_space(bounds, climate_variables=[])

    design = LatinDesign(parameter_space)

    initial_num_data_points = 5
    train_x = design.get_samples(initial_num_data_points)

    # Modify train_x to include input parameters
    train_x = extend_train_data(train_x[:, :2], climate_variables)
    train_y = get_batch_air_pollutant_levels(train_x[:, :2])

    model = get_model(train_x=train_x, train_y=train_y)
    acquisition_func = ModelVariance(model=model)

    parameter_space = get_parameter_space(bounds, climate_variables=climate_variables)

    results = run_bayes_optimization(
        model,
        parameter_space,
        acquisition_func,
        climate_variables=climate_variables,
    )
    process_results(
        results.model,
        results.loop_state.X[:, :2],
        200,
        climate_variables=climate_variables
    )
    # plot_results(
    #     results.model,
    #     results.loop_state.X,
    #     bounds,
    #     climate_variables=climate_variables,
    # )


if __name__ == "__main__":
    main()
