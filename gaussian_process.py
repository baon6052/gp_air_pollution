import json
import math
from pathlib import Path
from typing import Literal, Optional

import click
import geopandas as gpd
import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.interfaces import IModel
from emukit.core.loop.model_updaters import FixedIntervalUpdater
from emukit.experimental_design.acquisitions import (
    IntegratedVarianceReduction,
    ModelVariance,
)
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression
from GPy.util.normalizer import Standardize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from custom_experimental_design_loop import CustomExperimentalDesignLoop
from dataset import (
    INPUTS_DIR,
    extend_train_data,
    generate_air_pollution_cache,
    get_air_pollution_data,
    get_cached_air_pollution_data,
    get_cached_openweather_data,
    setup_cached_climate_data,
)


def get_model(
    train_x: npt.ArrayLike,
    train_y: npt.ArrayLike,
    kernel_name: str = "Matern52",
):
    num_input_parameters = train_x.shape[1]

    if kernel_name == "RBF":
        kernel = GPy.kern.RBF(num_input_parameters)
    elif kernel_name == "Matern52":
        kernel = GPy.kern.Matern52(num_input_parameters)
    elif kernel_name == "Linear":
        kernel = GPy.kern.Linear(num_input_parameters)
    elif kernel_name == "Exponential":
        kernel = GPy.kern.Exponential(num_input_parameters)
    elif kernel_name == "Custom1":
        rbf_kernel = GPy.kern.RBF(input_dim=num_input_parameters)
        matern52_kernel = GPy.kern.Matern52(
            input_dim=num_input_parameters,
        )
        kernel = rbf_kernel + matern52_kernel
    elif kernel_name == "Custom2":
        linear_kernel = GPy.kern.Linear(input_dim=num_input_parameters)
        matern52_kernel = GPy.kern.Matern52(
            input_dim=num_input_parameters,
        )
        kernel = linear_kernel + matern52_kernel
    elif kernel_name == "Custom3":
        rbf_kernel = GPy.kern.RBF(input_dim=num_input_parameters)
        linear_kernel = GPy.kern.Linear(input_dim=num_input_parameters)
        matern52_kernel = GPy.kern.Matern52(
            input_dim=num_input_parameters,
        )
        kernel = linear_kernel + matern52_kernel + rbf_kernel

    model_gpy = GPRegression(train_x, train_y, kernel, normalizer=Standardize())
    return GPyModelWrapper(model_gpy)


def get_parameter_space(
    input_bounds: dict[str, tuple[float, float]], climate_variables: list[str]
):
    parameter_spaces = [
        ContinuousParameter(name, min_bound, max_bound)
        for name, (min_bound, max_bound) in input_bounds.items()
    ]

    constant_spaces = [
        DiscreteParameter(name, [1]) for name in climate_variables
    ]
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

    expdesign_loop.run_loop(
        lambda x: get_air_pollution_data(x[0][:2].reshape(1, 2)),
        max_iterations,
    )
    return expdesign_loop


def evaluate_model(
    y_pred: npt.ArrayLike,
    y_true: npt.ArrayLike,
    metric: Literal["MAE", "MSE", "RMSE"] = "MAE",
) -> float:
    return (
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred, squared=False),
    )
    # if metric == "MAE":
    #     return mean_absolute_error(y_true, y_pred)
    # if metric == "MSE":
    #     return mean_squared_error(y_true, y_pred)
    # if metric == "RMSE":
    #     return mean_squared_error(y_true, y_pred, squared=False)


def plot_results(mean, uncertainty, coords, observations):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, figsize=(12, 6), constrained_layout=True
    )

    num_samples = math.isqrt(coords.shape[0])

    mean = mean.reshape((num_samples, num_samples))
    uncertainty = uncertainty.reshape((num_samples, num_samples))

    ax1.set_title("Mean PM2.5 Concentrations")
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Longitude")
    cs1 = ax1.contourf(
        coords[:, 0].reshape((num_samples, num_samples)),
        coords[:, 1].reshape((num_samples, num_samples)),
        mean,
    )
    # ax1.scatter(observations[:, 0], observations[:, 1], c="red", marker="o")
    plt.colorbar(cs1, ax=ax1)

    ax2.set_title("Uncertainty in estimation")
    ax2.set_xlabel("Latitude")
    ax2.set_ylabel("Longitude")
    cs2 = ax2.contourf(
        coords[:, 0].reshape((num_samples, num_samples)),
        coords[:, 1].reshape((num_samples, num_samples)),
        uncertainty,
    )
    ax2.scatter(observations[:, 0], observations[:, 1], c="red", marker="o")
    plt.colorbar(cs2, ax=ax2)
    fig.suptitle("Simple Gaussian Process Model for PM2.5 Concentrations")
    fig.savefig("figs/gaussian_process2.png")
    fig.savefig("figs/gaussian_process2.pdf")
    plt.show()


def process_results(
    model: IModel,
    observations: np.ndarray,
    num_samples: int,
    climate_variables: list[str] = [],
    plot_enabled: bool = True,
    add_multifidelity_column: bool = False,
    test_data: list[list[float]] = [],
):
    if test_data:
        model_inputs = test_data
    else:
        model_inputs, _ = get_cached_openweather_data(
            num_samples**2, climate_variables
        )

    ground_truth = get_cached_air_pollution_data(num_samples**2)

    if add_multifidelity_column:
        high_fidelity_column_and_values = np.array(
            [[0] for _ in range(model_inputs.shape[0])]
        )
        model_inputs = np.append(
            model_inputs, high_fidelity_column_and_values, axis=1
        )

    mean, uncertainty = model.predict(model_inputs)

    # plot the results
    if plot_enabled:
        plot_results(mean, uncertainty, model_inputs[:, :2], observations)

    save_to_geojson(
        coordinates=model_inputs[:, :2], mean=mean, uncertainty=uncertainty
    )

    # evaluate model performance
    mae, mse, rmse = evaluate_model(mean, ground_truth)
    return mae, mse, rmse


def save_to_geojson(coordinates, mean, uncertainty):
    geojson = {"type": "FeatureCollection", "features": []}

    # Add points to the GeoJSON
    for point, m, u in zip(coordinates, mean, uncertainty):
        feature = {
            "type": "Feature",
            "properties": {
                "mean": float(m),
                "uncertainty": float(u),
            },  # Empty properties, can be filled with relevant information
            "geometry": {
                "type": "Point",
                "coordinates": [float(point[1]), float(point[0])],
            },
        }
        geojson["features"].append(feature)
    geojson_string = json.dumps(geojson, indent=2)
    with open("output.geojson", "w") as file:
        file.write(geojson_string)
    pass


def process_items(ctx, param, value):
    if value:
        items = value.split(",")
        items = [item.strip() for item in items]
        return items


@click.command()
@click.option(
    "--climate_variables",
    callback=process_items,
    required=False,
)
@click.option("--num_samples", default=5)
@click.option("--plotting/--no-plotting", default=True)
@click.option(
    "--acquisition_function",
    type=click.Choice(["ModelVariance", "IVR"], case_sensitive=False),
    default="ModelVariance",
)
@click.option("--kernel_name", default="Matern52")
@click.option("--regenerate_cache", default=False)
def main(
    climate_variables: Optional[list[str]],
    num_samples: int,
    plotting: bool,
    acquisition_function: str,
    kernel_name: str,
    regenerate_cache: bool,
):
    run_model(
        climate_variables,
        num_samples,
        plotting,
        acquisition_function,
        kernel_name,
        regenerate_cache,
    )


def run_model(
    climate_variables: Optional[list[str]],
    num_samples: int,
    plotting: bool,
    acquisition_function: str,
    kernel_name: str,
    regenerate_cache: bool = False,
):
    if not climate_variables:
        climate_variables = []

    extent_gpd = gpd.read_file("data/extent.geojson")

    bounds = {}

    # latitude_bounds = (
    #     extent_gpd.MINY[
    #         0
    #     ],  # 50.866580,  # sample_locations_air_pollution_df.longitude.min(),
    #     extent_gpd.MAXY[
    #         0
    #     ],  # 52.608829,  # sample_locations_air_pollution_df.longitude.max(),
    # )

    # longitude_bounds = (
    #     extent_gpd.MINX[
    #         0
    #     ],  # -2.173528,  # sample_locations_air_pollution_df.latitude.min(),
    #     extent_gpd.MAXX[
    #         0
    #     ],  # 0.312971,  # sample_locations_air_pollution_df.latitude.max(),
    # )

    latitude_bounds = (50.866580, 52.608829)
    longitude_bounds = (-2.173528, 0.312971)
    bounds["latitude"] = latitude_bounds
    bounds["longitude"] = longitude_bounds

    if regenerate_cache:
        xmin, xmax, ymin, ymax = (
            latitude_bounds[0],
            latitude_bounds[1],
            longitude_bounds[0],
            longitude_bounds[1],
        )
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200)
        )
        xc = xx.flatten()
        yc = yy.flatten()
        coordinates = list(zip(xc, yc))
        generate_air_pollution_cache(coordinates)

        # need_df = pd.read_csv("need_df.csv")
        # coordinates = list(zip(list(need_df["latitude"]), list(need_df["longitude"])))
        setup_cached_climate_data(
            coordinates,
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
        )

    parameter_space = get_parameter_space(bounds, climate_variables=[])

    design = LatinDesign(parameter_space)

    train_x = design.get_samples(num_samples)

    train_x = extend_train_data(train_x[:, :2], climate_variables)

    train_y = get_air_pollution_data(train_x[:, :2])

    model = get_model(train_x=train_x, train_y=train_y, kernel_name=kernel_name)

    parameter_space = get_parameter_space(
        bounds, climate_variables=climate_variables
    )

    if acquisition_function == "IVR":
        acquisition_func = IntegratedVarianceReduction(
            model=model, space=parameter_space, num_monte_carlo_points=10_000
        )
    else:
        acquisition_func = ModelVariance(model=model)

    results = run_bayes_optimization(
        model,
        parameter_space,
        acquisition_func,
        climate_variables=climate_variables,
        max_iterations=30,
    )
    mae, mse, rmse = process_results(
        model,
        model.X[:, :2],
        200,
        climate_variables=climate_variables,
        plot_enabled=plotting,
    )
    return mae, mse, rmse, results.model, parameter_space


if __name__ == "__main__":
    main()
