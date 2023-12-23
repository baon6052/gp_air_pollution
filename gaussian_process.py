from pathlib import Path

import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from GPy.models import GPRegression
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.interfaces import IModel
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import GPyModelWrapper

from dataset import get_air_pollutant_level, get_batch_air_pollutant_levels

CWD = Path.cwd()
DATA_DIR = Path(CWD, "data")
INPUTS_DIR = Path(DATA_DIR, "inputs")
OUTPUTS_DIR = Path(DATA_DIR, "outputs")


def get_model(train_x: npt.ArrayLike, train_y: npt.ArrayLike):
    kernel = GPy.kern.Matern52(2)
    model_gpy = GPRegression(train_x, train_y, kernel)
    return GPyModelWrapper(model_gpy)


def get_parameter_space(
        longitude_bounds: tuple[float, float], latitude_bounds: tuple[float, float]
):
    return ParameterSpace(
        [
            ContinuousParameter(
                "latitude", latitude_bounds[0], latitude_bounds[1]
            ),
            ContinuousParameter(
                "longitude", longitude_bounds[0], longitude_bounds[1]
            ),
        ]
    )


def read_sample_locations_air_pollution(path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_bayes_optimization(
        model: IModel,
        parameter_space: ParameterSpace,
        acquisition_func: ModelVariance,
        batch_size=1,
        max_iterations: int = 30,
):
    expdesign_loop = ExperimentalDesignLoop(
        model=model,
        space=parameter_space,
        acquisition=acquisition_func,
        batch_size=batch_size,
    )

    expdesign_loop.run_loop(lambda x: get_air_pollutant_level(x[0]), max_iterations)
    return expdesign_loop


def plot_results(model: IModel, observations: np.ndarray,
                 bounds: dict[str, tuple[float, float]]):
    lats = np.linspace(bounds["latitude"][0], bounds["latitude"][1], 1_000)
    longs = np.linspace(bounds["longitude"][0], bounds["longitude"][1], 1_000)
    X, Y = np.meshgrid(lats, longs)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    mean, uncertainty = model.predict(coords)
    mean = mean.reshape((1_000, 1_000))
    uncertainty = uncertainty.reshape((1_000, 1_000))
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6), constrained_layout=True)
    ax1.set_title("Mean PM2.5 Concentrations")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    cs1 = ax1.contourf(Y, X, mean)
    ax1.scatter(observations[:, 1], observations[:, 0], c="red", marker="o")
    plt.colorbar(cs1, ax=ax1)

    ax2.set_title("Uncertainty in estimation")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    cs2 = ax2.contourf(Y, X, uncertainty)
    ax2.scatter(observations[:, 1], observations[:, 0], c="red", marker="o")
    plt.colorbar(cs2, ax=ax2)
    fig.suptitle("Simple Gaussian Process Model for PM2.5 Concentrations")
    fig.savefig("gaussian_process.png")


def run_basic_gp_regression(sample_locations_air_pollution_df: pd.DataFrame):
    matplotlib.use("Agg")
    GPy.plotting.change_plotting_library("matplotlib")

    filtered_df = sample_locations_air_pollution_df[
        sample_locations_air_pollution_df["datetime"]
        == "2023-12-01 17:00:00+00:00"
        ]
    train_x = filtered_df[["latitude", "longitude"]].to_numpy()
    train_y = np.expand_dims(filtered_df["pm2_5"].to_numpy(), axis=1)

    kernel = GPy.kern.Matern52(2)
    model = GPy.models.GPRegression(train_x, train_y, kernel)
    model.optimize(messages=True, max_iters=1_000)

    ax = model.plot()
    dataplot = ax["gpmean"][0]
    dataplot.figure.savefig("model.png")


def main():
    sample_locations_air_pollution_df = read_sample_locations_air_pollution(
        Path(INPUTS_DIR, "air_pollution_per_lat_lng.csv")
    )
    latitude_bounds = (
        sample_locations_air_pollution_df.latitude.min(),
        sample_locations_air_pollution_df.latitude.max(),
    )
    longitude_bounds = (
        sample_locations_air_pollution_df.longitude.min(),
        sample_locations_air_pollution_df.longitude.max(),
    )

    parameter_space = get_parameter_space(longitude_bounds, latitude_bounds)
    design = LatinDesign(parameter_space)

    initial_num_data_points = 5
    train_x = design.get_samples(initial_num_data_points)
    train_y = get_batch_air_pollutant_levels(train_x)

    model = get_model(train_x=train_x, train_y=train_y)
    acquisition_func = ModelVariance(model=model)

    results = run_bayes_optimization(model, parameter_space, acquisition_func)
    plot_results(results.model, results.loop_state.X, {
        "latitude": latitude_bounds,
        "longitude": longitude_bounds
    })


if __name__ == "__main__":
    main()
