from pathlib import Path

import GPy
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression

from dataset import get_air_pollutant_level, get_batch_air_pollutant_levels

CWD = Path.cwd()
DATA_DIR = Path(CWD, "data")
INPUTS_DIR = Path(DATA_DIR, "inputs")
OUTPUTS_DIR = Path(DATA_DIR, "outputs")


def get_model(train_x: npt.ArrayLike, train_y: npt.ArrayLike):
    model_gpy = GPRegression(train_x, train_y)
    return GPyModelWrapper(model_gpy)


def get_parameter_space(
    longitude_bounds: tuple[float, float], latitude_bounds: tuple[float, float]
):
    return ParameterSpace(
        [
            ContinuousParameter(
                "longitude", longitude_bounds[0], longitude_bounds[1]
            ),
            ContinuousParameter(
                "latitude", latitude_bounds[0], latitude_bounds[1]
            ),
        ]
    )


def read_sample_locations_air_pollution(path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_bayes_optimization(
    model: GPyModelWrapper,
    parameter_space: ParameterSpace,
    expected_improvement: ExpectedImprovement,
    batch_size=1,
    max_iterations: int = 30,
):
    bayesopt_loop = BayesianOptimizationLoop(
        model=model,
        space=parameter_space,
        acquisition=expected_improvement,
        batch_size=batch_size,
    )

    bayesopt_loop.run_loop(get_air_pollutant_level, max_iterations)
    return bayesopt_loop.get_results()


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
    design = RandomDesign(parameter_space)

    initial_num_data_points = 5
    train_x = design.get_samples(initial_num_data_points)
    train_y = get_batch_air_pollutant_levels(train_x)

    model = get_model(train_x=train_x, train_y=train_y)
    expected_improvement = ExpectedImprovement(model=model)

    return run_bayes_optimization(model, parameter_space, expected_improvement)


if __name__ == "__main__":
    main()
