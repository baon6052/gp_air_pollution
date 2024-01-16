from copy import deepcopy

import click
import GPy
import numpy as np
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_xy_lists_to_arrays,
)
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.test_functions.forrester import forrester, forrester_low

from dataset import get_cached_air_pollution_data, get_cached_openweather_data
from gaussian_process import process_items, process_results


def get_multifidelity_model(
    kernel_name, num_input_parameters, num_fidelities, X_train, Y_train
):
    kernels = []

    for _ in range(2):
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

        kernels.append(kernel)

    linear_mf_kernel = LinearMultiFidelityKernel(kernels)
    return GPyLinearMultiFidelityModel(
        X_train, Y_train, linear_mf_kernel, n_fidelities=num_fidelities
    )


def get_datasets(climate_variables):
    num_values_high_fidelity = 10
    num_values_low_fidelity = 200

    dataset_size = num_values_high_fidelity + num_values_low_fidelity

    model_inputs, test_data = get_cached_openweather_data(
        num_samples=dataset_size,
        climate_variables=climate_variables,
        shuffle_coordinates=True,
    )
    coordinates = [
        (latitude, longitude) for latitude, longitude in model_inputs[:, :2]
    ]
    coordinates = [
        tuple(np.float64(coord) for coord in pair) for pair in coordinates
    ]

    ground_truth = get_cached_air_pollution_data(
        num_samples=dataset_size, coordinates=coordinates
    )

    x_low_fidelity = deepcopy(model_inputs)[:num_values_low_fidelity]
    y_low_fidelity = deepcopy(ground_truth)[:num_values_low_fidelity]

    std_y_low_fidelity = np.std(y_low_fidelity)
    mean_y_low_fidelity = np.mean(y_low_fidelity)
    random_noise_list = np.random.normal(
        mean_y_low_fidelity, std_y_low_fidelity, len(y_low_fidelity)
    )
    y_low_fidelity = np.array(
        [
            y_value + (0.5 * random_noise)
            for y_value, random_noise in zip(y_low_fidelity, random_noise_list)
        ]
    )

    high_fidelity_dataset = deepcopy(
        np.random.permutation(
            [
                [x_sample, y_sample]
                for x_sample, y_sample in zip(model_inputs, ground_truth)
            ]
        )
    )[:num_values_high_fidelity]

    x_high_fidelity = np.array(
        [item[0] for item in high_fidelity_dataset[:, :1]]
    )
    y_high_fidelity = np.array(
        [item[0] for item in high_fidelity_dataset[:, 1:]]
    )

    X_train, Y_train = convert_xy_lists_to_arrays(
        [x_high_fidelity, x_low_fidelity], [y_high_fidelity, y_low_fidelity]
    )

    return X_train, Y_train, test_data


# from gaussian_process import run_model


@click.command()
@click.option("--climate_variables", callback=process_items, required=False)
@click.option("--num_high_fidelity_samples", default=10)
@click.option("--num_low_fidelity_samples", default=200)
@click.option("--plotting/--no-plotting", default=True)
@click.option("--kernel_name", default="Matern52")
@click.option("--amount_of_noise", default="0.5")
def run_multifidelity_analysis(
    climate_variables,
    num_high_fidelity_samples,
    num_low_fidelity_samples,
    plotting: bool,
    kernel_name,
    amount_of_noise,
):
    if not climate_variables:
        climate_variables = []

    X_train, Y_train, test_data = get_datasets(climate_variables)

    num_fidelities = 2
    num_input_parameters = len(climate_variables) + 2

    gpy_linear_mf_model = get_multifidelity_model(
        kernel_name, num_input_parameters, num_fidelities, X_train, Y_train
    )

    gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

    gpy_linear_mf_model.optimize()

    model = GPyMultiOutputWrapper(gpy_linear_mf_model, 1, 1)

    process_results(
        model,
        model.X[:, :2],
        num_samples=200,
        climate_variables=climate_variables,
        plot_enabled=plotting,
        add_multifidelity_column=True,
        test_data=test_data,
    )


if __name__ == "__main__":
    run_multifidelity_analysis()
