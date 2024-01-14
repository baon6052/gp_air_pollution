from copy import deepcopy

import GPy
import numpy as np
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.test_functions.forrester import forrester_low, forrester

from dataset import get_cached_openweather_data, get_cached_air_pollution_data
from gaussian_process import process_results


# from gaussian_process import run_model


def run_multifidelity_analysis():

    climate_variables = [
        'temp'
    ]
    # mae, model, parameter_space = run_model(
    #     climate_variables=climate_variables,
    #     num_samples=5,
    #     plotting=False,
    #     acquisition_function="ModelVariance",
    #     kernel_name="Matern52",
    # )

    # x_train_l = np.atleast_2d(np.random.rand(12)).T
    # x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:6])

    num_values_high_fidelity = 2
    num_values_low_fidelity = 10

    model_inputs = get_cached_openweather_data(num_samples = num_values_low_fidelity, climate_variables=climate_variables)
    ground_truth = get_cached_air_pollution_data(num_samples = num_values_low_fidelity)

    x_low_fidelity = deepcopy(model_inputs)[:num_values_low_fidelity]
    y_low_fidelity = deepcopy(ground_truth)[:num_values_low_fidelity]

    std_y_low_fidelity = np.std(y_low_fidelity)
    mean_y_low_fidelity = np.mean(y_low_fidelity)
    random_noise_list = np.random.normal(mean_y_low_fidelity, std_y_low_fidelity, len(y_low_fidelity))
    y_low_fidelity = np.array([y_value + random_noise for y_value, random_noise in zip(y_low_fidelity, random_noise_list)])

    high_fidelity_dataset = deepcopy(np.random.permutation([[x_sample, y_sample] for x_sample, y_sample in zip(model_inputs, ground_truth)]))[:num_values_high_fidelity]

    x_high_fidelity = np.array([item[0] for item in high_fidelity_dataset[:, :1]])
    y_high_fidelity = np.array([item[0] for item in high_fidelity_dataset[:, 1:]])

    # breakpoint()
    X_train, Y_train = convert_xy_lists_to_arrays([x_high_fidelity, x_low_fidelity], [y_high_fidelity, y_low_fidelity])

    num_fidelities = 2
    kernels = [GPy.kern.RBF(len(climate_variables) + 2), GPy.kern.RBF(len(climate_variables) + 2)]
    linear_mf_kernel = LinearMultiFidelityKernel(kernels)
    gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, linear_mf_kernel, n_fidelities=num_fidelities)

    gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

    print("Chk1")
    gpy_linear_mf_model.optimize()
    print("Chk2")

    model = GPyMultiOutputWrapper(gpy_linear_mf_model, 1, 1)
    out = model.predict(X_train)
    breakpoint()
    mae = process_results(
        model,
        model.X[:, :2],
        num_samples=1,
        climate_variables=climate_variables,
        plot_enabled=True,
    )


if __name__ == "__main__":
    run_multifidelity_analysis()