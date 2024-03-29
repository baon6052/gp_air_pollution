from typing import Callable

import numpy as np
from emukit.core import ParameterSpace
from emukit.core.interfaces import IModel
from emukit.core.loop import UserFunctionWrapper
from emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity

from dataset import get_cached_openweather_data


class CustomModelFreeMonteCarloSensitivity(ModelFreeMonteCarloSensitivity):
    """
    Class to do sensitivity analysis of a function. It computes Monte Carlo approximations to
    the Sobol indexes and the total variance components of each input variable of some objective
    of interest.
    """

    def __init__(
        self, objective: Callable, input_domain: ParameterSpace
    ) -> None:
        """
        :param objective: python function in which the sensitivity analysis will be performed.
        :param input_domain: parameter space.
        """
        self.objective = UserFunctionWrapper(objective)
        self.input_domain = input_domain

        super().__init__(objective, input_domain)

    def _generate_samples(
        self,
        num_monte_carlo_points: int = int(1e5),
        climate_variables: list[str] = [],
    ) -> None:
        """
        Generates the two samples that are used to compute the main and total indices

        :param num_monte_carlo_points: number of samples to generate
        """
        # self.main_sample = self.input_domain.sample_uniform(num_monte_carlo_points)

        data, _ = get_cached_openweather_data(
            num_monte_carlo_points, climate_variables
        )
        # self.main_sample = get_cached_openweather_data(
        #     num_monte_carlo_points, climate_variables
        # )
        main_idx = np.random.randint(0, data.shape[0], num_monte_carlo_points)
        fixing_idx = np.random.randint(0, data.shape[0], num_monte_carlo_points)
        self.main_sample = data[main_idx]
        self.fixing_sample = data[fixing_idx]

    def compute_effects(
        self,
        main_sample: np.ndarray = None,
        fixing_sample: np.ndarray = None,
        num_monte_carlo_points: int = int(1e5),
        climate_variables: list[str] = [],
    ) -> tuple:
        """
        Computes the main and total effects using Monte Carlo and a give number of samples.
        - Main effects: contribution of x_j alone to the variance of f.
        - Total effects: contribution to all Sobol terms in which x_j is involved to the variance of f.

        The (unbiased) Monte Carlo estimates are computed using:

        "A. Saltelli, Making best use of model evaluations to compute sensitivity indices, Computer Physics Com.
        608 munications, 145 (2002), pp. 280-297"

        :param main_sample: original sample that is used in the Monte Carlo computations.
        :param fixing_sample: supplementary sample that is used in the Monte Carlo computations.
        :param num_monte_carlo_points: number of points used to compute the effects.

        :return: A tuple (main effects, total effects, total variance).
        """
        if main_sample is None or fixing_sample is None:
            self.num_monte_carlo_points = num_monte_carlo_points
            self._generate_samples(
                self.num_monte_carlo_points, climate_variables
            )
        else:
            self.main_sample = main_sample
            self.fixing_sample = fixing_sample
            self.num_monte_carlo_points = self.main_sample.shape[0]

        f_main_sample = self.objective.f(self.main_sample)
        f_fixing_sample = self.objective.f(self.fixing_sample)

        total_mean, total_variance = self.compute_statistics(f_main_sample)
        variable_names = ["longitude", "latitude"]
        variable_names.extend(climate_variables)

        main_effects = {}
        total_effects = {}
        var_index = 0

        for variable in variable_names:
            # --- All columns are the same but the one of interest that is replaced by the original sample
            self.new_fixing_sample = self.fixing_sample.copy()
            self.new_fixing_sample[:, var_index] = self.main_sample[
                :, var_index
            ]

            # --- Evaluate the objective at the new fixing sample
            f_new_fixing_sample = self.objective.f(self.new_fixing_sample)

            # --- Compute the main and total variances
            (
                variable_main_variance,
                variable_total_variance,
            ) = self.saltelli_estimators(
                f_main_sample,
                f_fixing_sample,
                f_new_fixing_sample,
                self.num_monte_carlo_points,
                total_mean,
                total_variance,
            )

            # --- Compute the effects
            main_effects[variable] = variable_main_variance / total_variance
            total_effects[variable] = variable_total_variance / total_variance

            var_index += 1
        return main_effects, total_effects, total_variance


class CustomMonteCarloSensitivity(CustomModelFreeMonteCarloSensitivity):
    """
    Class to compute the sensitivity coefficients of given model. This class wraps the model and calls the mean
    predictions that are used to compute the sensitivity inputs using Monte Carlo.
    """

    def __init__(self, model: IModel, input_domain: ParameterSpace) -> None:
        """
        :param model: model wrapper with the interface IModel.
        :param input_domain: space class.
        """

        self.model = model
        self.model_objective = lambda x: self.model.predict(x)[0]

        super().__init__(self.model_objective, input_domain)
