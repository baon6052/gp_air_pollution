import numpy as np
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.loop import (
    CandidatePointCalculator,
    FixedIntervalUpdater,
    OuterLoop,
)
from emukit.core.loop.loop_state import LoopState, create_loop_state
from emukit.core.optimization import (
    AcquisitionOptimizerBase,
    GradientAcquisitionOptimizer,
)
from emukit.experimental_design.acquisitions import ModelVariance

from dataset import extend_train_data


class CustomSequentialPointCalculator(CandidatePointCalculator):
    """This candidate point calculator chooses one candidate point at a time"""

    def __init__(
        self,
        acquisition: Acquisition,
        acquisition_optimizer: AcquisitionOptimizerBase,
        climate_variables: list[str] = [],
    ) -> None:
        """
        :param acquisition: Acquisition function to maximise
        :param acquisition_optimizer: Optimizer of acquisition function
        """
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.climate_variables = climate_variables

    def compute_next_points(
        self, loop_state: LoopState, context: dict = None
    ) -> np.ndarray:
        """
        Computes point(s) to evaluate next

        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: List of function inputs to evaluate the function at next
        """
        self.acquisition.update_parameters()
        x, _ = self.acquisition_optimizer.optimize(self.acquisition, context)

        # Call out to API here
        x = extend_train_data(x[:, :2], self.climate_variables)

        return x


class CustomExperimentalDesignLoop(OuterLoop):
    def __init__(
        self,
        space: ParameterSpace,
        model: IModel,
        climate_variables: list[str] = [],
        acquisition: Acquisition = None,
        update_interval: int = 1,
        batch_size: int = 1,
        acquisition_optimizer: AcquisitionOptimizerBase = None,
    ):
        """
        An outer loop class for use with Experimental design

        :param space: Definition of domain bounds to collect points within
        :param model: The model that approximates the underlying function
        :param acquisition: experimental design acquisition function object. Default: ModelVariance acquisition
        :param update_interval: How many iterations pass before next model optimization
        :param batch_size: Number of points to collect in a batch. Defaults to one.
        :param acquisition_optimizer: Optimizer selecting next evaluation points
                                      by maximizing acquisition.
                                      Gradient based optimizer is used if None.
                                      Defaults to None.
        """

        if acquisition is None:
            acquisition = ModelVariance(model)

        # This AcquisitionOptimizer object deals with optimizing the acquisition to find the next point to collect
        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)

        # Construct emukit classes
        if batch_size == 1:
            candidate_point_calculator = CustomSequentialPointCalculator(
                acquisition, acquisition_optimizer, climate_variables
            )
        # elif batch_size > 1:
        #     candidate_point_calculator = GreedyBatchPointCalculator(
        #         model, acquisition, acquisition_optimizer, batch_size
        #     )
        else:
            raise ValueError(
                "Batch size value of " + str(batch_size) + " is invalid."
            )

        model_updater = FixedIntervalUpdater(model, update_interval)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
