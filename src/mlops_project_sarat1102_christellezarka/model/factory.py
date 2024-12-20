from typing import Any, Dict
from .logistic_model import LogisticRegressionModel
from .svc_model import SVCModel
from .base_model import Model


class ModelFactory:
    """Factory class to create model instances based on the model type."""

    @staticmethod
    def get_model(model_type: str) -> Model:
        """Returns an instance of a model based on the provided model type.

        Args:
            model_type (str): The type of model to create. Options are "logistic" or "svc".

        Returns:
            Model: An instance of the requested model type.

        Raises:
            ValueError: If the provided model type is unsupported.
        """

        if model_type == "logistic":
            return LogisticRegressionModel()
        elif model_type == "svc":
            return SVCModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
