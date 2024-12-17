from .logistic_model import LogisticRegressionModel
from .svc_model import SVCModel
from .base_model import Model

class ModelFactory:
    @staticmethod
    def get_model(model_type: str) -> Model:
        if model_type == "logistic":
            return LogisticRegressionModel()
        elif model_type == "svc":
            return SVCModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")