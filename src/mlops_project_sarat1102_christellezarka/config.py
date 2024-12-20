from pydantic import BaseModel, field_validator, validator
from omegaconf import OmegaConf


class DataLoaderConfig(BaseModel):
    """Configuration for the data loader.

    Attributes:
        file_path (str): The path to the data file.
        file_type (str): The type of the data file (csv or json).
    """

    file_path: str
    file_type: str

    @validator("file_type")
    def validate_file_type(cls, value):
        """Validates the file type.

        Args:
            value (str): The file type to validate.

        Returns:
            str: The validated file type.

        Raises:
            ValueError: If the file type is not 'csv' or 'json'.
        """
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value


class TransformationConfig(BaseModel):
    """Configuration for the data transformation.

    Attributes:
        normalize (bool): Whether to normalize the data.
        scaling_method (str): The method to use for scaling (standard or minmax).
    """

    normalize: bool
    scaling_method: str

    @field_validator("scaling_method")
    def validate_scaling_method(cls, value: str) -> str:
        """Validates the scaling method.

        Args:
            value (str): The scaling method to validate.

        Returns:
            str: The validated scaling method.

        Raises:
            ValueError: If the scaling method is not 'standard' or 'minmax'.
        """
        if value not in {"standard", "minmax"}:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        return value


class ModelConfig(BaseModel):
    """Configuration for the model.

    Attributes:
        type (str): The type of the model (linear or tree).
        params (Dict[str, Any]): Additional parameters for the model.
    """

    type: str
    params: Dict[str, Any] = {}

    @field_validator("type")
    def validate_model_type(cls, value: str) -> str:
        """Validates the model type.

        Args:
            value (str): The model type to validate.

        Returns:
            str: The validated model type.

        Raises:
            ValueError: If the model type is not 'logistic' or 'svc'.
        """
        if value not in {"logistic", "svc"}:
            raise ValueError("model type must be 'logistic' or 'svc'")
        return value
    

class MLflowConfig(BaseModel):
    """Configuration for MLflow.

     Attributes:
        tracking_uri: The URI of the server where MLflow's
        tracking service is hosted.
        experiment_name: The name of the MLflow experiment.
    """
    tracking_uri: str
    experiment_name: str


class Config(BaseModel):
    """Overall configuration for the pipeline.

    Attributes:
        data_loader (DataLoaderConfig): Configuration for the data loader.
        transformation (TransformationConfig): Configuration for the data transformer.
        model (ModelConfig): Configuration for the model.
        mlflow (MLflowConfig): Configuration for the mlflow.
    """

    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig
    mlflow: MLflowConfig


def load_config(config_path: str) -> Config:
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config(**config_dict)  # type: ignore
