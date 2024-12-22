from __future__ import annotations

import pandas as pd
from loguru import logger
import numpy as np
from mlops_project_sarat1102_christellezarka.config import (
    ModelConfig,
    TransformationConfig,
)
from mlops_project_sarat1102_christellezarka.data_transform.base_transformer import (
    DataTransformer,
)
from mlops_project_sarat1102_christellezarka.data_transform.factory import (
    TransformerFactory,
)
from mlops_project_sarat1102_christellezarka.data_transform.data_preprocessing import (
    DataPreprocessing,
)
from mlops_project_sarat1102_christellezarka.model.base_model import Model
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient


# Retrieve the model with the highest accuracy from MLflow
def get_best_model_from_mlflow(experiment_name: str):
    """
    Fetches the model with the highest accuracy from the specified MLflow experiment.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Get the experiment ID from the name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    experiment_id = experiment.experiment_id

    # Search runs and find the best model by accuracy
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        order_by=["metrics.accuracy DESC"],  # Sort by accuracy in descending order
    )
    if not runs:
        raise ValueError("No runs found in the experiment.")

    best_run = runs[0]
    logger.info(
        f"Best run ID: {best_run.info.run_id}, Accuracy: {best_run.data.metrics['accuracy']}"
    )

    # Load the best model
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    return mlflow.sklearn.load_model(best_model_uri), best_run


def load_pipeline(
    transformation_config: TransformationConfig, model_config: ModelConfig
) -> InferencePipeline:
    data_transformer = TransformerFactory.get_transformer(
        transformation_config.scaling_method
    )
    # Update pipeline to use the best model
    logger.info("Loading the best model from MLflow.")
    try:
        model, best_run = get_best_model_from_mlflow(
            "mlops_project_sarat1102_christellezarka_experiments"
        )
        logger.info("Successfully loaded the best model.")
    except Exception as e:
        logger.error(f"Failed to load the best model from MLflow: {e}")
        raise

    # model = ModelFactory.get_model(model_config.type)
    return InferencePipeline(data_transformer, model)


class InferencePipeline:
    _data_transformer: DataTransformer
    _model: Model

    def __init__(self, data_transformer: DataTransformer, model: Model):
        self._data_transformer = data_transformer
        self._model = model

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs the inference data pipeline on the input data.

        Args:
            data (pd.DataFrame): The input data to process.

        Returns:
            pd.DataFrame: The processed data.
        """
        allcols = [
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "person_gender_male",
            "person_education_Bachelor",
            "person_education_Doctorate",
            "person_education_High School",
            "person_education_Master",
            "person_home_ownership_OTHER",
            "person_home_ownership_OWN",
            "person_home_ownership_RENT",
            "loan_intent_EDUCATION",
            "loan_intent_HOMEIMPROVEMENT",
            "loan_intent_MEDICAL",
            "loan_intent_PERSONAL",
            "loan_intent_VENTURE",
            "previous_loan_defaults_on_file_Yes",
        ]

        try:
            logger.info("Pipeline execution started.")

            logger.info("Applying Data transformation.")
            pre_data = DataPreprocessing.transform(data)
            transformed_data = self._data_transformer.transform(pre_data)
            for col in allcols:
                if col not in pre_data.columns:
                    transformed_data[col] = np.zeros(transformed_data.shape[0])
            transformed_data = pd.DataFrame(transformed_data[allcols])
            logger.info("Data transformed successfully.")

            logger.info("Running Inference.")

            predictions = self._model.predict(transformed_data)
            logger.info("Model prediction completed successfully.")

            logger.info("Pipeline execution completed.")
            return pd.DataFrame(predictions)

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            return pd.DataFrame()
