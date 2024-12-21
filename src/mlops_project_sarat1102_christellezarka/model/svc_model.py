import pandas as pd
from .base_model import Model
from sklearn.svm import SVC
from loguru import logger


class SVCModel(Model):
    """A svc model for training and prediction."""

    def __init__(self, **kwargs):
        """
        Initializes the SVC with the given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the SVC.
        """
        self.model = SVC(**kwargs)
        logger.info("Initialized SVC with parameters: {}", kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the SVC model on the provided data.
        Args:
            X(pd.DataFrame): A pandas DataFrame containing the training data.
            y(pd.Series): A pandas Series containing the labels of the training data.
        """
        logger.info("Training svc model")
        try:
            self.model.fit(X, y)
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error("Error during model training: {}", e)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the SVC model.
        Args:
            X(pd.DataFrame): A pandas DataFrame containing the data for prediction.
        Returns:
            pd.Series: A Series with model predictions.
        """
        logger.info(
            "Making predictions on data with {} samples and {} features",
            X.shape[0],
            X.shape[1],
        )
        try:
            predictions = self.model.predict(X)
            logger.info("Predictions completed")
            return pd.Series(predictions, index=X.index)
        except Exception as e:
            logger.error("Error during prediction: {}", e)
            raise
