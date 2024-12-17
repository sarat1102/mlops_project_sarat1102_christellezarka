import pandas as pd
from .base_model import Model
from sklearn.svm import SVC

class SVCModel(Model):
    """A svc model for training and prediction."""
    def __init__(self, **kwargs):
        """
        Initializes the SVC with the given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the SVC.
        """
        self.model = SVC(**kwargs)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the SVC model on the provided data.
        Args:
            X(pd.DataFrame): A pandas DataFrame containing the training data.
            y(pd.Series): A pandas Series containing the labels of the training data.
        """
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the SVC model.
        Args:
            X(pd.DataFrame): A pandas DataFrame containing the data for prediction.
        Returns:
            pd.Series: A Series with model predictions.
        """
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=data.index)
