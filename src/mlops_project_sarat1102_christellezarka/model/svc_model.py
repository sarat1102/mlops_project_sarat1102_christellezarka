import pandas as pd
from .base_model import Model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class SVCModel(Model):
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        """
        Train the SVC model on the provided data.
        Args:
            X(pd.DataFrame): A pandas DataFrame containing the training data.
            y(pd.Series): A pandas Series containing the labels of the training data.
        """
        
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