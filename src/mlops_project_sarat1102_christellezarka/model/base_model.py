from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
