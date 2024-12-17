# src/ml_data_pipeline/data_loader/json_loader.py
import pandas as pd
import json
from .base_loader import DataLoader


class JSONLoader(DataLoader):
    def load_data(self, file_path: str)-> pd.DataFrame:
        """
        Load data from a JSON file.
        Args:
            file_path(str): The path to the JSON file.
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)