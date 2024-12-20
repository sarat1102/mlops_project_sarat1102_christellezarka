# src/ml_data_pipeline/data_loader/json_loader.py
import pandas as pd
import json
from .base_loader import DataLoader
from loguru import logger


class JSONLoader(DataLoader):
    """A data loader for loading JSON files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a JSON file.
        Args:
            file_path(str): The path to the JSON file.
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        logger.info(f"Loading data from JSON file at {file_path}")
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            logger.info(f"Successfully loaded data from {file_path}")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
