# src/ml_data_pipeline/data_loader/csv_loader.py
import pandas as pd
from .base_loader import DataLoader
from loguru import logger

class CSVLoader(DataLoader):
    """A data loader for loading CSV files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        Args:
            file_path(str): The path to the CSV file.
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        logger.info(f"Loading data from CSV file at {file_path}")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise