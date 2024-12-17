# src/ml_data_pipeline/data_loader/factory.py
from .csv_loader import CSVLoader
from .json_loader import JSONLoader
from .base_loader import DataLoader


class DataLoaderFactory:
    """Factory class to create data loader instances based on the file type."""

    @staticmethod
    def get_data_loader(file_type: str) -> DataLoader:
        """Returns an instance of a data loader based on the provided file type.

        Args:
            file_type (str): The type of file to load. Options are "csv" or "json".

        Returns:
            DataLoader: An instance of the requested data loader.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if file_type == "csv":
            return CSVLoader()
        elif file_type == "json":
            return JSONLoader()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
