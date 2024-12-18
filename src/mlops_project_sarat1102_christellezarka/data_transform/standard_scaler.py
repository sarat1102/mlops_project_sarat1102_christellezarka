import pandas as pd
from sklearn.preprocessing import StandardScaler
from .base_transformer import DataTransformer
from loguru import logger


class StandardScalerTransformer(DataTransformer):
    """A transformer that scales data using Standard scaling (z-score normalization)."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data using Standard scaling.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with standardized values.
        """
        logger.info(f"transforming data using standard scaler")
        try: 
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            logger.info(f"Successfully transformed data.")
            return pd.DataFrame(scaled_data, columns=data.columns)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
