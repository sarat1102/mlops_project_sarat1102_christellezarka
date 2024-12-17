import pandas as pd
from .base_transformer import DataTransformer


class DataPreprocessing(DataTransformer):
    """A transformer that preprocess the data."""
    @staticmethod
    def transform(data: pd.DataFrame) -> pd.DataFrame:
        """Transforms to clean the input data.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed cleaned data.
        """
        """Remove duplicate rows and rows with missing values
        from the DataFrame."""
        data = data.drop_duplicates()
        data = data.dropna()
        
        """Preprocess categorical variable."""
        # Identify categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # Preprocessing categorical data using One-Hot Encoding
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        data = data.apply(pd.to_numeric)

        return data