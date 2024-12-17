import pandas as pd
from .base_transformer import DataTransformer


class DataPreprocessing(DataTransformer):
    @staticmethod
    def transform(data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        :param data: The dataframe
        :return: The dataframe preprocessed
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