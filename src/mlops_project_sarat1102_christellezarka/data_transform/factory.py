from .data_preprocessing import DataPreprocessing
from .base_transformer import DataTransformer


class TransformerFactory:
    @staticmethod
    def get_transformer() -> DataTransformer:
        return DataPreprocessing()