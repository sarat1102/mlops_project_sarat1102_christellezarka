from .data_preprocessing import DataPreprocessing
from .base_transformer import DataTransformer
from .minmax_scaler import MinMaxScalerTransformer
from .standard_scaler import StandardScalerTransformer

class TransformerFactory:
    """Factory class to create transformer instances based on the scaling method."""

    @staticmethod
    def get_transformer(scaling_method: str) -> DataTransformer:
        """Returns an instance of a transformer based on the provided scaling method.

        Args:
            scaling_method (str): The scaling method to use. Options are "standard" or "minmax".

        Returns:
            DataTransformer: An instance of the requested transformer.

        Raises:
            ValueError: If the scaling method is unsupported.
        """
        if scaling_method == "standard":
            return StandardScalerTransformer()
        elif scaling_method == "minmax":
            return MinMaxScalerTransformer()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")