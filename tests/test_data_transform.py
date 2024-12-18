# tests/test_data_transform.py
import pandas as pd
import pytest
from mlops_project_sarat1102_christellezarka.data_transform import TransformerFactory
from mlops_project_sarat1102_christellezarka.data_transform import DataPreprocessing


@pytest.fixture
def sample_data():
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

def test_datapreprocessing(sample_data):
    pre_data = DataPreprocessing.transform(sample_data)
    assert not pre_data.isnull().any().any()
    assert pre_data.select_dtypes(include=["object", "category"]).empty

def test_standard_scaler_transform(sample_data):
    transformer = TransformerFactory.get_transformer("standard")
    transformed_data = transformer.transform(sample_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == sample_data.shape


def test_minmax_scaler_transform(sample_data):
    transformer = TransformerFactory.get_transformer("minmax")
    transformed_data = transformer.transform(sample_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == sample_data.shape
