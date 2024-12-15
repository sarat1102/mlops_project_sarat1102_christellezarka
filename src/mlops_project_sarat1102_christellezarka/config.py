# src/ml_data_pipeline/config.py
from pydantic import BaseModel, validator
from omegaconf import OmegaConf
import os
class DataLoaderConfig(BaseModel):
    file_path: str
    file_type: str
    @validator("file_type")
    def validate_file_type(cls, value):
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value
class Config(BaseModel):
    data_loader: DataLoaderConfig
def load_config(config_path: str)-> Config:
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config(**config_dict)