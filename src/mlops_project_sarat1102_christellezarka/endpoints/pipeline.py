from typing import Any, Dict, List
import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from prometheus_client import Counter, Summary
from pydantic import BaseModel

from mlops_project_sarat1102_christellezarka.config import (
    ModelConfig,
    TransformationConfig,
)
from mlops_project_sarat1102_christellezarka.core import load_pipeline

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "predict_requests_total", "Total number of requests to the predict endpoint"
)
REQUEST_LATENCY = Summary(
    "predict_request_latency_seconds", "Latency of predict requests in seconds"
)
REQUEST_ERRORS = Counter(
    "predict_request_errors_total", "Total number of errors in predict requests"
)


# Input and Output schemas
class PredictInput(BaseModel):
    data: List[Dict[str, Any]]  # List of dictionaries, each representing a row


class PredictOutput(BaseModel):
    predictions: List[Any]  # List of dictionaries with predictions


# Create a router instance
router = APIRouter()


# Instantiate the Pipeline With Default Configuration
TRANSFORMATION_CONFIG = TransformationConfig(scaling_method="standard", normalize=True)
MODEL_CONFIG = ModelConfig(type="logistic", params={})

logger.info("Loading pipeline")
pipeline_endpoint = load_pipeline(TRANSFORMATION_CONFIG, MODEL_CONFIG)
logger.info("successfully loaded pipeline")


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput) -> PredictOutput:
    """
    Converts input JSON to DataFrame, runs the pipeline, and converts output DataFrame to JSON.
    """
    REQUEST_COUNT.inc()  # Increment request count
    with REQUEST_LATENCY.time():
        try:
            # Convert input JSON to pandas DataFrame
            input_df = pd.DataFrame(input_data.data)
            logger.info("Input data converted to DataFrame.")
            predictions_df = pipeline_endpoint.run(input_df)
            return PredictOutput(predictions=list(predictions_df))
        except Exception as e:
            REQUEST_ERRORS.inc()  # Increment error count
            logger.error(f"Error in predict endpoint: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed.")
