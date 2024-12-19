import argparse
from loguru import logger
from typing import Dict
from mlops_project_sarat1102_christellezarka.config import load_config
from mlops_project_sarat1102_christellezarka.data_loader import DataLoaderFactory
from mlops_project_sarat1102_christellezarka.data_transform import TransformerFactory
from mlops_project_sarat1102_christellezarka.data_transform import DataPreprocessing
from mlops_project_sarat1102_christellezarka.model import ModelFactory
from sklearn.model_selection import train_test_split

logger.add("logs/pipeline.log", rotation="500 MB") # Log rotation at 500 MB
parser = argparse.ArgumentParser( description="Run the ML data pipeline with specified configuration.")
parser.add_argument( "--config", type=str, required=True, help="Path to the configuration YAML file.")

def main():
    args = parser.parse_args()
    logger.info("Pipeline execution started.")
    config = load_config(args.config)
    logger.info("Loaded configuration successfully.")
    print("Loaded Configuration:")
    print(config)

    # Use DataLoaderFactory to load data
    try: 
        data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
        data = data_loader.load_data(config.data_loader.file_path)
        logger.info("Data loaded successfully.")
        print("Loaded Data:")
        print(data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    # Use TransformerFactory to transform data
    try: 
        pre_data = DataPreprocessing.transform(data)
        X = pre_data.drop(columns=["loan_status"])
        y = pre_data["loan_status"]
        transformer = TransformerFactory.get_transformer(
            config.transformation.scaling_method
        )
        transformed_data = transformer.transform(pre_data)
        print("Transformed Data:")
        print(transformed_data)
        logger.info("Data transformed successfully.")
    except Exception as e:
        logger.error(f"Failed to transform data: {e}")
        return

    # Prepare train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use ModelFactory to select and train the model
    try: 
        model = ModelFactory.get_model(config.model.type)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        print("Predictions:")
        print(predictions)
    except Exception as e:
        logger.error(f"Model training/prediction failed: {e}")
        return

from fastapi import FastAPI

app = FastAPI(title="ML Data Pipeline API", version="1.0")
@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

