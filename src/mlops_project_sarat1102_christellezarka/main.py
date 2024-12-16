import argparse
from mlops_project_sarat1102_christellezarka.config import load_config
from mlops_project_sarat1102_christellezarka.data_loader import DataLoaderFactory
from mlops_project_sarat1102_christellezarka.model import TransformerFactory
from mlops_project_sarat1102_christellezarka.model import ModelFactory

parser = argparse.ArgumentParser(description = "Run the ML data pipeline with specified configuration.")
parser.add_argument(
"--config",
type = str,
required = True,
help = "Path to the configuration YAML file."
)


def main():
    args = parser.parse_args()
    config = load_config(args.config)
    print("Loaded Configuration:")
    print(config)
    
    # Use DataLoaderFactory to load data
    data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
    data = data_loader.load_data(config.data_loader.file_path)
    print("Loaded Data:")
    print(data)

    # Use TransformerFactory to transform data
    transformer = TransformerFactory.get_transformer()
    transformed_data = transformer.transform(data)
    print("Transformed Data:")
    print(transformed_data)

    # Use ModelFactory to select and train the model
    model = ModelFactory.get_model(config.model.type)
    model.train(transformed_data)
    predictions = model.predict(transformed_data)
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()