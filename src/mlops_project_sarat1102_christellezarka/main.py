import argparse
from mlops_project_sarat1102_christellezarka.config import load_config
from mlops_project_sarat1102_christellezarka.data_loader import DataLoaderFactory
from mlops_project_sarat1102_christellezarka.data_transform import TransformerFactory
from mlops_project_sarat1102_christellezarka.data_transform import DataPreprocessing
from mlops_project_sarat1102_christellezarka.model import ModelFactory
from sklearn.model_selection import train_test_split

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
    pre_data= DataPreprocessing.transform(data)
    X = pre_data.drop(columns=["loan_status"])
    y = pre_data["loan_status"]
    transformer = TransformerFactory.get_transformer(config.transformation.scaling_method)
    transformed_data = transformer.transform(pre_data)
    print("Transformed Data:")
    print(transformed_data)

    # Prepare train-test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )


    # Use ModelFactory to select and train the model
    model = ModelFactory.get_model(config.model.type)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()