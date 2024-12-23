End-to-End Machine Learning Pipeline Project

Project Description
This project offers a comprehensive framework for building and deploying an end-to-end machine learning (ML) pipeline that follows industry-standard MLOps best practices. It specificaly create a pipeline for binary classification.

The train dataset can be found on kaggle: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data 

Example usage:
command: curl -X POST "http://127.0.0.1:8000/api/predict" \-H "Content-Type: application/json" \-d @payload.json
we are predicting the loan status for the idividual in this json file the result was 0. the loan will not be accepted.

CLI Usage
1.	Training the Model: poetry run mlops_train --config config/config.yaml
Or config/config_dev.yaml Or onfig/config_prod.yaml
This will initiate the model training process using the selected configurations.
2.	Inference: poetry run mlops_project --config config/config.yaml
Or config/config_dev.yaml Or onfig/config_prod.yaml
This will generate predictions using the trained model on new input data.

NOTE: please train the model and register it before using the inference CLI as the model used for predictions is the model with the best accuracy registered on mlflow.

Contributing
We welcome contributions! If you want to contribute to this project, follow these steps:
1.	Fork the repository.
2.	Clone your fork locally and create a new branch.
3.	Make your changes and commit them with descriptive messages.
4.	Push your changes to your fork.
5.	Create a Pull Request for review and merging.


