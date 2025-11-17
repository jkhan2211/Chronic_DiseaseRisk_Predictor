# Description 
The configs/ folder stores all model configuration files used by the project.
These configurations define the selected model hyperparameters, feature settings, and training options that ensure the ML pipeline is reproducible, consistent, and easy to modify without changing code.

This directory acts as the single source of truth for all model-related decisions used throughout experimentation, training, and deployment.

![MlFlow](../images/mlflow_dags.png)

# Folder Structure

The configs/ folder stores all model configuration files used by the project.
These configurations define the selected model hyperparameters, feature settings, and training options that ensure the ML pipeline is reproducible, consistent, and easy to modify without changing code.

This directory acts as the single source of truth for all model-related decisions used throughout experimentation, training, and deployment.

### About MLflow in This Project

*** MLflow is used here for: ***

- Tracking experiment runs

- Logging metrics (accuracy, recall, precision, F1-score)

- Logging parameters & model configs

- Saving model artifacts

- Comparing multiple algorithms

*** MLflow Dashboard ***

👉 https://dagshub.com/jkhan2211/DiseaseFeatureClassifiers.mlflow/#/experiments/4/runs/556b9ce7cf7144c8b63330c5225d9fb9
