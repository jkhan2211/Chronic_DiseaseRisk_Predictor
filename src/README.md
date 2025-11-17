# 📁 src/ Directory – Source Code Overview

This folder contains all source code for the Disease Feature Classifiers project. It includes feature engineering, model training operations, API services, and individual experiment notebooks contributed by team members.

This document helps new developers understand the structure and navigate the codebase efficiently

## 📂 Directory Structure
```
src/
│
├── api/
│   ├── inference.py
│   ├── main.py
│   ├── requirements.txt
│   └── schemas.py
│
├── Experiments_JunaidKhan/
│   ├── 00_data_engineering_experiment.ipynb
│   └── 01_experimenting.ipynb
│
├── Experiments_NO/
│   ├── KmeansBernoulli_preprocessing.ipynb
│   ├── Model_BernoulliNB.ipynb
│   ├── Model_Kmeans_no_dupls.ipynb
│   └── Model_Kmeans.ipynb
│
├── features/
│   └── engineering.py
│
├── models/
│   └── train_model.py
│
├── models_operations/
│   ├── generate_feature_order.py
│   └── train_model.py

```

# 📘 Folder-by-Folder Description
## 📁 api/ : Model Inference & FastAPI Service
The api/ directory provides the complete deployment layer for the disease prediction system. It includes:
- A FastAPI service (main.py)
    - Defines the HTTP API and routes for predicting diseases.
    - Responsibilities:
        - Initializes FastAPI with metadata
        - Enables CORS for frontend access
    - Implements:
        - /health -> Health check endpoint
        - /predict -> Single prediction
        - /batch-predict -> Bulk prediction
    All prediction logic is delegated to inference.py

- The core inference engine (inference.py)
    - Contains the core ML logic.
    - Functions include:
        - Loading model + feature order
        - Preparing input into the exact ordered 131-feature vector
        - Predicting disease probabilities
        - Returning top-3 predictions
        - Batch prediction utilities
    - This file ensures model consistency and handles all preprocessing required for inference

- Data validation schemas (schemas.py)
    - Defines the structured request/response formats:
    - DiseasePredictionRequest -> Symptom dictionary input
    - DiseasePredictionResponse -> Predicted disease + top-3 probabilities
    - BatchDiseasePredictionRequest -> List of multiple requests
    - These schemas enforce correct input format and generate clean documentation in Swagger (/docs)

- Runtime dependencies (requirements.txt)

- API Workflow
    ```
    Request JSON
        ↓
    schemas.py → validate & normalize
        ↓
    inference.py → prepare features → model.predict()
        ↓
    main.py → return structured response
        ↓
    API Response JSON

    ```

## 📁 Experiments_JunaidKhan/
A sandbox for experiment notebooks:
Data engineering tests and Initial model trials
These notebooks capture experiment history and support reproducibility

## 📁 features/engineering.py
Contains utilities for:
- Feature engineering, Preprocessing and Transformations used before training models
    1. Feature Creation
    Generates enriched numerical features from the raw binary symptoms
        a. Cleanup
            Removes any Unnamed: ... junk columns
        b. Standardize Symptom Columns
            Converts all symptom values to numeric (0/1)
            Ensures consistent binary format across all symptoms
        c. Engineered Features Added
            The function creates several new derived features:
            - Symptom Count : Total number of symptoms marked as 1
            - Severity Score : Weighted severity score where rarer symptoms contribute more
            - Symptom Group Scores : Aggregated symptom categories ->
                - chronic_score, acute_score, pain_score, mental_score, infection_score, skin_score
    2. Preprocessor creation
        a. Imputation : ensures no missing values remain in the dataset
        b. Scaling : Normalizes all numeric values for models sensitive to scale
    3. Feature Engineering Pipeline
    Delivers reproducible feature generation and preprocessing for use in model training
        - Load dataset (Training.csv)
        - Apply engineered feature creation
        - Split features/target (prognosis)
        - Fit the preprocessing pipeline
        - Save:
            - The fitted preprocessor (.pkl via joblib)
            - The processed dataset (transformed features + prognosis)
    4. Command Line Interface
    This makes the feature-engineering process scriptable and automatable (e.g., in ML pipelines or CI/CD)
        ```
        python engineer.py --input Training.csv \
                        --output processed.csv \
                        --preprocessor preproc.pkl

        ```
- Summary
    ```
    Raw Data → Engineered Features → Preprocessing → Saved Dataset + Preprocessor
    ```
## 📁 models_operations/
Contains all advanced model-management components:
train_model.py -> Builds, trains, and evaluates models
    - Applies notebook-style noise augmentation (default 15%)
    - Performs train/test split
    - Logs metrics, artifacts, and model with MLflow
    - Registers model versions in MLflow
    - Saves trained model locally in models/trained/
    ```
    python train_model.py \
    --config config.yaml \  
    --data data/raw_data/Training.csv \
    --models-dir models \
    --mlflow-tracking-uri <optional-mlflow-uri>
    ```
generate_feature_order.py -> Ensures consistent feature ordering (critical for reproducibility)
    - Reads data/raw_data/Training.csv
    - Drops unnecessary columns (prognosis, fluid_overload, unnamed columns)
    - Saves feature order to models/trained/feature_order.json
feature_order.json – Ensures input features match the model’s expected format during inference
- MLflow is used for experiment tracking; configure mlflow-tracking-uri if needed
- Notebook-style noise augmentation (default 15%) is applied to training data for reproducibility
- The model is saved locally and also registered in MLflow for version control
- Ensure data/raw_data/Training.csv exists before running scripts

## Workflow 

```
(features) → (models_operations) → (api)
          ↑
          └── Experiments inform improvements

```
- features prepares the dataset
- models_operations train and evaluate ML models
- api exposes the selected model through FastAPI
- experiments folders capture research & iterative development

## License
This project is licensed under the MIT License