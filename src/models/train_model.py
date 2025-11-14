import argparse
import os
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt


# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------
# Remove unnamed CSV junk columns
# ----------------------------------------------------
def clean_unnamed(df):
    drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if drop_cols:
        logger.info(f"Dropping unnamed columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


# ----------------------------------------------------
# Notebook-style global noise (15%)
# ----------------------------------------------------
def apply_notebook_noise(X, flip_percent=0.15, seed=42):
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()

    n_cells = X_noisy.size
    n_flip = int(flip_percent * n_cells)

    flip_indices = rng.choice(n_cells, size=n_flip, replace=False)
    rows, cols = np.unravel_index(flip_indices, X_noisy.shape)
    X_noisy.values[rows, cols] = 1 - X_noisy.values[rows, cols]

    return X_noisy


# ----------------------------------------------------
# Parse arguments
# ----------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train disease classification model using notebook-accurate logic."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)   # Training.csv
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


# ----------------------------------------------------
# Main training logic
# ----------------------------------------------------
def main(args):

    # Load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    aug_cfg = config.get("augmentation", {})
    target = model_cfg["target_variable"]

    # ------------------------------------------------
    # Initialize DagsHub MLflow
    # ------------------------------------------------
    dagshub.init(
        repo_owner="jkhan2211",
        repo_name="DiseaseFeatureClassifiers",
        mlflow=True
    )
    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        logger.info(f"Tracking URI overridden → {args.mlflow_tracking_uri}")

    mlflow.set_experiment(model_cfg["name"])

    # ------------------------------------------------
    # Load ONLY Training.csv (as notebook did)
    # ------------------------------------------------
    train_path = args.data
    logger.info(f"Loading training data from: {train_path}")

    df_train = pd.read_csv(train_path)

    # Clean unnamed junk
    df_train = clean_unnamed(df_train)

    # Drop fluid_overload if present
    if "fluid_overload" in df_train.columns:
        df_train = df_train.drop(columns=["fluid_overload"])
        logger.info("Dropped fluid_overload")

    mergedDF = df_train.copy()
    logger.info(f"Training dataframe shape: {mergedDF.shape}")

    # ------------------------------------------------
    # Prepare X and y
    # ------------------------------------------------
    X = mergedDF.drop(columns=[target])
    y = mergedDF[target]

    # ------------------------------------------------
    # Apply 15% global noise flip (notebook logic)
    # ------------------------------------------------
    if aug_cfg.get("enabled", True):
        flip_percent = aug_cfg.get("flip_percent", 0.15)
        logger.info(f"Applying global noise: {flip_percent*100}% of all cells")
        X = apply_notebook_noise(X, flip_percent=flip_percent)

    # ------------------------------------------------
    # Train/test split EXACTLY like notebook
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ------------------------------------------------
    # Build EXACT notebook RF model
    # ------------------------------------------------
    params = model_cfg["parameters"]
    model = RandomForestClassifier(**params)
    logger.info(f"RF Params: {params}")

    # ------------------------------------------------
    # MLflow run
    # ------------------------------------------------
    with mlflow.start_run(run_name="rf_notebook_exact"):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        })

        logger.info(f"Accuracy = {accuracy:.4f}")

        # Confusion matrix
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Classification report
        mlflow.log_text(
            classification_report(y_test, y_pred),
            "classification_report.txt"
        )

        # Register model
        mlflow.sklearn.log_model(model, "rf_model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/rf_model"
        client = MlflowClient()

        model_name = model_cfg["name"]

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )

        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
        )

        # Save locally
        save_dir = os.path.join(args.models_dir, "trained")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Saved model to → {model_path}")


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
