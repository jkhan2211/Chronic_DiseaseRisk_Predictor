# src/features/engineer.py

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ───────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature-engineering")


# ───────────────────────────────────────────────────────────────
# 1. FEATURE CREATION
# ───────────────────────────────────────────────────────────────
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for disease classification data.
    Includes:
        - Binary symptom normalization
        - Symptom count
        - Severity score (frequency-based)
        - Symptom grouping features
    """
    logger.info("Creating engineered features...")
    df = df.copy()

    # Remove unnamed junk columns
    junk_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if junk_cols:
        df.drop(columns=junk_cols, inplace=True)
        logger.info(f"Dropped junk columns: {junk_cols}")

    # Identify symptom columns (all except prognosis)
    symptom_cols = [c for c in df.columns if c != "prognosis"]

    # Convert symptom values to numeric 0/1
    df[symptom_cols] = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[symptom_cols] = df[symptom_cols].clip(0, 1)

    # ───────────────────────────────────────────────
    # Symptom Count
    # ───────────────────────────────────────────────
    df["symptom_count"] = df[symptom_cols].sum(axis=1)

    # ───────────────────────────────────────────────
    # Severity Score (rarer symptoms get higher weight)
    # ───────────────────────────────────────────────
    logger.info("Computing severity score from frequency weights...")
    freq = df[symptom_cols].mean()
    weights = 1 / (freq + 0.0001)
    df["severity_score"] = df[symptom_cols].mul(weights).sum(axis=1)

    # ───────────────────────────────────────────────
    # Symptom Groups
    # ───────────────────────────────────────────────
    chronic = ["fatigue", "weight_loss", "lethargy", "anxiety", "mood_swings"]
    acute = ["fever", "chills", "shivering", "vomiting", "high_fever"]
    pain = ["joint_pain", "stomach_pain", "headache", "back_pain"]
    mental = ["anxiety", "depression", "irritability", "restlessness"]
    infection = ["cough", "cold_hands_and_feets", "breathlessness", "swelled_lymph_nodes"]
    skin = ["skin_rash", "itching", "blister", "skin_peeling", "red_sore_around_nose"]

    def safe_sum(cols):
        existing = [c for c in cols if c in df.columns]
        return df[existing].sum(axis=1) if existing else 0

    df["chronic_score"] = safe_sum(chronic)
    df["acute_score"] = safe_sum(acute)
    df["pain_score"] = safe_sum(pain)
    df["mental_score"] = safe_sum(mental)
    df["infection_score"] = safe_sum(infection)
    df["skin_score"] = safe_sum(skin)

    logger.info("Feature engineering complete.")
    return df


# ───────────────────────────────────────────────────────────────
# 2. PREPROCESSOR CREATION
# ───────────────────────────────────────────────────────────────
def create_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """
    Create a numeric preprocessing pipeline.
    All features are numeric; apply imputation + scaling.
    """
    logger.info("Creating preprocessing pipeline...")

    numeric_features = [c for c in feature_df.columns if c != "prognosis"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    return preprocessor


# ───────────────────────────────────────────────────────────────
# 3. FULL PIPELINE EXECUTION
# ───────────────────────────────────────────────────────────────
def run_feature_engineering(input_file, output_file, preprocessor_file):
    logger.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)

    # Create features
    df_feat = create_features(df)
    logger.info(f"Feature DataFrame shape: {df_feat.shape}")

    # Separate X/y
    X = df_feat.drop(columns=["prognosis"])
    y = df_feat["prognosis"]

    # Create + fit the preprocessor
    preprocessor = create_preprocessor(df_feat)
    X_transformed = preprocessor.fit_transform(X)

    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor → {preprocessor_file}")

    # Save processed dataset
    df_out = pd.DataFrame(X_transformed)
    df_out["prognosis"] = y.values
    df_out.to_csv(output_file, index=False)

    logger.info(f"Saved processed dataset → {output_file}")
    return df_out


# ───────────────────────────────────────────────────────────────
# 4. CLI ENTRY POINT
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering for disease dataset")
    parser.add_argument("--input", required=True, help="Path to Training.csv")
    parser.add_argument("--output", required=True, help="Path for output processed CSV")
    parser.add_argument("--preprocessor", required=True, help="Where to save the preprocessor .pkl")

    args = parser.parse_args()

    run_feature_engineering(args.input, args.output, args.preprocessor)
