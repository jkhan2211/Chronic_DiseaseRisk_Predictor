import json
import joblib
import pandas as pd
from datetime import datetime
from .schemas import DiseasePredictionRequest, DiseasePredictionResponse

# -------------------------------------------------------------------
# Load model + feature order
# -------------------------------------------------------------------

MODEL_PATH = "models/trained/DiseaseRandomForest_v1_final.pkl"
FEATURE_ORDER_PATH = "models/trained/feature_order.json"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

try:
    with open(FEATURE_ORDER_PATH, "r") as f:
        FEATURE_ORDER = json.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Could not load feature_order.json: {e}")


# -------------------------------------------------------------------
# Helper: Convert incoming symptoms → ordered feature vector
# -------------------------------------------------------------------

def prepare_input(symptom_dict: dict):
    """
    Incoming JSON contains a subset of symptoms.
    We must build a row matching the exact 131-feature order.
    Missing symptoms = 0.
    """

    row = {}

    for feat in FEATURE_ORDER:
        row[feat] = int(symptom_dict.get(feat, 0))  # default 0 if not sent

    return pd.DataFrame([row])


# -------------------------------------------------------------------
# Main prediction function
# -------------------------------------------------------------------

def predict_disease(request: DiseasePredictionRequest) -> DiseasePredictionResponse:

    df = prepare_input(request.symptoms)

    # No preprocessor → use raw features
    processed = df

    # Predict
    probs = model.predict_proba(processed)[0]
    diseases = model.classes_

    # Build probability dict
    prob_dict = {diseases[i]: float(round(probs[i], 3)) for i in range(len(diseases))}

    # Pick top 3
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_formatted = [{"disease": d, "probability": p} for d, p in top3]

    return DiseasePredictionResponse(
        predicted_disease=top3[0][0],
        top_3=top3_formatted,
        prediction_time=datetime.now().isoformat()
    )


# -------------------------------------------------------------------
# Batch prediction
# -------------------------------------------------------------------

def batch_predict(requests: list[DiseasePredictionRequest]):
    outputs = []

    for req in requests:
        pred = predict_disease(req)
        outputs.append(pred.dict())

    return outputs
