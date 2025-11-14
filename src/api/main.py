from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List


from .inference import predict_disease, batch_predict
from .schemas import DiseasePredictionRequest, DiseasePredictionResponse


app = FastAPI(
    title="Disease Prediction API",
    description=(
        "Predict disease based on binary symptom indicators (0/1). "
        "Model: RandomForest trained on Kaggle Disease Prediction dataset."
    ),
    version="1.0.0",
    contact={
        "name": "Junaid Khan",
        "url": "https://dagshub.com/jkhan2211/DiseaseFeatureClassifiers",
    },
)

# CORS (you can tighten this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=DiseasePredictionResponse)
async def predict(request: DiseasePredictionRequest):
    return predict_disease(request)

@app.post("/batch-predict", response_model=List[DiseasePredictionResponse])
async def batch_predict_endpoint(requests: List[DiseasePredictionRequest]):
    return batch_predict(requests)