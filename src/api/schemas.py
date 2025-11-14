from pydantic import BaseModel, Field
from typing import Dict, List

class DiseasePredictionRequest(BaseModel):
    symptoms: Dict[str, int]

class DiseasePredictionResponse(BaseModel):
    predicted_disease: str
    top_3: List[dict]
    prediction_time: str

class BatchDiseasePredictionRequest(BaseModel):
    items: List[DiseasePredictionRequest]