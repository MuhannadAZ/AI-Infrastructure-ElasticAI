
from pathlib import Path
from functools import lru_cache

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="ElasticAI", version="1.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)


@lru_cache
def load_model():
    model_path = Path(__file__).with_name("model.joblib")
    return joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "Welcome to ElasticAI!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    model = load_model()

    prediction = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]

    return {
        "text": request.text,
        "sentiment": str(prediction),
        "confidence": round(float(max(probabilities)), 3)
    }
