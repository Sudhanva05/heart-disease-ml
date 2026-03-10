from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Create FastAPI app
app = FastAPI(title="Heart Disease Prediction API")


# Load trained model
model = joblib.load("models/heart_model.pkl")


# Input data schema
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# Root endpoint
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):

    features = np.array([
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result
    }