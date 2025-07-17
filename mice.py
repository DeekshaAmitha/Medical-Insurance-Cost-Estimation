from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load saved artifacts (make sure these files are alongside mice.py)
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
features = joblib.load('model_features.pkl')

class InsuranceInput(BaseModel):
    age: float
    bmi: float
    children: int
    sex_male: int
    smoker_yes: int
    region_northwest: int
    region_southeast: int
    region_southwest: int
    age_smoker: float
    bmi_smoker: float

@app.post("/predict")
def predict(input: InsuranceInput):
    input_dict = input.dict()
    input_array = np.array([input_dict.get(f, 0) for f in features]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    return {"predicted_charge": prediction}
