from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# Request schema
class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex: str          # "male" or "female"
    smoker: str       # "yes" or "no"
    region: str       # "southeast", "southwest", "northwest", "northeast"

@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        sex_male = 1 if data.sex.lower() == "male" else 0
        smoker_yes = 1 if data.smoker.lower() == "yes" else 0

        region = data.region.lower()
        region_southeast = 1 if region == "southeast" else 0
        region_southwest = 1 if region == "southwest" else 0
        region_northwest = 1 if region == "northwest" else 0

        age_smoker = data.age * smoker_yes
        bmi_smoker = data.bmi * smoker_yes

        features = [
            data.age,
            data.bmi,
            data.children,
            sex_male,
            smoker_yes,
            region_northwest,
            region_southeast,
            region_southwest,
            age_smoker,
            bmi_smoker
        ]

        input_df = pd.DataFrame([features], columns=scaler.feature_names_in_)

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        return {"estimated_cost": round(prediction, 2)}
    
    except Exception as e:
        return {"error": str(e)}

