FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mice.py .
COPY random_forest_model.joblib .
COPY scaler.joblib .
COPY model_features.pkl .

EXPOSE 8000

CMD ["uvicorn", "mice:app", "--host", "0.0.0.0", "--port", "8000"]
