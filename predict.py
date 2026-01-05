import joblib
import pandas as pd
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

app = FastAPI(title="job-fit-prediction")

# --- Load pipeline ---
with open('model.bin', 'rb') as f_in:
    pipeline = joblib.load(f_in)


def predict_single(candidate: Dict[str, str]) -> float:
    candidate = pd.DataFrame([candidate])
    result = pipeline.predict_proba(candidate)[0, 1]
    return float(result)


@app.post("/predict")
def predict(candidate: Dict[str, str]):
    prob = predict_single(candidate)
    pred  = int(prob >= 0.5)

    return {
        "fit_probability": prob,
        "Prediction": "Good Fit" if pred == 1 else "Bad Fit"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)