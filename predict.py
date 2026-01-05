import joblib
import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from pydantic import ConfigDict

#request

class CandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume_text: str
    job_description_text: str

#response

class PredictionResponse(BaseModel):
    fit_probability: float
    prediction: str

app = FastAPI(title="job-fit-prediction")

# --- Load pipeline ---
with open('model.bin', 'rb') as f_in:
    pipeline = joblib.load(f_in)


def predict_single(candidate: CandidateRequest) -> float:
    candidate = pd.DataFrame([candidate.model_dump()])
    result = pipeline.predict_proba(candidate)[0, 1]
    return float(result)


@app.post("/predict")
def predict(candidate: CandidateRequest):
    prob = predict_single(candidate)
    pred  = int(prob >= 0.5)
    prediction = "Good Fit" if pred == 1 else "Bad Fit"

    return PredictionResponse(fit_probability=prob, prediction=prediction)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)