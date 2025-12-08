import os
import joblib
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response


# LOAD MODEL
MODEL_PATH = "C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Monitoring dan Logging\model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# FASTAPI APP
app = FastAPI(title="Model Serving with Prometheus Metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PROMETHEUS METRICS
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests received")
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Latency for inference requests")
ERROR_COUNT = Counter("inference_error_total", "Total inference errors")
PRED_MEAN = Gauge("prediction_mean", "Mean of last predictions")
PRED_STD = Gauge("prediction_std", "Std dev of last predictions")

prediction_buffer = []


# PAYLOAD FORMAT
class InputData(BaseModel):
    inputs: list


# INFERENCE API
@app.post("/predict")
def predict(data: InputData):
    try:
        REQUEST_COUNT.inc()

        with REQUEST_LATENCY.time():
            arr = np.array(data.inputs)
            preds = model.predict(arr)

            prediction_buffer.extend(list(preds))
            if len(prediction_buffer) > 100:
                prediction_buffer[:] = prediction_buffer[-100:]

            PRED_MEAN.set(float(np.mean(prediction_buffer)))
            PRED_STD.set(float(np.std(prediction_buffer)))

        return {"predictions": preds.tolist()}

    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))



# PROMETHEUS ENDPOINT
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



# RUN UVICORN
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
