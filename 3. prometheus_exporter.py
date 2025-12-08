import os
import time
import joblib
import psutil
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CONTENT_TYPE_LATEST,
    generate_latest
)
import uvicorn


# FASTAPI SETUP
app = FastAPI(title="ML Model Exporter", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# LOAD MODEL
MODEL_PATH = "C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Monitoring dan Logging\model.pkl"

if not os.path.exists("C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Monitoring dan Logging\model.pkl"):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

model = joblib.load("C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Monitoring dan Logging\model.pkl")


# PROMETHEUS METRICS
REQUEST_COUNT = Counter("inference_requests_total", "Total request masuk")
REQUEST_SUCCESS = Counter("request_success_count", "Total request berhasil")
REQUEST_FAILURE = Counter("request_failure_count", "Total request gagal")

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Durasi proses prediksi"
)

LATEST_PRED = Gauge("latest_prediction", "Nilai prediksi terakhir")
REQUEST_AVG_LATENCY = Gauge("request_avg_latency", "Rata-rata latency request")

CPU_USAGE = Gauge("cpu_usage_percent", "Persentase penggunaan CPU")
MEMORY_USAGE = Gauge("memory_usage_mb", "Penggunaan memori FastAPI")
DISK_IO = Gauge("disk_io_kb", "Disk IO dalam KB/s")

ERROR_RATE = Gauge("error_rate_percent", "Persentase error")
QUEUE_LENGTH = Gauge("queue_length", "Jumlah request dalam antrean")


# Sliding window latency
latency_history = []


# REQUEST SCHEMA
class InputData(BaseModel):
    data: list


# ROOT ENDPOINT
@app.get("/")
def home():
    return {"message": "Model Serving + Prometheus Exporter aktif!"}


# INFERENCE ENDPOINT
def predict(input_data: InputData):
    REQUEST_COUNT.inc()

    start = time.time()

    try:
        # CPU / memory metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used / 1024 / 1024)
        DISK_IO.set(psutil.disk_io_counters().read_bytes / 1024)

        # queue length (dummy)
        QUEUE_LENGTH.set(np.random.randint(0, 10))

        # inference
        x = np.array(input_data.data).reshape(1, -1)
        pred = model.predict(x)[0]

        # set last prediction
        LATEST_PRED.set(float(pred))

        # record success
        REQUEST_SUCCESS.inc()

        # latency
        duration = time.time() - start
        INFERENCE_LATENCY.observe(duration)

        latency_history.append(duration)
        if len(latency_history) > 50:
            latency_history.pop(0)

        avg_latency = np.mean(latency_history)
        REQUEST_AVG_LATENCY.set(avg_latency)

        # error rate (dummy calculation)
        total_req = REQUEST_SUCCESS._value.get() + REQUEST_FAILURE._value.get()
        if total_req > 0:
            ERROR_RATE.set((REQUEST_FAILURE._value.get() / total_req) * 100)

        return {"prediction": float(pred)}

    except Exception as e:
        REQUEST_FAILURE.inc()
        ERROR_RATE.set(100)  # semua error
        return {"error": str(e)}


# METRICS ENDPOINT
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# RUN UVICORN
if __name__ == "__main__":
    uvicorn.run("prometheus_exporter:app", host="0.0.0.0", port=8000, reload=True)
