import time
import joblib
from prometheus_client import start_http_server, Gauge, Counter
from flask import Flask, request, jsonify
import traceback


#  LOAD MODEL
MODEL_PATH = "C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Monitoring dan Logging\model.pkl"
model = joblib.load(MODEL_PATH)

#  METRICS PROMETHEUS
request_total = Counter("inference_request_total", "Jumlah request inference masuk")
request_latency = Gauge("inference_request_latency", "Latency inference (detik)")
inference_error = Counter("inference_error_total", "Jumlah error inference")
model_accuracy = Gauge("model_accuracy_static", "Akurasi model dari evaluasi")


model_accuracy.set(0.88)

#  API SERVER
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    request_total.inc()

    try:
        data = request.json

        # Ambil fitur dari input JSON
        features = [
            data["gender"],
            data["race_ethnicity"],
            data["parental_education"],
            data["lunch"],
            data["test_preparation"],
            data["math_score"]
        ]

        prediction = model.predict([features])[0]

        latency = time.time() - start_time
        request_latency.set(latency)

        return jsonify({
            "prediction": float(prediction),
            "latency": latency
        })

    except Exception as e:
        inference_error.inc()
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

#  MAIN
if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus metrics exposed at http://localhost:8000/metrics")
    app.run(host="0.0.0.0", port=5000)
