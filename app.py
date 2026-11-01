
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# -------------------------------
# 1. Logging Configuration
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

# -------------------------------
# 2. Prometheus Metrics
# -------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded"
)

MODEL_LOADED.set(0)

# -------------------------------
# 3. Model Configuration
# -------------------------------

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# -------------------------------
# 4. Request Logging and Metrics
# -------------------------------

@app.before_request
def start_request_logging():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def finish_request_logging(response):

    duration_seconds = time.perf_counter() - g.start_time
    duration_ms = duration_seconds * 1000

    # Exclude the /metrics endpoint from API traffic metrics
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = g.request_id
    return response

# -------------------------------
# 5. Model Loading
# -------------------------------

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "student_result_model.pkl was not found. "
                "Run the training pipeline first."
            )

        model = joblib.load(MODEL_PATH)

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH,
        )

        return model

    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed")
        raise

# -------------------------------
# 6. Health Check API
# -------------------------------

@app.route("/health", methods=["GET"])
def health():
    logger.info("event=health_check")
    return jsonify({"status": "healthy"}), 200


@app.get("/")
def health_check():
    logger.info("event=home_check")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    })

# -------------------------------
# 7. Prediction API
# -------------------------------

@app.post("/predict")
def predict():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            logger.warning(
                "event=invalid_request reason=missing_or_invalid_json"
            )

            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing_fields = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_fields:
            logger.warning(
                "event=validation_failed missing_fields=%s",
                missing_fields,
            )

            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
            }), 400

        logger.info(
            "event=prediction_started feature_names=%s",
            FEATURES,
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        try:
            model = load_model()
            prediction_code = int(model.predict(sample)[0])

        except Exception:
            PREDICTIONS.labels(result="error").inc()
            raise

        PREDICTIONS.labels(result="success").inc()

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        logger.info(
            "event=prediction_success prediction=%s",
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except Exception:
        logger.exception("event=prediction_failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500

# -------------------------------
# 8. Prometheus Metrics API
# -------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

# -------------------------------
# 9. Application Startup
# -------------------------------

if __name__ == "__main__":
    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000,
    )
