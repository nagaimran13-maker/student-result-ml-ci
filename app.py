
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

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
# 2. Model Configuration
# -------------------------------

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# -------------------------------
# 3. Request Logging
# -------------------------------

@app.before_request
def start_request_logging():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def finish_request_logging(response):
    duration_ms = (
        time.perf_counter() - g.start_time
    ) * 1000

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
# 4. Model Loading
# -------------------------------

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "student_result_model.pkl was not found. "
                "Run the training pipeline first."
            )

        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH,
        )

        return model

    except Exception:
        logger.exception("event=model_load_failed")
        raise

# -------------------------------
# 5. Health Check API
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
# 6. Prediction API
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

        model = load_model()

        prediction_code = int(model.predict(sample)[0])

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
# 7. Application Startup
# -------------------------------

if __name__ == "__main__":
    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000,
    )
