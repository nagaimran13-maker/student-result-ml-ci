
import json
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

MODEL_FILE = "student_result_model.pkl"
DATASET_FILE = "student_results.csv"
METRICS_FILE = "metrics.json"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. Run the training script first."
    )

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"{DATASET_FILE} not found. Run the dataset creation script first."
    )

df = pd.read_csv(DATASET_FILE)

features = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

if not all(column in df.columns for column in features + ["result"]):
    raise ValueError(
        "Dataset must contain attendance, internal_marks, "
        "assignment_marks, previous_score, and result columns."
    )

X = df[features]
y = df["result"]

model = joblib.load(MODEL_FILE)
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

metrics = {
    "accuracy": round(float(accuracy), 4),
    "total_samples": int(len(df)),
}

with open(METRICS_FILE, "w") as file:
    json.dump(metrics, file, indent=4)

print("Model evaluation completed successfully.")
print(json.dumps(metrics, indent=4))
