"""
Flask API + web page for the Bank Customer Churn Prediction model.

Run locally:
    pip install -r requirements.txt
    python train_model.py        # generates model/bank_churn_model.pkl (once)
    python app.py
    open http://127.0.0.1:5000
"""

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model/bank_churn_model.pkl"
_package = joblib.load(MODEL_PATH)
preprocessor = _package["preprocessor"]
model = _package["model"]


def age_group(age):
    if age < 30:
        return "Young"
    elif age < 45:
        return "Adult"
    elif age < 60:
        return "Middle_Aged"
    return "Senior"


def build_features(payload: dict) -> pd.DataFrame:
    row = {
        "CreditScore": float(payload["credit_score"]),
        "Geography": payload["geography"],
        "Gender": payload["gender"],
        "Age": float(payload["age"]),
        "Tenure": float(payload["tenure"]),
        "Balance": float(payload["balance"]),
        "NumOfProducts": float(payload["num_of_products"]),
        "HasCrCard": int(payload["has_cr_card"]),
        "IsActiveMember": int(payload["is_active_member"]),
        "EstimatedSalary": float(payload["estimated_salary"]),
    }
    df = pd.DataFrame([row])
    df["AgeGroup"] = df["Age"].apply(age_group)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["BalanceSalaryRatio"] = np.log1p(df["Balance"] / (df["EstimatedSalary"] + 1))
    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        features = build_features(payload)
        processed = preprocessor.transform(features)
        prediction = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[0][1])
        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": "Likely to churn" if prediction == 1 else "Likely to stay",
        })
    except Exception as exc:  # keep the demo resilient to bad input
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
