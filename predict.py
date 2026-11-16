"""Flask web service that predicts whether a student is at risk of failing.

Run locally:
    gunicorn --bind 0.0.0.0:9696 predict:app
"""

import pickle

from flask import Flask, jsonify, request

MODEL_PATH = "model.bin"

with open(MODEL_PATH, "rb") as f_in:
    saved = pickle.load(f_in)

pipeline = saved["pipeline"]
threshold = saved["threshold"]

app = Flask("student-risk")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    student = request.get_json()
    if not isinstance(student, dict):
        return jsonify({"error": "Send one student as a JSON object"}), 400

    probability = float(pipeline.predict_proba([student])[0, 1])
    result = {
        "fail_probability": round(probability, 3),
        "at_risk": bool(probability >= threshold),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
