from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}, r"/logs": {"origins": "*"}})

# Load model and scaler
try:
    model = joblib.load("fraud_model_new.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or scaler: {e}")

# Log file path
LOG_FILE = "logs.json"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract and prepare features
        features = [
            float(data.get("avg_speed", 0)),
            float(data.get("max_speed", 0)),
            float(data.get("num_events", 0)),
            float(data.get("event_clicks", 0))
        ]

        scaled_features = scaler.transform([features])
        trust_score = model.predict_proba(scaled_features)[0][1]
        prediction = int(model.predict(scaled_features)[0])

        # Log the entry
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": {
                "avg_speed": features[0],
                "max_speed": features[1],
                "num_events": features[2],
                "event_clicks": features[3]
            },
            "prediction": prediction,
            "trust_score": round(trust_score, 4)
        }

        with open(LOG_FILE, "r+") as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=2)

        return jsonify({
            "success": True,
            "trust_score": round(trust_score, 4),
            "prediction": prediction
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        return jsonify({"success": True, "logs": logs}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True)
