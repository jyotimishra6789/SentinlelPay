from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}, r"/logs": {"origins": "*"}})

# Load trained model
try:
    model = joblib.load("fraud_model.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Log file path
LOG_FILE = "logs.json"

# Initialize log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data.get("keystroke_delay", 0)),
            float(data.get("mouse_speed", 0)),
            float(data.get("swipe_velocity", 0)),
            float(data.get("erratic_moves", 0))
        ]

        trust_score = model.predict_proba([features])[0][1]
        prediction = int(model.predict([features])[0])

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": {
                "keystroke_delay": features[0],
                "mouse_speed": features[1],
                "swipe_velocity": features[2],
                "erratic_moves": features[3]
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
            "prediction": prediction  # 1 = trusted, 0 = fraud
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
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(debug=True)
