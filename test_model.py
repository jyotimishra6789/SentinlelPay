import joblib
import numpy as np

model = joblib.load('fraud_model_new.pkl')

# Try a range of feature combinations
samples = [
    [0.025, 8, 100, 0],     # very trusted
    [0.04, 10, 80, 5],      # moderately trusted
    [0.06, 15, 70, 10],     # suspicious
    [0.08, 20, 60, 20],     # fraudulent
    [0.03, 7, 110, 1],      # trusted
]

for i, features in enumerate(samples):
    prob = model.predict_proba([features])[0][1]
    print(f"Sample {i+1}: Trust Score = {prob:.4f}")