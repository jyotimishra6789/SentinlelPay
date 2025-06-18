import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Create synthetic dataset
np.random.seed(42)
data = {
    "keystroke_delay": np.random.normal(70, 10, 500),       # ms
    "mouse_speed": np.random.normal(200, 50, 500),          # px/s
    "swipe_velocity": np.random.normal(60, 15, 500),        # px/s
    "erratic_moves": np.random.poisson(3, 500),             # count
    "label": np.random.choice([0, 1], 500, p=[0.3, 0.7])    # 0 = fraud, 1 = trusted
}

df = pd.DataFrame(data)

# Features and labels
X = df[["keystroke_delay", "mouse_speed", "swipe_velocity", "erratic_moves"]]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained and saved as fraud_model.pkl")
