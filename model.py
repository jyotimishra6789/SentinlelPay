import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv('augmented_mouse_data.csv')
data.sort_values(['session_id', 'timestamp'], inplace=True)

# --- Feature extraction ---
def extract_features(df):
    features = []
    for sid, group in df.groupby('session_id'):
        dx = group['screen_x'].diff().fillna(0)
        dy = group['screen_y'].diff().fillna(0)
        dt = group['timestamp'].diff().fillna(1)
        speed = np.sqrt(dx**2 + dy**2) / dt

        feature = {
            'avg_speed': speed.mean(),
            'max_speed': speed.max(),
            'num_events': len(group),
            'event_clicks': sum(group['event_type'] == 1),
            'is_fraud': group['is_fraud'].iloc[0]
        }
        features.append(feature)
    return pd.DataFrame(features)

features = extract_features(data)
X = features.drop('is_fraud', axis=1)
y = features['is_fraud']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Base model
base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)

# Calibrated model (sigmoid improves probability smoothness)
model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'fraud_model_new.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Calibrated model and scaler saved.")
