import pandas as pd
import numpy as np

# Load your augmented dataset
data = pd.read_csv("augmented_mouse_data.csv")

# Count existing fraud/trusted sessions
fraud_count = data[data['is_fraud'] == 1]['session_id'].nunique()
trusted_sessions = data[data['is_fraud'] == 0]['session_id'].unique()
trusted_count = len(trusted_sessions)

# Define how many fraud sessions you need to balance
needed = trusted_count - fraud_count

def simulate_fraud_on_session(df):
    df = df.copy()
    
    # Add jitter instead of huge jumps
    df['screen_x'] += np.random.normal(0, 50, size=len(df))
    df['screen_y'] += np.random.normal(0, 50, size=len(df))
    
    # Irregular and fast click patterns
    df['timestamp'] = df['timestamp'].diff().fillna(0.1) * np.random.uniform(0.01, 0.5)
    df['event_type'] = np.random.choice([0, 1, 2], size=len(df))
    
    df['is_fraud'] = 1
    return df

# Simulate additional fraud sessions
extra_fraud_data = pd.concat([
    simulate_fraud_on_session(data[data['session_id'] == sid])
    for sid in np.random.choice(trusted_sessions, size=needed, replace=True)
])

# Rename session IDs
extra_fraud_data['original_session_id'] = extra_fraud_data['session_id']
extra_fraud_data['session_id'] = 'fraud_' + extra_fraud_data['session_id'].astype(str)

# Final balanced dataset
balanced_df = pd.concat([data[data['is_fraud'] == 0], data[data['is_fraud'] == 1], extra_fraud_data])
balanced_df.to_csv("balanced_augmented_mouse_data.csv", index=False)

print("Balanced dataset saved as 'balanced_augmented_mouse_data.csv'")
print(balanced_df['is_fraud'].value_counts())


