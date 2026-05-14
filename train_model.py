import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("heatstrokedataset.csv")

# Features
X = df[['ambient_temp', 'humidity', 'body_temp', 'duration', 'hour', 'heat_index']]
y = df['risk_label']


noise = np.random.normal(0, 1.0, X.shape)
X = X + noise

model = RandomForestClassifier(
    n_estimators=60,
    max_depth=6,
    random_state=42
)

# Train
model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")

print(" Model retrained and saved!")