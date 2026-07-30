from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Locate the generated dataset
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "raw" / "transactions.csv"

# Load data
df = pd.read_csv(data_path)

# Features used for prediction
X = df[
    [
        "amount",
        "new_device",
        "transactions_last_hour",
        "failed_attempts",
        "amount_deviation",
        "international_transaction",
    ]
]

# Target column
y = df["is_fraud"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

# Save model
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "fraud_model.joblib")

print("\nModel saved successfully.")