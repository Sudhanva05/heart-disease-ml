import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/heart_raw.csv", names=[
"age","sex","cp","trestbps","chol","fbs","restecg",
"thalach","exang","oldpeak","slope","ca","thal","target"
])

# Clean dataset
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)

# Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values
df = df.fillna(df.median())

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = KNeighborsClassifier()

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/heart_model.pkl")

print("Model saved to models/heart_model.pkl")