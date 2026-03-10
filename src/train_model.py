import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
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

# Fill missing values
df = df.fillna(df.median())

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save pipeline
joblib.dump(pipeline, "models/heart_model.pkl")

print("Pipeline model saved successfully")