import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("features/model_dataset.csv")

print("Dataset shape:", df.shape)
print("Unique tracks:", df["track id"].nunique())

X = df.drop(columns=["track id", "mood"])
y = df["mood"]

print("\nFeatures:", X.shape[1])
print("Classes:", sorted(y.unique()))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/mood_model.pkl"
)

print("\nModel saved: models/mood_model.pkl")