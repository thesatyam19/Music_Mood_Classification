import pandas as pd
import os

df = pd.read_csv("features/final_dataset.csv")

emotion_columns = [
    "amazement",
    "solemnity",
    "tenderness",
    "nostalgia",
    "calmness",
    "power",
    "joyful_activation",
    "tension",
    "sadness"
]

feature_columns = [
    col for col in df.columns
    if col not in ["track_id", "genre"] + emotion_columns
]

X = df[feature_columns]
Y = df[emotion_columns]

print("Feature shape:", X.shape)
print("Target shape:", Y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTargets:")
print(Y.columns.tolist())

print("\nMissing values in X:", X.isnull().sum().sum())
print("Missing values in Y:", Y.isnull().sum().sum())

os.makedirs("features", exist_ok=True)

X.to_csv("features/X_audio.csv", index=False)
Y.to_csv("features/Y_emotions.csv", index=False)

print("\nSaved:")
print("features/X_audio.csv")
print("features/Y_emotions.csv")