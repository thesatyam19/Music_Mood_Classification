import pandas as pd
import os

audio_df = pd.read_csv("features/audio_features.csv")
emotion_df = pd.read_csv("features/emotion_analysis.csv")

print("Audio dataset:", audio_df.shape)
print("Emotion dataset:", emotion_df.shape)

final_df = pd.merge(
    audio_df,
    emotion_df,
    left_on="track_id",
    right_on="track id",
    how="inner"
)

final_df = final_df.drop(columns=["track id"])

print("\nFinal dataset shape:", final_df.shape)
print("Unique tracks:", final_df["track_id"].nunique())

print("\nMissing values:")
print(final_df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(final_df.head().to_string())

os.makedirs("features", exist_ok=True)

final_df.to_csv(
    "features/final_dataset.csv",
    index=False
)

print("\nSaved: features/final_dataset.csv")