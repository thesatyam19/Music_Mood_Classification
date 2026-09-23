import pandas as pd
import os

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

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

audio_df = pd.read_csv("features/audio_features.csv")

print("CSV rows:", len(df))
print("Audio rows:", len(audio_df))

print("\nUnique CSV tracks:", df["track id"].nunique())
print("Unique audio tracks:", audio_df["track_id"].nunique())

emotion_summary = (
    df.groupby("track id")[emotion_columns]
    .mean()
    .reset_index()
)

print("\nEmotion summary shape:", emotion_summary.shape)

print("\nFirst 10 tracks:")
print(emotion_summary.head(10).to_string(index=False))

print("\nEmotion average across all tracks:")
print(
    emotion_summary[emotion_columns]
    .mean()
    .sort_values(ascending=False)
)

os.makedirs("features", exist_ok=True)

emotion_summary.to_csv(
    "features/emotion_analysis.csv",
    index=False
)

print("\nSaved: features/emotion_analysis.csv")