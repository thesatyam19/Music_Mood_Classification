import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

audio = pd.read_csv("features/audio_features.csv")

track_ids = audio["track_id"].unique()

df = df[df["track id"].isin(track_ids)]

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

track_profile = (
    df.groupby("track id")[emotion_columns]
    .mean()
    .reset_index()
)

print("Track-wise average emotional profile:\n")
print(track_profile.head(10).round(3))

print("\nShape:")
print(track_profile.shape)