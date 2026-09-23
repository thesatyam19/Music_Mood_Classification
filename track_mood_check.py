import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

audio = pd.read_csv("features/audio_features.csv")
audio = audio.rename(columns={"track_id": "track id"})

track_ids = audio["track id"].unique()

df = df[df["track id"].isin(track_ids)]

mood_distribution = (
    df.groupby(["track id", "mood"])
    .size()
    .unstack(fill_value=0)
)

print("Track-wise Mood Distribution:\n")
print(mood_distribution)

print("\nFirst 10 tracks:\n")
print(mood_distribution.head(10))

print("\nNumber of tracks with multiple mood labels:")

multiple = (mood_distribution > 0).sum(axis=1)

print((multiple > 1).sum())