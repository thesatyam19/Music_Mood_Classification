import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

df = df.drop_duplicates()

mood_table = (
    df.groupby("track id")["mood"]
    .value_counts()
    .unstack(fill_value=0)
)

print("Total unique tracks:", len(mood_table))

print("\nMood distribution across annotations:")
print(df["mood"].value_counts().sort_index())

print("\nMood distribution for tracks having audio:")
audio = pd.read_csv("features/audio_features.csv")
audio = audio.rename(columns={"track_id": "track id"})

track_ids = audio["track id"].unique()

available = df[df["track id"].isin(track_ids)]

print(available["mood"].value_counts().sort_index())

print("\nUnique tracks by mood:")
for mood in sorted(available["mood"].unique()):
    print(
        "Mood",
        mood,
        ":",
        available.loc[
            available["mood"] == mood,
            "track id"
        ].nunique()
    )