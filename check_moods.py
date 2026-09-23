import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

df = df.drop_duplicates()

audio = pd.read_csv("features/audio_features.csv")
audio = audio.rename(columns={"track_id": "track id"})

track_ids = audio["track id"].unique()

available = df[df["track id"].isin(track_ids)]

print("Total unique audio tracks:", len(track_ids))

print("\nMood distribution in original dataset:")
print(df["mood"].value_counts().sort_index())

print("\nMood distribution for audio tracks:")
print(available["mood"].value_counts().sort_index())

print("\nUnique audio tracks having each mood:")

for mood in sorted(available["mood"].unique()):
    count = available.loc[
        available["mood"] == mood,
        "track id"
    ].nunique()

    print("Mood", mood, ":", count)