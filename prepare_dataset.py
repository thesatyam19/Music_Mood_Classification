import pandas as pd
import os

audio_df = pd.read_csv("features/audio_features.csv")

label_df = pd.read_csv("data.csv")
label_df.columns = label_df.columns.str.strip()

label_df = label_df.drop_duplicates()

label_df = label_df[
    ["track id", "mood"]
].drop_duplicates()

audio_df = audio_df.rename(
    columns={"track_id": "track id"}
)

print("Audio feature rows:", len(audio_df))
print("Unique audio tracks:", audio_df["track id"].nunique())

print("CSV label rows:", len(label_df))
print("Unique CSV tracks:", label_df["track id"].nunique())

final_df = audio_df.merge(
    label_df,
    on="track id",
    how="inner"
)

print("\nFinal dataset shape:", final_df.shape)
print("Unique final tracks:", final_df["track id"].nunique())

print("\nMood distribution:")
print(final_df["mood"].value_counts().sort_index())

os.makedirs("features", exist_ok=True)

final_df.to_csv(
    "features/final_dataset.csv",
    index=False
)

print("\nSaved: features/final_dataset.csv")