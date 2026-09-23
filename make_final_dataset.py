import pandas as pd
import os

df = pd.read_csv("features/final_dataset.csv")

print("Original rows:", len(df))
print("Unique tracks:", df["track id"].nunique())

final_df = (
    df.groupby("track id")
    .agg({
        "mood": lambda x: x.mode()[0]
    })
    .reset_index()
)

feature_columns = [
    col for col in df.columns
    if col not in ["track id", "mood", "genre"]
]
features = (
    df.groupby("track id")[feature_columns]
    .mean()
    .reset_index()
)

final_df = final_df.merge(
    features,
    on="track id"
)

print("\nFinal dataset shape:", final_df.shape)
print("Unique tracks:", final_df["track id"].nunique())

print("\nMood distribution:")
print(final_df["mood"].value_counts().sort_index())

os.makedirs("features", exist_ok=True)

final_df.to_csv(
    "features/model_dataset.csv",
    index=False
)

print("\nSaved: features/model_dataset.csv")