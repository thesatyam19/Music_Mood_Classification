import pandas as pd

df = pd.read_csv("features/final_dataset.csv")

print("Total rows:", len(df))
print("Unique tracks:", df["track id"].nunique())

label_counts = df.groupby("track id")["mood"].nunique()

print("\nMood labels per track:")
print(label_counts.value_counts().sort_index())

print("\nTracks having multiple mood labels:")
print(label_counts[label_counts > 1])