import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

print("Total rows:", len(df))
print("Unique track IDs:", df["track id"].nunique())

print("\nFirst 30 track IDs with genre:")
print(
    df[["track id", "genre"]]
    .drop_duplicates()
    .sort_values("track id")
    .head(30)
    .to_string(index=False)
)

print("\nTrack IDs per genre:")

for genre in sorted(df["genre"].unique()):
    temp = df[df["genre"] == genre]["track id"].unique()
    print(genre, ":", len(temp))
    print("First 20:", sorted(temp)[:20])