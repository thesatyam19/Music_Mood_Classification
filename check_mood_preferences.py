import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

print("Average liked/disliked value for each mood:\n")

result = df.groupby("mood")[["liked", "disliked"]].mean()

print(result.round(3))

print("\nMood distribution by liked:\n")

print(
    pd.crosstab(
        df["liked"],
        df["mood"],
        normalize="index"
    ).round(3)
)

print("\nMood distribution by disliked:\n")

print(
    pd.crosstab(
        df["disliked"],
        df["mood"],
        normalize="index"
    ).round(3)
)