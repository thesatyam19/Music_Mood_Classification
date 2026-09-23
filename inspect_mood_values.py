import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

df = df.drop_duplicates()

print("Mood unique values:")
print(sorted(df["mood"].unique()))

print("\nMood value counts:")
print(df["mood"].value_counts().sort_index())

print("\nData types:")
print(df[[
    "mood",
    "liked",
    "disliked",
    "age",
    "gender",
    "mother tongue"
]].dtypes)

print("\nFirst 20 rows:")
print(
    df[
        [
            "track id",
            "mood",
            "liked",
            "disliked",
            "age",
            "gender",
            "mother tongue"
        ]
    ].head(20)
)