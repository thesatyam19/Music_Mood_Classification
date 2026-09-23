import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

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

print("Average emotion values for each mood:\n")

result = df.groupby("mood")[emotion_columns].mean()

print(result.round(3))

print("\nNumber of samples for each mood:\n")
print(df["mood"].value_counts().sort_index())