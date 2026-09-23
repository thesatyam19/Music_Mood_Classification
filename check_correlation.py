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

correlation = df[emotion_columns + ["mood"]].corr()

print("Correlation of emotions with mood:\n")

print(
    correlation["mood"]
    .drop("mood")
    .sort_values(ascending=False)
    .round(3)
)