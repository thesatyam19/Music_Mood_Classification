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

for column in emotion_columns:
    print("\n", column)
    print(df[column].value_counts().sort_index())

print("\nUnique values in all emotion columns:")

for column in emotion_columns:
    print(column, sorted(df[column].unique()))