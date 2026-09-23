import pandas as pd

df = pd.read_csv("features/final_dataset.csv")

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

for emotion in emotion_columns:

    print("\n", emotion)
    print("-" * 30)

    print("Minimum:", round(df[emotion].min(), 3))
    print("Maximum:", round(df[emotion].max(), 3))
    print("Mean:", round(df[emotion].mean(), 3))
    print("Median:", round(df[emotion].median(), 3))

    print("\nDistribution:")
    print(df[emotion].describe())