import pandas as pd
import numpy as np

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

    values = df[emotion]

    print("\n" + "=" * 50)
    print(emotion.upper())
    print("=" * 50)

    print("25th percentile :", round(values.quantile(0.25), 3))
    print("50th percentile :", round(values.quantile(0.50), 3))
    print("60th percentile :", round(values.quantile(0.60), 3))
    print("70th percentile :", round(values.quantile(0.70), 3))
    print("75th percentile :", round(values.quantile(0.75), 3))
    print("80th percentile :", round(values.quantile(0.80), 3))
    print("90th percentile :", round(values.quantile(0.90), 3))