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

thresholds = [0.30, 0.40, 0.50]

for threshold in thresholds:

    print("\n" + "=" * 50)
    print("THRESHOLD:", threshold)
    print("=" * 50)

    for emotion in emotion_columns:

        positive = (df[emotion] >= threshold).sum()

        print(
            f"{emotion:20s} : "
            f"{positive:3d} positive / "
            f"{len(df) - positive:3d} negative"
        )