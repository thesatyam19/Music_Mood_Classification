import pandas as pd
import joblib
import numpy as np

X = pd.read_csv("features/X_audio.csv")
Y = pd.read_csv("features/Y_emotions.csv")

model = joblib.load(
    "models/random_forest_emotion_model.pkl"
)

feature_names = X.columns

for i, emotion in enumerate(Y.columns):

    importance = model.estimators_[i].feature_importances_

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    result = result.sort_values(
        "importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print(emotion.upper())
    print("=" * 60)

    print(
        result.head(10).to_string(index=False)
    )

print("\nFeature importance analysis completed.")