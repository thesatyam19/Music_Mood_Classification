import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("features/X_test.csv")
Y_test = pd.read_csv("features/Y_test.csv")

models = {
    "Linear Regression": "models/linear_emotion_model.pkl",
    "Random Forest": "models/random_forest_emotion_model.pkl",
    "XGBoost": "models/xgboost_emotion_model.pkl"
}

results = []

for name, path in models.items():

    model = joblib.load(path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })

result_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print("=" * 70)
print(result_df.to_string(index=False))

result_df.to_csv(
    "features/model_comparison.csv",
    index=False
)

print("\nSaved: features/model_comparison.csv")