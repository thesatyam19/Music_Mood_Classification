import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

X = pd.read_csv("features/X_audio.csv")
Y = pd.read_csv("features/Y_emotions.csv")

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
)

model.fit(
    X_train,
    Y_train
)

Y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("XGBoost Results")
print("---------------")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("R2 Score:", round(r2, 4))

print("\nEmotion-wise R2 Scores:")

for i, emotion in enumerate(Y.columns):

    score = r2_score(
        Y_test.iloc[:, i],
        Y_pred[:, i]
    )

    print(
        f"{emotion:20s}: {score:.4f}"
    )

joblib.dump(
    model,
    "models/xgboost_emotion_model.pkl"
)

print("\nModel saved:")
print("models/xgboost_emotion_model.pkl")