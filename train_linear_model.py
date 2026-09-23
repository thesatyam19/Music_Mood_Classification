import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = pd.read_csv("features/X_audio.csv")
Y = pd.read_csv("features/Y_emotions.csv")

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MultiOutputRegressor(
    LinearRegression()
)

model.fit(
    X_train_scaled,
    Y_train
)

Y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Linear Regression Results")
print("-------------------------")
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
    "models/linear_emotion_model.pkl"
)

joblib.dump(
    scaler,
    "models/audio_scaler.pkl"
)

print("\nModel saved:")
print("models/linear_emotion_model.pkl")
print("models/audio_scaler.pkl")