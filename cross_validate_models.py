import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

X = pd.read_csv("features/X_audio.csv")
Y = pd.read_csv("features/Y_emotions.csv")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_mae = []
rf_mse = []
rf_r2 = []

xgb_mae = []
xgb_mse = []
xgb_r2 = []

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]

    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt"
        )
    )

    rf_model.fit(X_train, Y_train)

    rf_pred = rf_model.predict(X_test)

    rf_mae.append(mean_absolute_error(Y_test, rf_pred))
    rf_mse.append(mean_squared_error(Y_test, rf_pred))
    rf_r2.append(r2_score(Y_test, rf_pred))

    xgb_model = MultiOutputRegressor(
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

    xgb_model.fit(X_train, Y_train)

    xgb_pred = xgb_model.predict(X_test)

    xgb_mae.append(mean_absolute_error(Y_test, xgb_pred))
    xgb_mse.append(mean_squared_error(Y_test, xgb_pred))
    xgb_r2.append(r2_score(Y_test, xgb_pred))

    print("\nFold", fold)

    print("Random Forest:")
    print("MAE:", round(rf_mae[-1], 4))
    print("MSE:", round(rf_mse[-1], 4))
    print("R2 :", round(rf_r2[-1], 4))

    print("XGBoost:")
    print("MAE:", round(xgb_mae[-1], 4))
    print("MSE:", round(xgb_mse[-1], 4))
    print("R2 :", round(xgb_r2[-1], 4))


print("\n" + "=" * 50)
print("5-FOLD CROSS VALIDATION AVERAGE")
print("=" * 50)

print("\nRandom Forest")
print("Average MAE:", round(np.mean(rf_mae), 4))
print("Average MSE:", round(np.mean(rf_mse), 4))
print("Average R2 :", round(np.mean(rf_r2), 4))

print("\nXGBoost")
print("Average MAE:", round(np.mean(xgb_mae), 4))
print("Average MSE:", round(np.mean(xgb_mse), 4))
print("Average R2 :", round(np.mean(xgb_r2), 4))