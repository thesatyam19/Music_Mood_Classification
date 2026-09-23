import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

X_train_scaled.to_csv(
    "features/X_train.csv",
    index=False
)

X_test_scaled.to_csv(
    "features/X_test.csv",
    index=False
)

Y_train.to_csv(
    "features/Y_train.csv",
    index=False
)

Y_test.to_csv(
    "features/Y_test.csv",
    index=False
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training features:", X_train.shape[1])
print("Target emotions:", Y_train.shape[1])

print("\nSaved:")
print("X_train.csv")
print("X_test.csv")
print("Y_train.csv")
print("Y_test.csv")