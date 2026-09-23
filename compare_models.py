import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("features/model_dataset.csv")

X = df.drop(columns=["track id", "mood"])
y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ]),
    
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": f1
    })

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Macro F1:", f1)

results_df = pd.DataFrame(results)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df)

os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

best_model_name = results_df.loc[
    results_df["Macro F1"].idxmax(),
    "Model"
]

print("\nModel with highest Macro F1:", best_model_name)

print("\nSaved: models/model_comparison.csv")