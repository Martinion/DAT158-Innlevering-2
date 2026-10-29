import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report

import datasets



linear_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC())
])

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5))
])

svc_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False))
])

rfc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier())
])

nb = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GaussianNB())
])

ensemble = VotingClassifier(
    estimators=[
        ("linear_svc", linear_svc),
        ("knn", knn),
        ("svc_rbf", svc_rbf),
        ("rfc", rfc),
        ("nb", nb)

    ],
    voting="hard"
)

models = {
    "linear_svc": linear_svc,
    "knn": knn,
    "svc_rbf": svc_rbf,
    "rfc": rfc,
    "nb": nb,
    "ensemble": ensemble,
}

parser = argparse.ArgumentParser(description="Tren modell på eit valgt dataset")
parser.add_argument(
    "--dataset",
    type=str,
    default="cancer",
    choices=list(datasets.DATASETS.keys()),
    help="Navn på dataset"
)
args = parser.parse_args()

d_set = args.dataset

X, y = datasets.DATASETS[d_set]()

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

transformers=[]
if numeric_features:
    transformers.append(
        ("num", StandardScaler(), numeric_features)
    )

if categorical_features:
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

model = RandomForestClassifier(
    n_estimators=200
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y
)

results = {name: {"acc": [], 
                  "f1": [],
                  "rec": [],
                  "prec": [],
                  "model": []} 
                  for name in models.keys()}

for n in range(5, 11):
    rkf = RepeatedKFold(n_splits=n, n_repeats=5)

    for fold_index, (train_index, val_index) in enumerate(rkf.split(X_train, y_train), start=1):
        print(f"\n=== Fold {fold_index} ===")
        X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        for name, model in models.items():
            from sklearn.base import clone
            m = clone(model)

            m.fit(X_t, y_t)
            y_pred = m.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="binary")

            results[name]["acc"].append(acc)
            results[name]["rec"].append(rec)
            results[name]["prec"].append(prec)
            results[name]["f1"].append(f1)
            results[name]["model"].append(m)

            print(f"{name:12} acc={acc:.4f} rec={rec:.4f} f1={f1:.4f}")


best_name = {
    "name": None,
    "mean_rec": -np.inf,
}

print("Navn        Accuracy   Recall     Precision  f1")

for name in models:
    mean_rec = np.mean(results[name]["rec"])
    mean_acc = np.mean(results[name]["acc"])
    mean_prec = np.mean(results[name]["prec"])
    mean_f1 = np.mean(results[name]["f1"])

    print(f"{name:12}{mean_acc:.4f}     {mean_rec:.4f}     {mean_prec:.4f}     {mean_f1:.4f}")

    if(best_name["mean_rec"] < mean_rec):
        best_name["name"] = name
        best_name["mean_rec"] = mean_rec

print(f"Vinnende estimator: {best_name["name"]}")

best_model = {
    "model": None,
    "rec": -np.inf,
    "y_pred": None
}
for m in results[best_name["name"]]["model"]:
    y_pred = m.predict(X_test)
    rec = recall_score(y_test, y_pred)

    if best_model["rec"] < rec:
        best_model["rec"] = rec
        best_model["model"] = m
        best_model["y_pred"] = y_pred


print(classification_report(y_test, best_model["y_pred"]))

joblib.dump(best_model["model"], "models/" + d_set + "_model.joblib")


