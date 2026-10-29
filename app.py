import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

import gradio as gr

import mapping

col_names = mapping.col_names

feature_map = mapping.feature_map

df = pd.read_csv("data/agaricus-lepiota.data",
                 header=None,
                 names=col_names)

target_col = "class"

X = df.drop(columns=target_col)
Y = df[target_col]

categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=200
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y
)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

joblib.dump(clf, "mushroom_model.joblib")


loaded_model = joblib.load("mushroom_model.joblib")
feature_names = list(feature_map.keys())

def predict_mushroom(*inputs):
    data_dict = {}

    for feature, value in zip(feature_names, inputs):
        data_dict[feature] = [feature_map[feature][value]]

    input_df= pd.DataFrame(data_dict)
    pred = loaded_model.predict(input_df)[0]
    proba = loaded_model.predict_proba(input_df)[0]
    classes = loaded_model.classes_
    proba_str = ", ".join(
        [f"{cls}: {p:.2f}" for cls, p in zip(classes, proba)]
    )
    return f"Prediction: {pred} (probabilites -> {proba_str})"

inputs = []
for feature in feature_names:
    mapping = feature_map[feature]
    inputs.append(gr.Dropdown(choices=list(mapping.keys()), label=feature.replace("_", " ").title(), value=list(mapping.keys())[0]))

output = gr.Textbox(label="Result")

demo = gr.Interface(
    fn=predict_mushroom,
    inputs=inputs,
    outputs=output,
    title="Poisonus mushroom predictor",
    description="Predicts poisonus mushrooms by Martin"
)

if __name__ == "__main__":
    demo.launch()