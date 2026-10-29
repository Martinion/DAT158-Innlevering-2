import pandas as pd
import gradio as gr
import joblib
from sklearn.datasets import load_breast_cancer
import datasets

feature_map = datasets.feature_map

MODELS_DIR = "models/"
MODEL_SUFFIX = ".joblib"

loaded_model = joblib.load("models/cancer_model.joblib")
feature_names = list(loaded_model.feature_names_in_)


def predict(*inputs):
    data_dict = {name: [value] for name, value in zip(feature_names, inputs)}

    input_df= pd.DataFrame(data_dict)
    pred = loaded_model.predict(input_df)[0]
    proba = loaded_model.predict_proba(input_df)[0]
    classes = loaded_model.classes_
    proba_str = ", ".join(
        [f"{cls}: {p:.2f}" for cls, p in zip(classes, proba)]
    )
    return f"Prediction: {bool(pred)} (probabilites -> {proba_str})"

inputs = []
for feature in feature_names:
    inputs.append(gr.Textbox(label=feature))

output = gr.Textbox(label="Prediksjon")

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=output,
    title="DAT158 prosjekt",
    description="Brystkreft pred"
)


if __name__ == "__main__":
    demo.launch()