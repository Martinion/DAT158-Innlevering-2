import pandas as pd
import gradio as gr
import joblib
import mapping

feature_map = mapping.feature_map

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