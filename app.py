# ============================================================
# app.py — Gradio Web Interface for Threat Detection
# (Dynamically built from the model's actual top features)
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ── Load saved model files ────────────────────────────────
model         = joblib.load('model/final_model.pkl')
scaler        = joblib.load('model/final_scaler.pkl')
le            = joblib.load('model/final_label_encoder.pkl')
top_features  = joblib.load('model/top_features.pkl')   # actual 20 features used
all_features  = joblib.load('model/all_features.pkl')   # all 78 original features
sample_benign = joblib.load('model/sample_benign.pkl')
sample_ddos   = joblib.load('model/sample_ddos.pkl')

print("Model expects these top features:")
for f in top_features:
    print(f"  - {f}")


# ── Prediction function ───────────────────────────────────
# Takes *args matching the order of top_features, builds a full
# 78-feature vector (zeros for everything not in top_features),
# scales it, then selects just the top_features columns for prediction.
def predict_threat(*values):
    input_dict = dict(zip(top_features, values))

    # Build full feature vector (zeros for unused features)
    full_vector = pd.DataFrame([{f: 0 for f in all_features}])
    for feat, val in input_dict.items():
        full_vector[feat] = val

    # Scale using the SAME scaler fitted on all 78 features
    scaled = scaler.transform(full_vector)
    scaled_df = pd.DataFrame(scaled, columns=all_features)

    # Select only the top features, in the exact order the model expects
    top_input = scaled_df[top_features].values

    # Predict
    pred = model.predict(top_input)[0]
    proba = model.predict_proba(top_input)[0]
    confidence = round(max(proba) * 100, 2)
    label = le.inverse_transform([pred])[0]

    if label == 'BENIGN':
        result = f"BENIGN TRAFFIC\nConfidence: {confidence}%\n\nThis network flow appears normal. No threat detected."
    else:
        result = f"ATTACK DETECTED: {label}\nConfidence: {confidence}%\n\nThis network flow matches patterns of a {label} attack. Immediate review recommended."

    return result


# ── Load Sample function ──────────────────────────────────
# Returns values in the exact order of top_features, taken from
# real rows in the test set (never seen during training).
def load_sample(sample_type):
    if sample_type == "BENIGN":
        row = sample_benign.sample(1).iloc[0]
    else:
        row = sample_ddos.sample(1).iloc[0]

    return [row[feat] for feat in top_features]


# ── Build Gradio Interface ────────────────────────────────
with gr.Blocks(title="Telecom Threat Detector") as app:

    gr.Markdown("""
    # Telecom Network Threat Detector
    ### AI-powered intrusion detection using CICIDS 2017 dataset
    *Based on: Using AI in Cyber Security Risk Management for Telecom Industry 4.0*
    ---
    Enter the network flow features below and click **Analyse** to detect threats.

    The fields below are generated automatically from the model's actual top 20
    most important features — so the UI always matches what the model expects.
    """)

    gr.Markdown("### Load a Real Sample")
    with gr.Row():
        load_benign_btn = gr.Button("Load random BENIGN sample")
        load_ddos_btn   = gr.Button("Load random DDoS sample")

    gr.Markdown("### Network Flow Features")

    # Dynamically create one gr.Number() input per top feature,
    # distributed round-robin across 3 columns for a clean layout.
    box_lookup = {}
    with gr.Row():
        with gr.Column():
            for feat in top_features[0::3]:
                box_lookup[feat] = gr.Number(label=feat, value=0)
        with gr.Column():
            for feat in top_features[1::3]:
                box_lookup[feat] = gr.Number(label=feat, value=0)
        with gr.Column():
            for feat in top_features[2::3]:
                box_lookup[feat] = gr.Number(label=feat, value=0)

    # Reorder boxes to match top_features order, so predict_threat()
    # and load_sample() (which use top_features order) line up correctly.
    ordered_inputs = [box_lookup[feat] for feat in top_features]

    analyse_btn = gr.Button("Analyse Traffic", variant="primary", size="lg")

    gr.Markdown("### Detection Result")
    result_box = gr.Textbox(label="AI Analysis Result", lines=4, interactive=False)

    gr.Markdown("""
    ---
    ### Tip
    Click **"Load random BENIGN sample"** or **"Load random DDoS sample"** above to
    auto-fill the fields with real data from the test set, then click **Analyse Traffic**.
    """)

    analyse_btn.click(
        fn=predict_threat,
        inputs=ordered_inputs,
        outputs=result_box
    )

    load_benign_btn.click(
        fn=lambda: load_sample("BENIGN"),
        outputs=ordered_inputs
    )

    load_ddos_btn.click(
        fn=lambda: load_sample("DDOS"),
        outputs=ordered_inputs
    )

app.launch()
