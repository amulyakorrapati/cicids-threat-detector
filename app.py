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
# Returns the result text AND a gr.update() that recolors the textbox
# itself via elem_classes -- no custom JS needed.
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
        result_text = f"✅ BENIGN TRAFFIC\nConfidence: {confidence}%\n\nThis network flow appears normal. No threat detected."
        new_classes = ["result-box", "status-safe"]
    else:
        result_text = f"🚨 ATTACK DETECTED: {label}\nConfidence: {confidence}%\n\nThis network flow matches patterns of a {label} attack. Immediate review recommended."
        new_classes = ["result-box", "status-danger"]

    return gr.update(value=result_text, elem_classes=new_classes)


# ── Load Sample function ──────────────────────────────────
def load_sample(sample_type):
    if sample_type == "BENIGN":
        row = sample_benign.sample(1).iloc[0]
    else:
        row = sample_ddos.sample(1).iloc[0]

    return [row[feat] for feat in top_features]


# ── Theme + custom CSS: gradient background, highlighted inputs,
#    and green/red result coloring (pure CSS, no custom JS) ─────
custom_theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="slate",
)

custom_css = """
footer {visibility: hidden}

/* Gradient page background */
.gradio-container {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f0f4f8 100%) !important;
}

/* Highlight all number input boxes */
input[type="number"] {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
input[type="number"]:focus {
    border-color: #ef4444 !important;
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.15) !important;
}

/* Result textbox base styling */
.result-box textarea {
    font-size: 15px !important;
    font-weight: 500 !important;
    border-width: 2px !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #334155 !important;
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}

/* Safe (green) state -- applied via elem_classes from Python */
.status-safe textarea {
    background: #f0fdf4 !important;
    border-color: #22c55e !important;
    color: #15803d !important;
}

/* Danger (red) state -- applied via elem_classes from Python */
.status-danger textarea {
    background: #fef2f2 !important;
    border-color: #ef4444 !important;
    color: #b91c1c !important;
}
"""


# ── Build Gradio Interface ────────────────────────────────
with gr.Blocks(title="Telecom Threat Detector", theme=custom_theme, css=custom_css) as app:

    gr.Markdown("""
    # 🛡️ Telecom Network Threat Detector
    ### AI-powered intrusion detection using CICIDS 2017 dataset
    *Based on: Using AI in Cyber Security Risk Management for Telecom Industry 4.0*
    ---
    Enter the network flow features below and click **Analyse** to detect threats.

    The fields below are generated automatically from the model's actual top 20
    most important features — so the UI always matches what the model expects.
    """)

    gr.Markdown("### 🎲 Load a Real Sample")
    with gr.Row():
        load_benign_btn = gr.Button("📥 Load random BENIGN sample")
        load_ddos_btn   = gr.Button("📥 Load random DDoS sample")

    gr.Markdown("### 🧮 Network Flow Features")

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

    analyse_btn = gr.Button("🔍 Analyse Traffic", variant="primary", size="lg")

    gr.Markdown("### 🎯 Detection Result")
    result_box = gr.Textbox(
        label="AI Analysis Result",
        lines=4,
        interactive=False,
        elem_classes=["result-box"]
    )

    gr.Markdown("""
    ---
    ### 🧪 Tip
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

app.launch(server_name="0.0.0.0", server_port=7860)