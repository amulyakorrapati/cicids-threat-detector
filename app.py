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
        result = f"""
<div class="result-card result-positive">
    <div class="result-icon">✅</div>
    <div class="result-text">
        <h3>BENIGN TRAFFIC</h3>
        <p class="confidence-text">Confidence: <strong>{confidence}%</strong></p>
        <p class="detail-text">This network flow appears normal. No threat detected.</p>
    </div>
</div>
<div class="confidence-bar-track">
    <div class="confidence-bar-fill confidence-fill-positive" style="width:{confidence}%;"></div>
</div>
"""
    else:
        result = f"""
<div class="result-card result-negative">
    <div class="result-icon">🚨</div>
    <div class="result-text">
        <h3>ATTACK DETECTED: {label}</h3>
        <p class="confidence-text">Confidence: <strong>{confidence}%</strong></p>
        <p class="detail-text">This network flow matches patterns of a {label} attack. Immediate review recommended.</p>
    </div>
</div>
<div class="confidence-bar-track">
    <div class="confidence-bar-fill confidence-fill-negative" style="width:{confidence}%;"></div>
</div>
"""

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


# ── Custom Theme ───────────────────────────────────────────
custom_theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
)

custom_css = """
#header-banner {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 16px;
}
#header-banner h1 {
    color: white !important;
    margin: 0 !important;
}
#header-banner p {
    color: #cbd5e1 !important;
    margin: 4px 0 0 0 !important;
}
.result-card {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 12px;
}
.result-card.result-positive {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
}
.result-card.result-negative {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
}
.result-icon {
    font-size: 32px;
    line-height: 1;
}
.result-text h3 {
    margin: 0 0 6px 0;
    font-size: 18px;
    color: #1e293b;
}
.confidence-text {
    margin: 0 0 6px 0;
    font-size: 14px;
    color: #334155;
}
.detail-text {
    margin: 0;
    font-size: 13px;
    color: #64748b;
}
.confidence-bar-track {
    height: 8px;
    background: #e2e8f0;
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 8px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}
.confidence-fill-positive {
    background: #22c55e;
}
.confidence-fill-negative {
    background: #ef4444;
}
footer {visibility: hidden}
"""


# ── Build Gradio Interface ────────────────────────────────
with gr.Blocks(title="Telecom Threat Detector", theme=custom_theme, css=custom_css) as app:

    gr.HTML("""
    <div id="header-banner">
        <h1>🛡️ Telecom Network Threat Detector</h1>
        <p>AI-powered intrusion detection using CICIDS 2017 dataset · Random Forest classifier</p>
    </div>
    """)

    gr.Markdown("""
    Enter the network flow features below and click **Analyse Traffic** to detect threats,
    or load a real sample from the test set with the buttons below.
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
    result_box = gr.HTML(value="<p style='color:#94a3b8;'>Results will appear here after analysis.</p>")

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