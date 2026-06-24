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


# ── Friendly descriptions for technical feature names ──────
# Purely cosmetic -- shown as a short subtitle under each input.
# If a feature isn't in this dictionary (e.g. after retraining
# changes the top 20), it just won't show a subtitle -- nothing breaks.
FEATURE_INFO = {
    "Fwd Packet Length Max":        "Largest packet sent by the source",
    "Init_Win_bytes_forward":       "Initial window size from sender",
    "Destination Port":             "Port number being connected to",
    "Avg Fwd Segment Size":         "Average size of data sent",
    "Subflow Fwd Packets":          "Packets sent in this sub-connection",
    "Fwd IAT Std":                  "Variation in time between sent packets",
    "act_data_pkt_fwd":             "Packets that actually carried data",
    "Subflow Fwd Bytes":            "Bytes sent in this sub-connection",
    "Total Length of Fwd Packets":  "Total data sent, in bytes",
    "Fwd Packet Length Mean":       "Average size of packets sent",
    "Fwd IAT Mean":                 "Average time between sent packets",
    "Fwd IAT Total":                "Total time spent sending packets",
    "Fwd Packet Length Std":        "Variation in size of packets sent",
    "Bwd Packet Length Max":        "Largest packet received in reply",
    "Bwd Packet Length Min":        "Smallest packet received in reply",
    "Avg Bwd Segment Size":         "Average size of data received",
    "Fwd IAT Max":                  "Longest pause between sent packets",
    "Fwd Header Length":            "Total size of packet headers sent",
    "Total Backward Packets":       "Number of packets received in reply",
    "Bwd Packet Length Mean":       "Average size of packets received",
}

# ── Group features into friendly categories for the accordion UI ──
# Falls back gracefully: any top_feature not explicitly grouped below
# is placed into "Other Flow Details" automatically, so this never
# breaks even if retraining changes which 20 features are selected.
GROUP_DEFINITIONS = [
    ("📤 Outgoing Traffic (what you sent)", [
        "Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Total Length of Fwd Packets", "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "act_data_pkt_fwd", "Avg Fwd Segment Size", "Fwd Header Length",
    ]),
    ("📥 Incoming Traffic (what you received)", [
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Avg Bwd Segment Size", "Total Backward Packets",
    ]),
    ("⏱️ Timing Patterns", [
        "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Total", "Fwd IAT Max",
    ]),
    ("🔌 Connection Details", [
        "Destination Port", "Init_Win_bytes_forward",
    ]),
]


def build_groups(features):
    """Split `features` into the predefined groups above, plus a
    catch-all 'Other' group for anything not explicitly listed."""
    remaining = list(features)
    groups = []
    for title, names in GROUP_DEFINITIONS:
        in_group = [f for f in names if f in remaining]
        for f in in_group:
            remaining.remove(f)
        if in_group:
            groups.append((title, in_group))
    if remaining:
        groups.append(("📊 Other Flow Details", remaining))
    return groups


# ── Prediction function ───────────────────────────────────
def predict_threat(*values):
    input_dict = dict(zip(top_features, values))

    full_vector = pd.DataFrame([{f: 0 for f in all_features}])
    for feat, val in input_dict.items():
        full_vector[feat] = val

    scaled = scaler.transform(full_vector)
    scaled_df = pd.DataFrame(scaled, columns=all_features)
    top_input = scaled_df[top_features].values

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


# ── Theme + custom CSS ─────────────────────────────────────
custom_theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="slate",
)

custom_css = """
footer {visibility: hidden}

.gradio-container {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f0f4f8 100%) !important;
}

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

.status-safe textarea {
    background: #f0fdf4 !important;
    border-color: #22c55e !important;
    color: #15803d !important;
}

.status-danger textarea {
    background: #fef2f2 !important;
    border-color: #ef4444 !important;
    color: #b91c1c !important;
}

/* Small grey helper text under each input field */
.feature-hint {
    font-size: 11px !important;
    color: #94a3b8 !important;
    margin: -8px 0 8px 4px !important;
}
"""


# ── Build Gradio Interface ────────────────────────────────
with gr.Blocks(title="Telecom Threat Detector", theme=custom_theme, css=custom_css) as app:

    gr.Markdown("""
    # 🛡️ Telecom Network Threat Detector
    ### AI-powered intrusion detection using CICIDS 2017 dataset
    *Based on: Using AI in Cyber Security Risk Management for Telecom Industry 4.0*
    ---
    Enter the network flow features below and click **Analyse** to detect threats,
    or load a real test sample using the buttons below.
    """)

    gr.Markdown("### 🎲 Load a Real Sample")
    with gr.Row():
        load_benign_btn = gr.Button("📥 Load random BENIGN sample")
        load_ddos_btn   = gr.Button("📥 Load random DDoS sample")

    gr.Markdown("### 🧮 Network Flow Features")
    gr.Markdown(
        "*Grouped below by category. Each field also shows a plain-English hint. "
        "Expand a section to edit its values, or just use the Load Sample buttons above.*"
    )

    # Build grouped, collapsible sections instead of one flat 3-column grid.
    box_lookup = {}
    grouped = build_groups(top_features)

    for i, (group_title, group_features) in enumerate(grouped):
        with gr.Accordion(group_title, open=(i == 0)):
            for feat in group_features:
                box_lookup[feat] = gr.Number(label=feat, value=0)
                hint = FEATURE_INFO.get(feat)
                if hint:
                    gr.Markdown(f"*{hint}*", elem_classes=["feature-hint"])

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