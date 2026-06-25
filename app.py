# ============================================================
# app.py — Gradio Web Interface for Multi-Attack Threat Detection
# (Dynamically built from the model's actual top features)
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ── Load saved model files ────────────────────────────────
model            = joblib.load('model/final_model.pkl')
scaler           = joblib.load('model/final_scaler.pkl')
le               = joblib.load('model/final_label_encoder.pkl')
top_features     = joblib.load('model/top_features.pkl')      # actual top 20 features used
all_features     = joblib.load('model/all_features.pkl')      # all original feature names
samples_by_class = joblib.load('model/samples_by_class.pkl')  # dict: class_name -> DataFrame of real rows

class_names = list(le.classes_)

print("Model expects these top features:")
for f in top_features:
    print(f"  - {f}")
print(f"\nModel can detect these classes: {class_names}")


# ── Friendly descriptions for technical feature names ──────
# Purely cosmetic. Any feature not listed here just shows no hint --
# nothing breaks, since this is decoupled from the model logic.
FEATURE_INFO = {
    "Bwd Packets/s":              "How many reply packets arrive per second",
    "Packet Length Variance":     "How much packet sizes vary",
    "Bwd Packet Length Max":      "Largest packet received in reply",
    "Fwd Packet Length Max":      "Largest packet sent by the source",
    "Packet Length Std":          "Spread/variation in packet sizes",
    "Avg Bwd Segment Size":       "Average size of data received",
    "Bwd Packet Length Mean":     "Average size of packets received",
    "Fwd Packet Length Mean":     "Average size of packets sent",
    "Packet Length Mean":         "Average size of all packets",
    "Total Length of Bwd Packets":"Total data received, in bytes",
    "Total Length of Fwd Packets":"Total data sent, in bytes",
    "Average Packet Size":        "Average size across the whole flow",
    "Max Packet Length":          "Largest packet seen in either direction",
    "Subflow Fwd Bytes":          "Bytes sent in this sub-connection",
    "ACK Flag Count":             "Number of acknowledgement signals sent",
    "PSH Flag Count":             "Number of 'push data now' signals sent",
    "Init_Win_bytes_forward":     "Initial window size from sender",
    "Destination Port":           "Port number being connected to",
    "Subflow Bwd Bytes":          "Bytes received in this sub-connection",
    "act_data_pkt_fwd":           "Packets that actually carried data",
}

# ── Group features into friendly categories ────────────────
GROUP_DEFINITIONS = [
    ("📤 Outgoing Traffic (what was sent)", [
        "Fwd Packet Length Max", "Fwd Packet Length Mean", "Total Length of Fwd Packets",
        "Subflow Fwd Bytes", "act_data_pkt_fwd",
    ]),
    ("📥 Incoming Traffic (what was received)", [
        "Bwd Packets/s", "Bwd Packet Length Max", "Bwd Packet Length Mean",
        "Avg Bwd Segment Size", "Total Length of Bwd Packets", "Subflow Bwd Bytes",
    ]),
    ("📦 Overall Packet Shape", [
        "Packet Length Variance", "Packet Length Std", "Packet Length Mean",
        "Average Packet Size", "Max Packet Length",
    ]),
    ("🚩 Connection Signals", [
        "ACK Flag Count", "PSH Flag Count", "Init_Win_bytes_forward", "Destination Port",
    ]),
]


def build_groups(features):
    """Split `features` into the predefined groups above, plus a
    catch-all 'Other' group for anything not explicitly listed.
    This means the UI never breaks even if retraining changes the
    top 20 features again in the future."""
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


# ── Severity / color mapping per class ─────────────────────
# BENIGN -> safe (green). Everything else -> danger (red).
# This keeps the color logic correct automatically even if the
# set of attack class names changes after a future retrain.
def classify_severity(label):
    return "safe" if label == "BENIGN" else "danger"


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

    severity = classify_severity(label)

    if severity == "safe":
        result_text = (
            f"✅ BENIGN TRAFFIC\n"
            f"Confidence: {confidence}%\n\n"
            f"This network flow appears normal. No threat detected."
        )
        new_classes = ["result-box", "status-safe"]
    else:
        result_text = (
            f"🚨 ATTACK DETECTED: {label}\n"
            f"Confidence: {confidence}%\n\n"
            f"This network flow matches patterns of a {label} attack. Immediate review recommended."
        )
        new_classes = ["result-box", "status-danger"]

    # Show the full probability breakdown across all classes too,
    # since with 6 classes the runner-up predictions are informative.
    proba_lines = "\n".join(
        f"   {cls}: {round(p * 100, 2)}%"
        for cls, p in sorted(zip(class_names, proba), key=lambda x: -x[1])
    )
    result_text += f"\n\nFull probability breakdown:\n{proba_lines}"

    return gr.update(value=result_text, elem_classes=new_classes)


# ── Load Sample function ──────────────────────────────────
# sample_type is now one of the actual class names (BENIGN, Bot, DDoS, etc.)
def load_sample(sample_type):
    class_samples = samples_by_class.get(sample_type)
    if class_samples is None or len(class_samples) == 0:
        # Should not happen given training always saves all classes,
        # but guards against an empty/missing class gracefully.
        return [0 for _ in top_features]
    row = class_samples.sample(1).iloc[0]
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
    font-size: 14px !important;
    font-weight: 500 !important;
    border-width: 2px !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #334155 !important;
    font-family: Consolas, monospace !important;
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

.feature-hint {
    font-size: 11px !important;
    color: #94a3b8 !important;
    margin: -8px 0 8px 4px !important;
}
"""


# ── Build Gradio Interface ────────────────────────────────
with gr.Blocks(title="Telecom Threat Detector", theme=custom_theme, css=custom_css) as app:

    gr.Markdown(f"""
    # 🛡️ Telecom Network Threat Detector
    ### AI-powered multi-attack intrusion detection using CICIDS 2017 dataset
    *Based on: Using AI in Cyber Security Risk Management for Telecom Industry 4.0*
    ---
    This model can now detect **{len(class_names)} traffic types**: {", ".join(class_names)}.

    Enter the network flow features below and click **Analyse** to detect threats,
    or load a real test sample for any class using the buttons below.
    """)

    gr.Markdown("### 🎲 Load a Real Sample")
    gr.Markdown("*Pick a traffic type below to instantly fill the fields with a real example from the test set.*")
    with gr.Row():
        sample_dropdown = gr.Dropdown(
            choices=class_names,
            value=class_names[0],
            label="Traffic type to load",
            scale=3
        )
        load_sample_btn = gr.Button("📥 Load Sample", scale=1)

    gr.Markdown("### 🧮 Network Flow Features")
    gr.Markdown(
        "*Grouped below by category. Each field also shows a plain-English hint. "
        "Expand a section to edit its values, or just use the Load Sample control above.*"
    )

    # Dynamically create one gr.Number() input per top feature,
    # grouped into friendly, collapsible sections.
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
        lines=10,
        interactive=False,
        elem_classes=["result-box"]
    )

    gr.Markdown("""
    ---
    ### 🧪 Tip
    Pick a traffic type above and click **"Load Sample"** to auto-fill the fields with
    real data from the test set, then click **Analyse Traffic** to see how confidently
    the model identifies it -- including the full probability breakdown across all classes.
    """)

    analyse_btn.click(
        fn=predict_threat,
        inputs=ordered_inputs,
        outputs=result_box
    )

    load_sample_btn.click(
        fn=load_sample,
        inputs=sample_dropdown,
        outputs=ordered_inputs
    )

app.launch(server_name="0.0.0.0", server_port=7860)