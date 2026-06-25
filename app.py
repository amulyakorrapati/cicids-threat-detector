# ============================================================
# app.py — Gradio Web Interface for Multi-Attack Threat Detection
# (Dynamically built from the model's actual top features)
# Now includes: manual entry, real sample loading, and batch CSV upload
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


# ── Attack knowledge base ──────────────────────────────────
ATTACK_INFO = {
    "BENIGN": {
        "severity": "none",
        "what_it_is": "This is normal, everyday network traffic with no signs of malicious activity.",
        "why_it_matters": "No action needed -- this is exactly what healthy traffic looks like.",
        "recommended_action": "No action required. Continue normal monitoring.",
    },
    "DDoS": {
        "severity": "critical",
        "what_it_is": (
            "A Distributed Denial-of-Service (DDoS) attack. The goal is to overwhelm a server or "
            "network with a flood of traffic, so it becomes too slow or crashes entirely, "
            "denying access to legitimate users."
        ),
        "why_it_matters": (
            "DDoS attacks can take critical services completely offline, causing direct financial "
            "loss, reputational damage, and -- in telecom networks specifically -- can disrupt service "
            "for thousands of customers at once."
        ),
        "recommended_action": (
            "Treat as urgent. Isolate or rate-limit the source if possible, alert your network "
            "security team immediately, and check if upstream DDoS mitigation (e.g. traffic scrubbing) "
            "needs to be activated."
        ),
    },
    "PortScan": {
        "severity": "medium",
        "what_it_is": (
            "A Port Scan. An attacker is systematically probing a target system to find which "
            "network ports are open, which is usually the reconnaissance step before a more "
            "serious attack."
        ),
        "why_it_matters": (
            "On its own, a port scan doesn't cause damage -- but it's a strong early warning sign "
            "that someone is actively looking for a way into your systems. Attacks often follow "
            "shortly after."
        ),
        "recommended_action": (
            "Investigate the source IP. Consider this an early warning -- review firewall rules and "
            "watch closely for follow-up activity from the same source over the next few hours."
        ),
    },
    "Bot": {
        "severity": "high",
        "what_it_is": (
            "Botnet activity. This traffic pattern suggests the device may be infected with malware "
            "and is communicating with a remote attacker (a 'command and control' server), often as "
            "part of a larger network of compromised devices."
        ),
        "why_it_matters": (
            "An infected device can be used to launch attacks on others, steal data, or spread "
            "further malware -- all without the device owner's knowledge."
        ),
        "recommended_action": (
            "Treat as serious. Isolate the affected device from the network, run a full malware scan, "
            "and check for unusual outbound connections."
        ),
    },
    "FTP-Patator": {
        "severity": "high",
        "what_it_is": (
            "An FTP brute-force attack. An attacker is rapidly trying many username/password "
            "combinations to break into a File Transfer Protocol (FTP) server."
        ),
        "why_it_matters": (
            "If successful, the attacker gains access to files stored on the server -- which may "
            "include sensitive documents, configuration files, or customer data."
        ),
        "recommended_action": (
            "Lock or rate-limit the targeted account. Enable login attempt limits and consider "
            "disabling FTP in favour of a more secure file transfer protocol if not already in place."
        ),
    },
    "SSH-Patator": {
        "severity": "high",
        "what_it_is": (
            "An SSH brute-force attack. An attacker is rapidly trying many username/password "
            "combinations to gain remote command-line access to a server via SSH."
        ),
        "why_it_matters": (
            "SSH access often grants deep control over a server. A successful breach here can lead "
            "to full system compromise."
        ),
        "recommended_action": (
            "Block the source IP, enforce key-based authentication instead of passwords if possible, "
            "and review recent login attempts for any that succeeded."
        ),
    },
}

DEFAULT_ATTACK_INFO = {
    "severity": "medium",
    "what_it_is": "This traffic pattern was not specifically described in our knowledge base, but the model has flagged it as different from normal traffic.",
    "why_it_matters": "Unclassified anomalies can still represent a real risk and are worth investigating.",
    "recommended_action": "Review the flow manually and consult your security team if the pattern repeats.",
}

SEVERITY_STYLE = {
    "none":     {"emoji": "✅", "css_class": "status-safe",   "label": "No Threat"},
    "low":      {"emoji": "🟡", "css_class": "status-low",    "label": "Low Severity"},
    "medium":   {"emoji": "🟠", "css_class": "status-medium", "label": "Medium Severity"},
    "high":     {"emoji": "🔴", "css_class": "status-high",   "label": "High Severity"},
    "critical": {"emoji": "🚨", "css_class": "status-danger", "label": "Critical Severity"},
}

# Ordering used to sort the batch results table, worst-first
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


# ── Core single-row prediction logic ───────────────────────
# Shared by both the manual-entry form and the batch CSV pipeline,
# so the two paths can never disagree about how a flow is classified.
def predict_single_row(row_dict):
    """row_dict: a dict of {feature_name: value} for ANY subset of
    all_features (missing ones default to 0). Returns (label, confidence, proba_dict)."""
    full_vector = pd.DataFrame([{f: 0 for f in all_features}])
    for feat, val in row_dict.items():
        if feat in full_vector.columns:
            full_vector[feat] = val

    scaled = scaler.transform(full_vector)
    scaled_df = pd.DataFrame(scaled, columns=all_features)
    top_input = scaled_df[top_features].values

    pred = model.predict(top_input)[0]
    proba = model.predict_proba(top_input)[0]
    confidence = round(max(proba) * 100, 2)
    label = le.inverse_transform([pred])[0]
    proba_dict = {cls: round(p * 100, 2) for cls, p in zip(class_names, proba)}

    return label, confidence, proba_dict


def build_result_text(label, confidence, proba_dict):
    info = ATTACK_INFO.get(label, DEFAULT_ATTACK_INFO)
    style = SEVERITY_STYLE.get(info["severity"], SEVERITY_STYLE["medium"])

    if label == "BENIGN":
        header = f"{style['emoji']} BENIGN TRAFFIC  —  {style['label']}"
    else:
        header = f"{style['emoji']} ATTACK DETECTED: {label}  —  {style['label']}"

    result_text = (
        f"{header}\n"
        f"Confidence: {confidence}%\n"
        f"{'-' * 50}\n"
        f"WHAT THIS MEANS:\n{info['what_it_is']}\n\n"
        f"WHY IT MATTERS:\n{info['why_it_matters']}\n\n"
        f"RECOMMENDED ACTION:\n{info['recommended_action']}\n"
        f"{'-' * 50}\n"
    )

    proba_lines = "\n".join(
        f"   {cls}: {p}%" for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1])
    )
    result_text += f"\nFull probability breakdown:\n{proba_lines}"

    return result_text, ["result-box", style["css_class"]]


# ── Manual-entry prediction function ───────────────────────
def predict_threat(*values):
    input_dict = dict(zip(top_features, values))
    label, confidence, proba_dict = predict_single_row(input_dict)
    result_text, css_classes = build_result_text(label, confidence, proba_dict)
    return gr.update(value=result_text, elem_classes=css_classes)


# ── Load Sample function ──────────────────────────────────
def load_sample(sample_type):
    class_samples = samples_by_class.get(sample_type)
    if class_samples is None or len(class_samples) == 0:
        return [0 for _ in top_features]
    row = class_samples.sample(1).iloc[0]
    return [row[feat] for feat in top_features]


# ── Batch CSV analysis function ────────────────────────────
def analyse_batch(file):
    if file is None:
        return (
            "No file uploaded yet. Please choose a CSV file above first.",
            ""
        )

    try:
        batch_df = pd.read_csv(file.name, low_memory=False)
    except Exception as e:
        return (f"Could not read this file as a CSV. Error: {e}", "")

    # Clean column names the same way training does, so a CSV exported
    # from the same kind of tool (e.g. CICFlowMeter) lines up correctly.
    batch_df.columns = batch_df.columns.str.strip()

    # Drop a Label column if present (so users can upload labelled
    # CICIDS-style files too, without it interfering with prediction).
    label_col = None
    for candidate in ["Label", "label", " Label"]:
        if candidate in batch_df.columns:
            label_col = candidate
            break
    true_labels = batch_df[label_col] if label_col else None
    if label_col:
        batch_df = batch_df.drop(columns=[label_col])

    # Replace infinities/NaNs the same way training does, but here we
    # fill rather than drop, so every uploaded row gets a result.
    batch_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    batch_df.fillna(0, inplace=True)

    if len(batch_df) == 0:
        return ("The uploaded file had no usable rows.", "")

    # Cap how many rows we process in one go, to keep the free-tier
    # Space responsive. 5,000 rows is generous for a demo and still fast.
    MAX_ROWS = 5000
    truncated = len(batch_df) > MAX_ROWS
    if truncated:
        batch_df = batch_df.iloc[:MAX_ROWS]

    results = []
    for i, row in batch_df.iterrows():
        row_dict = {f: row[f] for f in all_features if f in batch_df.columns}
        try:
            label, confidence, _ = predict_single_row(row_dict)
        except Exception:
            label, confidence = "ERROR", 0.0

        info = ATTACK_INFO.get(label, DEFAULT_ATTACK_INFO)
        severity = info["severity"] if label != "ERROR" else "medium"

        results.append({
            "Row": i + 1,
            "Prediction": label,
            "Confidence (%)": confidence,
            "Severity": SEVERITY_STYLE.get(severity, SEVERITY_STYLE["medium"])["label"],
            "_severity_rank": SEVERITY_RANK.get(severity, 2),
        })

    results_df = pd.DataFrame(results)
    # Worst-first ordering: critical/high threats surface at the top
    # of the table rather than getting buried among hundreds of BENIGN rows.
    results_df = results_df.sort_values(
        by=["_severity_rank", "Confidence (%)"], ascending=[True, False]
    ).drop(columns=["_severity_rank"])

    # Build an HTML table instead of returning a gr.Dataframe value.
    # (gr.Dataframe currently has a Gradio/gradio_client schema bug that
    # crashes the whole app on startup -- rendering plain HTML avoids
    # that code path entirely while still giving a clean, styled table.)
    severity_row_class = {
        "No Threat": "row-safe",
        "Low Severity": "row-low",
        "Medium Severity": "row-medium",
        "High Severity": "row-high",
        "Critical Severity": "row-danger",
    }

    # Only show up to 500 rows in the table itself (the summary above
    # already covers the full dataset) so the page doesn't become huge.
    DISPLAY_CAP = 500
    display_df = results_df.head(DISPLAY_CAP)

    table_rows_html = ""
    for _, r in display_df.iterrows():
        row_class = severity_row_class.get(r["Severity"], "row-medium")
        table_rows_html += (
            f"<tr class='{row_class}'>"
            f"<td>{r['Row']}</td>"
            f"<td>{r['Prediction']}</td>"
            f"<td>{r['Confidence (%)']}%</td>"
            f"<td>{r['Severity']}</td>"
            f"</tr>"
        )

    table_note = ""
    if len(results_df) > DISPLAY_CAP:
        table_note = (
            f"<p class='table-note'>Showing the {DISPLAY_CAP} most severe rows out of "
            f"{len(results_df)} analysed. The summary above reflects all {len(results_df)} rows.</p>"
        )

    results_html = f"""
    <div class="batch-table-wrap">
        {table_note}
        <table class="batch-table">
            <thead>
                <tr><th>Row</th><th>Prediction</th><th>Confidence</th><th>Severity</th></tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
    </div>
    """

    # Build the summary
    counts = pd.Series([r["Prediction"] for r in results]).value_counts()
    total = len(results)
    threat_total = total - counts.get("BENIGN", 0)

    summary_lines = [
        f"Analysed {total} row(s)" + (f" (first {MAX_ROWS} of {len(batch_df)} -- file truncated)" if truncated else "") + ".",
        f"Threats detected: {threat_total} / {total}",
        "",
        "Breakdown by traffic type:",
    ]
    for cls in class_names:
        c = counts.get(cls, 0)
        if c > 0:
            pct = round((c / total) * 100, 1)
            summary_lines.append(f"   {cls}: {c} ({pct}%)")

    summary_text = "\n".join(summary_lines)

    return summary_text, results_html


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
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border-width: 2px !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #334155 !important;
    font-family: Consolas, monospace !important;
    line-height: 1.5 !important;
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}

.status-safe textarea {
    background: #f0fdf4 !important;
    border-color: #22c55e !important;
    color: #15803d !important;
}
.status-low textarea {
    background: #fefce8 !important;
    border-color: #eab308 !important;
    color: #854d0e !important;
}
.status-medium textarea {
    background: #fff7ed !important;
    border-color: #f97316 !important;
    color: #9a3412 !important;
}
.status-high textarea {
    background: #fef2f2 !important;
    border-color: #dc2626 !important;
    color: #991b1b !important;
}
.status-danger textarea {
    background: #fef2f2 !important;
    border-color: #ef4444 !important;
    color: #b91c1c !important;
    font-weight: 700 !important;
}

.batch-summary textarea {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    border-width: 2px !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #1e293b !important;
    font-family: Consolas, monospace !important;
    line-height: 1.5 !important;
}

.feature-hint {
    font-size: 11px !important;
    color: #94a3b8 !important;
    margin: -8px 0 8px 4px !important;
}

/* Batch results table (replaces gr.Dataframe to avoid a Gradio schema bug) */
.batch-table-wrap {
    max-height: 500px;
    overflow-y: auto;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    background: #ffffff;
}
.batch-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.batch-table thead th {
    position: sticky;
    top: 0;
    background: #1e293b;
    color: #ffffff;
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
}
.batch-table tbody td {
    padding: 8px 14px;
    border-bottom: 1px solid #e2e8f0;
}
.batch-table tbody tr.row-safe   { background: #f0fdf4; color: #15803d; }
.batch-table tbody tr.row-low    { background: #fefce8; color: #854d0e; }
.batch-table tbody tr.row-medium { background: #fff7ed; color: #9a3412; }
.batch-table tbody tr.row-high   { background: #fef2f2; color: #991b1b; }
.batch-table tbody tr.row-danger { background: #fef2f2; color: #b91c1c; font-weight: 700; }
.table-note {
    font-size: 12px;
    color: #64748b;
    padding: 10px 14px;
    margin: 0;
    font-style: italic;
}
"""


# ── Build Gradio Interface ────────────────────────────────
with gr.Blocks(title="Telecom Threat Detector", theme=custom_theme, css=custom_css) as app:

    gr.Markdown(f"""
    # 🛡️ Telecom Network Threat Detector
    ### AI-powered multi-attack intrusion detection using CICIDS 2017 dataset
    *Based on: Using AI in Cyber Security Risk Management for Telecom Industry 4.0*
    ---
    This model can detect **{len(class_names)} traffic types**: {", ".join(class_names)}.
    """)

    with gr.Tabs():

        # ───────────── TAB 1: Single Flow Analysis ─────────────
        with gr.Tab("🔍 Single Flow Analysis"):

            gr.Markdown(
                "Enter the network flow features below and click **Analyse** to detect threats, "
                "or load a real test sample for any class using the controls below."
            )

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

            box_lookup = {}
            grouped = build_groups(top_features)

            for i, (group_title, group_features) in enumerate(grouped):
                with gr.Accordion(group_title, open=(i == 0)):
                    for feat in group_features:
                        box_lookup[feat] = gr.Number(label=feat, value=0)
                        hint = FEATURE_INFO.get(feat)
                        if hint:
                            gr.Markdown(f"*{hint}*", elem_classes=["feature-hint"])

            ordered_inputs = [box_lookup[feat] for feat in top_features]

            analyse_btn = gr.Button("🔍 Analyse Traffic", variant="primary", size="lg")

            gr.Markdown("### 🎯 Detection Result")
            result_box = gr.Textbox(
                label="AI Analysis Result",
                lines=16,
                interactive=False,
                elem_classes=["result-box"]
            )

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

        # ───────────── TAB 2: Batch CSV Analysis ─────────────
        with gr.Tab("📂 Batch CSV Analysis"):

            gr.Markdown("""
            ### Analyse Many Flows at Once
            Upload a CSV file containing network flow data (the same format as the CICIDS 2017
            dataset -- one row per flow, with the same column names the model was trained on).
            Every row will be classified individually.

            If your file includes the original `Label` column, it will simply be ignored for
            prediction purposes -- you don't need to remove it first.
            """)

            csv_input = gr.File(label="Upload CSV file", file_types=[".csv"])
            batch_btn = gr.Button("📊 Analyse Batch", variant="primary", size="lg")

            gr.Markdown("### 📈 Summary")
            batch_summary = gr.Textbox(
                label="Batch Summary",
                lines=8,
                interactive=False,
                elem_classes=["batch-summary"]
            )

            gr.Markdown("### 📋 Per-Row Results")
            gr.Markdown("*Sorted with the most severe detections first.*")
            batch_table = gr.HTML(value="<p class='table-note'>Results will appear here after analysis.</p>")

            batch_btn.click(
                fn=analyse_batch,
                inputs=csv_input,
                outputs=[batch_summary, batch_table]
            )

    gr.Markdown("""
    ---
    ### 🧪 Tip
    Use **Single Flow Analysis** to test one example at a time and see detailed plain-English
    explanations, or use **Batch CSV Analysis** to scan many flows from a real traffic log at once.
    """)

app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)