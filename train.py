# ============================================================
# train.py — CICIDS 2017 Multi-Attack Threat Detection Pipeline
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("  CICIDS 2017 Multi-Attack Threat Detection — Training")
print("=" * 60)

# ── Step 1: Load and combine multiple CSV files ───────────
print("\n[1/8] Loading dataset files...")

csv_files = [
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv',
]

dataframes = []
for file in csv_files:
    path = os.path.join('data', file)
    if os.path.exists(path):
        df_temp = pd.read_csv(path, low_memory=False)
        dataframes.append(df_temp)
        print(f"      Loaded {file} — {df_temp.shape[0]:,} rows")
    else:
        print(f"      WARNING: {file} not found in data/ — skipping")

if not dataframes:
    raise FileNotFoundError("No CSV files found in data/. Check your file names and location.")

df = pd.concat(dataframes, ignore_index=True)
print(f"      Total combined: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ── Step 2: Clean data ────────────────────────────────────
print("\n[2/8] Cleaning data...")
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"      Clean shape: {df.shape[0]:,} rows")
print(f"      Label distribution:\n{df['Label'].value_counts()}")

# ── Step 3: Encode labels (multiclass) ────────────────────
print("\n[3/8] Encoding labels...")
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print(f"      Classes found ({len(le.classes_)} total):")
for i, name in enumerate(le.classes_):
    print(f"        {i} -> {name}")

# ── Step 4: Feature selection ─────────────────────────────
print("\n[4/8] Selecting features...")
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train a quick initial model just to measure feature importance
rf_init = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_init.fit(X_train_scaled, y_train)

importances = pd.Series(rf_init.feature_importances_, index=X.columns)
top_features = importances.nlargest(20).index.tolist()
print(f"      Top 20 features selected:")
for f in top_features:
    print(f"        - {f}")

X_train_top = pd.DataFrame(X_train_scaled, columns=X.columns)[top_features].values
X_test_top  = pd.DataFrame(X_test_scaled,  columns=X.columns)[top_features].values

# ── Step 5: Train final model (anti-overfit settings) ─────
print("\n[5/8] Training Random Forest (multiclass)...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'   # helps when some attack types have far fewer rows than BENIGN
)
model.fit(X_train_top, y_train)
print("      Training complete!")

# ── Step 6: Evaluate ──────────────────────────────────────
print("\n[6/8] Evaluating model...")
train_acc = accuracy_score(y_train, model.predict(X_train_top)) * 100
test_acc  = accuracy_score(y_test,  model.predict(X_test_top))  * 100
gap = train_acc - test_acc

print(f"      Training Accuracy : {round(train_acc, 2)}%")
print(f"      Testing Accuracy  : {round(test_acc, 2)}%")
print(f"      Gap               : {round(gap, 2)}%")

if gap < 2:
    print("      No overfitting detected!")
else:
    print("      Overfitting present — gap is high")

print("\n      Classification Report:")
print(classification_report(
    y_test,
    model.predict(X_test_top),
    target_names=le.classes_
))

# Confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test_top))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix — Multi-Attack Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
print("      Confusion matrix saved to model/confusion_matrix.png")

# ── Step 7: Save model files ──────────────────────────────
print("\n[7/8] Saving model files...")
joblib.dump(model,        'model/final_model.pkl')
joblib.dump(scaler,       'model/final_scaler.pkl')
joblib.dump(le,           'model/final_label_encoder.pkl')
joblib.dump(top_features, 'model/top_features.pkl')
joblib.dump(list(X.columns), 'model/all_features.pkl')

print("      Saved:")
print("         model/final_model.pkl")
print("         model/final_scaler.pkl")
print("         model/final_label_encoder.pkl")
print("         model/top_features.pkl")
print("         model/all_features.pkl")

# ── Step 8: Save sample rows per class for testing ────────
print("\n[8/8] Saving sample rows for each attack type...")

X_test_original_top = X_test[top_features].copy()
X_test_original_top['Label'] = y_test.values

sample_dict = {}
for class_idx, class_name in enumerate(le.classes_):
    class_rows = X_test_original_top[X_test_original_top['Label'] == class_idx]
    n_samples = min(20, len(class_rows))
    if n_samples > 0:
        samples = class_rows.sample(n_samples, random_state=42).drop('Label', axis=1)
        sample_dict[class_name] = samples
        print(f"      Saved {n_samples} samples for class: {class_name}")
    else:
        print(f"      WARNING: no test rows found for class: {class_name}")

joblib.dump(sample_dict, 'model/samples_by_class.pkl')
print("      Saved: model/samples_by_class.pkl")

print("\n" + "=" * 60)
print("  Training complete! Model now detects multiple attack types.")
print("=" * 60)