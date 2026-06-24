# ============================================================
# train.py — CICIDS 2017 Threat Detection ML Pipeline
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

print("=" * 50)
print("  CICIDS 2017 Threat Detection — Training")
print("=" * 50)

# ── Step 1: Load dataset ──────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                 low_memory=False)
print(f"      Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# ── Step 2: Clean data ────────────────────────────────────
print("\n[2/7] Cleaning data...")
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"      Clean shape: {df.shape[0]:,} rows")
print(f"      Label distribution:\n{df['Label'].value_counts()}")

# ── Step 3: Encode labels ─────────────────────────────────
print("\n[3/7] Encoding labels...")
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print(f"      Classes: {list(le.classes_)}")

# ── Step 4: Feature selection ─────────────────────────────
print("\n[4/7] Selecting features...")
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train initial model to get feature importances
rf_init = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_init.fit(X_train_scaled, y_train)

importances = pd.Series(rf_init.feature_importances_, index=X.columns)
top_features = importances.nlargest(20).index.tolist()
print(f"      Top 20 features selected")

X_train_top = pd.DataFrame(X_train_scaled, columns=X.columns)[top_features].values
X_test_top  = pd.DataFrame(X_test_scaled,  columns=X.columns)[top_features].values

# ── Step 5: Train final model (anti-overfit settings) ─────
print("\n[5/7] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_top, y_train)
print("      Training complete!")

# ── Step 6: Evaluate ──────────────────────────────────────
print("\n[6/7] Evaluating model...")
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
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
print("      Confusion matrix saved to model/confusion_matrix.png")

# ── Step 7: Save model files ──────────────────────────────
print("\n[7/7] Saving model files...")
joblib.dump(model,        'model/final_model.pkl')
joblib.dump(scaler,       'model/final_scaler.pkl')
joblib.dump(le,           'model/final_label_encoder.pkl')
joblib.dump(top_features, 'model/top_features.pkl')
joblib.dump(list(X.columns), 'model/all_features.pkl')

# ── Step 8: Save sample rows for "Load Sample" feature ────
print("\n[8/8] Saving sample rows for testing...")

# Get original (unscaled) top-feature test data with true labels
X_test_original_top = X_test[top_features].copy()
X_test_original_top['Label'] = y_test.values

# Grab 20 random BENIGN samples and 20 random DDoS samples
benign_samples = X_test_original_top[X_test_original_top['Label'] == 0].sample(20, random_state=42)
ddos_samples   = X_test_original_top[X_test_original_top['Label'] == 1].sample(20, random_state=42)

joblib.dump(benign_samples.drop('Label', axis=1), 'model/sample_benign.pkl')
joblib.dump(ddos_samples.drop('Label', axis=1),   'model/sample_ddos.pkl')

print("      Saved: model/sample_benign.pkl")
print("      Saved: model/sample_ddos.pkl")

print("      Saved:")
print("         model/final_model.pkl")
print("         model/final_scaler.pkl")
print("         model/final_label_encoder.pkl")
print("         model/top_features.pkl")
print("         model/all_features.pkl")
print("\n" + "=" * 50)
print("  Training complete! Ready for deployment.")
print("=" * 50)