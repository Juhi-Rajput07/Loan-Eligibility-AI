import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   🏦 LOAN GUARD AI — MULTI-MODEL TRAINING PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('cleaned_loan_data.csv')
print(f"\n✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. DROP IRRELEVANT COLUMNS
# ─────────────────────────────────────────────
cols_to_drop = ['application_id', 'customer_id', 'application_date',
                'residential_address', 'fraud_type']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# ─────────────────────────────────────────────
# 3. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'loan_status':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Encode target
target_le = LabelEncoder()
df['loan_status'] = target_le.fit_transform(df['loan_status'].astype(str))
encoders['loan_status'] = target_le

print(f"✅ Encoding complete. Classes: {list(target_le.classes_)}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5. DEFINE ALL 5 MODELS
# ─────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, class_weight='balanced', random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
}

# ─────────────────────────────────────────────
# 6. TRAIN, EVALUATE & COMPARE ALL MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("   📊 MODEL TRAINING & EVALUATION RESULTS")
print("=" * 60)

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n🔄 Training: {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cv    = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

    results[name] = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
        "cv_score":  round(float(cv),   4)
    }
    trained_models[name] = model

    print(f"   Accuracy : {acc:.4f}  |  Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}  |  F1-Score : {f1:.4f}")
    print(f"   CV Score : {cv:.4f}")

# ─────────────────────────────────────────────
# 7. PICK THE BEST MODEL (by CV score)
# ─────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['cv_score'])
best_model = trained_models[best_name]

print("\n" + "=" * 60)
print(f"   🏆 BEST MODEL: {best_name}")
print(f"   CV Accuracy : {results[best_name]['cv_score']:.4f}")
print("=" * 60)

# Full report for best model
y_best_pred = best_model.predict(X_test)
print(f"\n📋 Classification Report — {best_name}:\n")
print(classification_report(y_test, y_best_pred,
                             target_names=target_le.classes_))

# ─────────────────────────────────────────────
# 8. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────
# Best model (used by default in app)
joblib.dump(best_model,         'loan_model.pkl')
joblib.dump(encoders,           'encoders.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# All trained models (for model selector in app)
joblib.dump(trained_models, 'all_models.pkl')

# Results JSON (for comparison chart in app)
with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Best model name
with open('best_model_name.txt', 'w') as f:
    f.write(best_name)

print("\n✅ Saved: loan_model.pkl, encoders.pkl, model_columns.pkl")
print("✅ Saved: all_models.pkl, model_results.json, best_model_name.txt")
print("\n🎉 Training pipeline complete!\n")
