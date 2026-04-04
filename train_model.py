import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load data
df = pd.read_csv('cleaned_loan_data.csv')

# 2. Clean columns
cols_to_drop = ['application_id', 'customer_id', 'application_date', 'residential_address', 'fraud_type']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. Handle Categorical Data & Save Encoders as a Dictionary
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'loan_status': # We handle target separately or let it encode
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Encode the Target (Loan Status)
target_le = LabelEncoder()
df['loan_status'] = target_le.fit_transform(df['loan_status'].astype(str))
encoders['loan_status'] = target_le

# 4. Train Model
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. SAVE EVERYTHING (The 3 keys to the UI)
joblib.dump(model, 'loan_model.pkl')
joblib.dump(encoders, 'encoders.pkl') 
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("✅ Success! Encoders saved as 'encoders.pkl'.")
print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")