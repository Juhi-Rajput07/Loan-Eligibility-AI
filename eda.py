import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
apps = pd.read_csv('loan_applications.csv')
txns = pd.read_csv('transactions.csv')

print("Files Loaded Successfully!")

# 2. FEATURE ENGINEERING (The 'How' of Fraud Detection)
# We aggregate transaction data to get a summary for each customer
txn_summary = txns.groupby('customer_id').agg({
    'transaction_amount': ['mean', 'sum', 'count'],
    'fraud_flag': 'sum'  # Counts how many fraudulent transactions they had
}).reset_index()

# Flatten the column names (e.g., 'transaction_amount_mean')
txn_summary.columns = ['customer_id', 'avg_spend', 'total_spend', 'txn_count', 'past_fraud_count']

# 3. MERGE DATASETS
# We link the Application data with our new Transaction Summary
df = pd.merge(apps, txn_summary, on='customer_id', how='left')

# 4. EXPLORATORY DATA ANALYSIS (EDA)

# A. Check for Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# B. Target Distribution (Eligibility)
plt.figure(figsize=(8,5))
sns.countplot(x='loan_status', data=df, palette='viridis')
plt.title('Distribution of Loan Status (Eligibility)')
plt.savefig('loan_status_dist.png') # Saves chart to folder
plt.show()

# C. Correlation Heatmap (What affects Approval?)
plt.figure(figsize=(12,8))
# We only use numeric columns for the heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_map.png')
plt.show()

# 5. ANOMALY DETECTION LOGIC (Quick Fraud Check)
# A simple rule-based check: Flag users who have > 0 past fraudulent transactions
df['high_risk_flag'] = (df['past_fraud_count'] > 0).astype(int)

print("\n--- High Risk Applications Detected ---")
print(df['high_risk_flag'].value_counts())

# Save the merged file for Step 2 (Machine Learning)
df.to_csv('cleaned_loan_data.csv', index=False)
print("\nSuccess! Merged data saved as 'cleaned_loan_data.csv'")
