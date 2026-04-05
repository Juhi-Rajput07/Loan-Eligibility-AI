import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   🔍 LOAN GUARD AI — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
apps = pd.read_csv('loan_applications.csv')
txns = pd.read_csv('transactions.csv')
print(f"\n✅ Applications loaded : {apps.shape[0]} rows")
print(f"✅ Transactions loaded : {txns.shape[0]} rows")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING — Transaction Summary
# ─────────────────────────────────────────────
txn_summary = txns.groupby('customer_id').agg(
    avg_spend        = ('transaction_amount', 'mean'),
    total_spend      = ('transaction_amount', 'sum'),
    txn_count        = ('transaction_amount', 'count'),
    past_fraud_count = ('fraud_flag', 'sum')
).reset_index()

print(f"\n✅ Transaction summary built for {txn_summary.shape[0]} customers")

# ─────────────────────────────────────────────
# 3. MERGE DATASETS
# ─────────────────────────────────────────────
df = pd.merge(apps, txn_summary, on='customer_id', how='left')
print(f"✅ Merged dataset shape : {df.shape}")

# ─────────────────────────────────────────────
# 4. MISSING VALUES
# ─────────────────────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found ✅")

# Fill any numeric NAs with 0 (from left join)
df.fillna(0, inplace=True)

# ─────────────────────────────────────────────
# 5. TARGET DISTRIBUTION
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 5))
order = df['loan_status'].value_counts().index
ax = sns.countplot(x='loan_status', data=df, palette='viridis', order=order)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.title('Distribution of Loan Status', fontsize=14, fontweight='bold')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('loan_status_dist.png', dpi=150)
plt.show()
print("\n✅ Saved: loan_status_dist.png")

# ─────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ─────────────────────────────────────────────
plt.figure(figsize=(14, 9))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
mask = (numeric_df.corr().abs() > 0.0)  # Show all
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.4, mask=~mask)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_map.png', dpi=150)
plt.show()
print("✅ Saved: correlation_map.png")

# ─────────────────────────────────────────────
# 7. CIBIL SCORE DISTRIBUTION BY STATUS
# ─────────────────────────────────────────────
if 'cibil_score' in df.columns:
    plt.figure(figsize=(10, 5))
    for status in df['loan_status'].unique():
        subset = df[df['loan_status'] == status]['cibil_score']
        subset.plot.kde(label=status)
    plt.title('CIBIL Score Distribution by Loan Status', fontsize=13, fontweight='bold')
    plt.xlabel('CIBIL Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cibil_distribution.png', dpi=150)
    plt.show()
    print("✅ Saved: cibil_distribution.png")

# ─────────────────────────────────────────────
# 8. FRAUD / HIGH-RISK FLAGGING
# ─────────────────────────────────────────────
df['high_risk_flag'] = (df['past_fraud_count'] > 0).astype(int)

total     = len(df)
high_risk = df['high_risk_flag'].sum()
safe      = total - high_risk

print("\n--- Fraud Risk Summary ---")
print(f"  Total Applications : {total:,}")
print(f"  ⚠️  High-Risk       : {high_risk:,}  ({high_risk/total*100:.1f}%)")
print(f"  ✅ Low-Risk         : {safe:,}  ({safe/total*100:.1f}%)")

# ─────────────────────────────────────────────
# 9. SAVE CLEANED DATA
# ─────────────────────────────────────────────
df.to_csv('cleaned_loan_data.csv', index=False)
print("\n✅ Saved: cleaned_loan_data.csv")
print("\n🎉 EDA complete! Proceed to train_model.py\n")
