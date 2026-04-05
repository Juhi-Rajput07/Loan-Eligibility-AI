# 🏦 Loan Guard AI — Eligibility & Fraud Detection System

An end-to-end ML application using **5 classifiers** to predict loan approval and detect fraud.

---

## ✅ Files Required to Run

Place all these in the **same folder**:

| File | Source |
|---|---|
| `app.py` | ✅ Provided |
| `loan_model.pkl` | ✅ Your uploaded file |
| `label_encoder.pkl` | ✅ Your uploaded file |
| `requirements.txt` | ✅ Provided |

> **No CSV files needed.** The app loads directly from your `.pkl` files.

---

## ▶️ How to Run

```bash
# Step 1 — Install dependencies (only once)
pip install -r requirements.txt

# Step 2 — Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 Models in the App

| # | Model | Details |
|---|---|---|
| 1 | **Random Forest** ✅ | Your pre-trained model from `loan_model.pkl` — 99% accuracy |
| 2 | Naive Bayes | Probabilistic classifier benchmark |
| 3 | Logistic Regression | Linear model benchmark |
| 4 | Decision Tree | Rule-based benchmark |
| 5 | Gradient Boosting | Sequential ensemble benchmark |

---

## 🔑 Loan Status Classes (from your label_encoder.pkl)

| Class | Meaning |
|---|---|
| ✅ Approved | Loan approved |
| ❌ Declined | Loan rejected |
| 🚨 Fraudulent - Detected | Fraud caught |
| ⚠️ Fraudulent - Undetected | Potential undetected fraud |

---

## 📂 App Features

- **Tab 1 — Risk Assessment:** Fill applicant details → get approval decision + confidence scores per class
- **Tab 2 — Model Comparison:** Bar charts + metrics table comparing all 5 models
