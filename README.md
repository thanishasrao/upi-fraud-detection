# 🛡️ UPI Fraud Detection System

A machine learning system that detects fraudulent UPI (Unified Payments Interface) transactions in real time, built as an end-to-end Data Science + FinTech project relevant to India's digital payments ecosystem.

## 🎯 Overview

This project classifies UPI transactions as **legitimate** or **fraudulent** using a Random Forest classifier trained on engineered behavioral features, and serves predictions through a live interactive dashboard.

**Live demo:** run `streamlit run dashboard.py` to launch the dashboard locally.

## 📊 Results

| Metric | Score |
|---|---|
| ROC-AUC | 1.0000 |
| Fraud Recall | 60 / 60 (100%) |
| False Positives | 2 out of 1,940 legitimate transactions |
| Precision / Recall / F1 | 1.00 / 1.00 / 1.00 |

## ⚙️ How It Works

1. **Data generation** — 10,000 synthetic UPI transactions with realistic statistical properties (97% legitimate, 3% fraud)
2. **Feature engineering** — 16 ML-ready features across time, amount, velocity, receiver risk, merchant category, and device signals
3. **Model training** — Random Forest (200 trees) with `class_weight='balanced'` to handle severe class imbalance
4. **Evaluation** — classification report, confusion matrix, ROC-AUC, feature importance analysis
5. **Dashboard** — Streamlit app for live transaction scoring (HIGH / MEDIUM / LOW risk verdicts)

## 🧠 Key Features Engineered

| Feature | Why it matters |
|---|---|
| `velocity_1h` | Catches rapid repeated transactions from the same sender |
| `is_micro_txn` | Flags ₹1–9 "probe" transactions fraudsters use to test active accounts |
| `is_off_hours` | 1AM–5AM window has disproportionately higher fraud rates |
| `is_high_risk_merchant` | Lottery, crypto, and unknown merchant categories |
| `is_new_device` + `device_age_days` | New/unrecognised devices combined with large amounts |
| `is_rare_receiver` | Receivers with very few past transactions are suspicious |

## 🛠️ Tech Stack

- **Python 3.13**
- **pandas / numpy** — data manipulation
- **scikit-learn** — Random Forest, train/test split, evaluation metrics
- **matplotlib / seaborn** — evaluation charts
- **Streamlit** — interactive dashboard
- **joblib** — model serialization

## 🚀 Getting Started

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib

# Train the model (generates fraud_model.pkl)
python fraud_detection.py

# Launch the dashboard
streamlit run dashboard.py
```

## 📁 Project Structure

```
├── fraud_detection.py    # Full ML pipeline: data, features, training, evaluation
├── dashboard.py           # Streamlit dashboard with live transaction scorer
├── fraud_model.pkl        # Saved trained Random Forest model
├── fraud_results.png      # Confusion matrix + feature importance charts
└── UPI_Fraud_Detection_Report.docx   # Full project report
```

## 📄 Full Report

See [`UPI_Fraud_Detection_Report.docx`](./UPI_Fraud_Detection_Report.docx) for the complete write-up covering problem statement, dataset design, feature engineering rationale, model architecture, and results.

## ⚠️ Note on Data

The dataset used here is **synthetically generated** to mirror realistic UPI transaction patterns, since real UPI transaction data is confidential and not publicly available. The fraud patterns encoded (micro-transaction probing, off-hours activity, high-risk merchant categories, new-device usage) are based on documented real-world UPI fraud typologies.
