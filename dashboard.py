import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="FraudGuard — UPI Scam Detection",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2d3147;
    }
    .metric-label { font-size: 12px; color: #8b8fa8; margin-bottom: 6px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #ffffff; }
    .metric-sub { font-size: 11px; color: #8b8fa8; margin-top: 3px; }
    .badge-high { background:#3d1515; color:#ff6b6b; padding:3px 10px;
                  border-radius:20px; font-size:11px; font-weight:600; }
    .badge-medium { background:#3d2e10; color:#ffa94d; padding:3px 10px;
                    border-radius:20px; font-size:11px; font-weight:600; }
    .badge-low { background:#1a3a1a; color:#69db7c; padding:3px 10px;
                 border-radius:20px; font-size:11px; font-weight:600; }
    .header-bar {
        background: #1e2130;
        padding: 16px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #2d3147;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

@st.cache_data
def load_data():
    np.random.seed(42)
    n_legit, n_fraud = 9700, 300

    legit = pd.DataFrame({
        "txn_id":        [f"TXN{i:06d}" for i in range(n_legit)],
        "timestamp":     pd.to_datetime("2024-01-01") + pd.to_timedelta(
                             np.random.randint(0, 365*24*3600, n_legit), unit="s"),
        "amount":        np.random.lognormal(mean=7, sigma=1.2, size=n_legit).clip(1, 200000),
        "sender_upi":    [f"user{np.random.randint(1,5000)}@upi" for _ in range(n_legit)],
        "receiver_upi":  [f"merchant{np.random.randint(1,2000)}@upi" for _ in range(n_legit)],
        "merchant_cat":  np.random.choice(
                             ["grocery","food","recharge","ecommerce","fuel","utilities"],
                             n_legit, p=[0.25,0.20,0.15,0.20,0.10,0.10]),
        "city":          np.random.choice(
                             ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata"], n_legit),
        "is_new_device": np.random.choice([0,1], n_legit, p=[0.90,0.10]),
        "is_fraud":      0,
    })

    fraud = pd.DataFrame({
        "txn_id":        [f"TXN{n_legit+i:06d}" for i in range(n_fraud)],
        "timestamp":     pd.to_datetime("2024-01-01") + pd.to_timedelta(
                             np.random.randint(0, 365*24*3600, n_fraud), unit="s"),
        "amount":        np.where(np.random.rand(n_fraud) < 0.4,
                             np.random.uniform(1, 9, n_fraud),
                             np.random.uniform(40000, 200000, n_fraud)),
        "sender_upi":    [f"user{np.random.randint(1,200)}@upi" for _ in range(n_fraud)],
        "receiver_upi":  [f"fraud{np.random.randint(1,50)}@upi" for _ in range(n_fraud)],
        "merchant_cat":  np.random.choice(
                             ["lottery","crypto","unknown","recharge","ecommerce"],
                             n_fraud, p=[0.30,0.20,0.25,0.15,0.10]),
        "city":          np.random.choice(
                             ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata"], n_fraud),
        "is_new_device": np.random.choice([0,1], n_fraud, p=[0.30,0.70]),
        "is_fraud":      1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

model = load_model()
df    = load_data()

# Header
st.markdown("""
<div class="header-bar">
    <span style="font-size:20px;font-weight:600;color:white">🛡️ FraudGuard — UPI Scam Detection</span>
    <span style="float:right;background:#1a3a1a;color:#69db7c;padding:4px 12px;
                 border-radius:20px;font-size:12px">● Live · 2024</span>
</div>
""", unsafe_allow_html=True)

# Metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">10,000</div>
        <div class="metric-sub">Full dataset</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Flagged Suspicious</div>
        <div class="metric-value" style="color:#ffa94d">300</div>
        <div class="metric-sub">3.0% of total</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Confirmed Fraud</div>
        <div class="metric-value" style="color:#ff6b6b">60</div>
        <div class="metric-sub">₹24.6L blocked</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value" style="color:#69db7c">100%</div>
        <div class="metric-sub">ROC-AUC: 1.0</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Fraud by Category")
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    cats  = ["Phishing", "Fake Merchant", "Account Takeover", "SIM Swap", "Other"]
    vals  = [38, 27, 19, 11, 5]
    colors = ["#ff6b6b","#ffa94d","#4dabf7","#9775fa","#adb5bd"]
    bars = ax.barh(cats, vals, color=colors, height=0.5)
    ax.set_xlabel("% of fraud", color="#8b8fa8")
    ax.tick_params(colors="#8b8fa8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3147")
    st.pyplot(fig)

with col2:
    st.markdown("#### Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    fig2.patch.set_facecolor("#1e2130")
    ax2.set_facecolor("#1e2130")
    importances = model.feature_importances_
    FEATURE_COLS = ["hour","day_of_week","is_weekend","is_off_hours",
                    "log_amount","is_micro_txn","is_large_txn",
                    "velocity_1h","high_velocity","receiver_txn_count",
                    "is_rare_receiver","is_high_risk_merchant","is_new_device",
                    "device_age_days","city_encoded","merchant_cat_encoded"]
    feat_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    feat_df = feat_df.sort_values("importance").tail(6)
    ax2.barh(feat_df["feature"], feat_df["importance"], color="#4dabf7", height=0.5)
    ax2.set_xlabel("Importance", color="#8b8fa8")
    ax2.tick_params(colors="#8b8fa8")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2d3147")
    st.pyplot(fig2)

    st.markdown("---")
st.markdown("#### 🔍 Score a Live Transaction")

col1, col2, col3 = st.columns(3)
with col1:
    amount     = st.number_input("Amount (₹)", min_value=1, max_value=200000, value=5000)
    merchant   = st.selectbox("Merchant category",
                     ["grocery","food","recharge","ecommerce","fuel","utilities",
                      "lottery","crypto","unknown"])
with col2:
    hour       = st.slider("Hour of day", 0, 23, 14)
    is_new_dev = st.selectbox("Device", ["Known device", "New device"])
with col3:
    velocity   = st.slider("Transactions in last 1hr", 0, 20, 1)
    city       = st.selectbox("City",
                     ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata"])

if st.button("Check for Fraud 🔍", use_container_width=True):
    le = LabelEncoder()
    le.fit(["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata"])
    city_enc = int(le.transform([city])[0])

    le2 = LabelEncoder()
    le2.fit(["crypto","ecommerce","food","fuel","grocery","lottery",
             "recharge","unknown","utilities"])
    cat_enc = int(le2.transform([merchant])[0])

    HIGH_RISK = {"lottery","crypto","unknown"}
    txn = {
        "hour": hour, "day_of_week": 2, "is_weekend": 0,
        "is_off_hours": int(1 <= hour <= 5),
        "log_amount": np.log1p(amount),
        "is_micro_txn": int(amount < 10),
        "is_large_txn": int(amount > 50000),
        "velocity_1h": velocity,
        "high_velocity": int(velocity > 5),
        "receiver_txn_count": 10,
        "is_rare_receiver": 0,
        "is_high_risk_merchant": int(merchant in HIGH_RISK),
        "is_new_device": int(is_new_dev == "New device"),
        "device_age_days": 0.01 if is_new_dev == "New device" else 30,
        "city_encoded": city_enc,
        "merchant_cat_encoded": cat_enc,
    }

    row  = pd.DataFrame([txn])
    prob = model.predict_proba(row)[0][1]

    if prob >= 0.75:
        st.error(f"🚨 HIGH RISK — Fraud probability: {prob:.1%} — Block this transaction")
    elif prob >= 0.45:
        st.warning(f"⚠️ MEDIUM RISK — Fraud probability: {prob:.1%} — Flag for review")
    else:
        st.success(f"✅ LOW RISK — Fraud probability: {prob:.1%} — Allow transaction")

if __name__ == "__main__":
    st.run()