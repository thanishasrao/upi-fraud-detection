import pandas as pd
import numpy as np

np.random.seed(42)  # makes results reproducible every run
n_legit = 9700

legit = pd.DataFrame({
    "txn_id":        [f"TXN{i:06d}" for i in range(n_legit)],
    "timestamp":     pd.to_datetime("2024-01-01") + pd.to_timedelta(
                         np.random.randint(0, 365*24*3600, n_legit), unit="s"),
    "amount":        np.random.lognormal(mean=7, sigma=1.2, size=n_legit).clip(1, 200000),
    "sender_upi":    [f"user{np.random.randint(1, 5000)}@upi" for _ in range(n_legit)],
    "receiver_upi":  [f"merchant{np.random.randint(1, 2000)}@upi" for _ in range(n_legit)],
    "device_id":     [f"DEV{np.random.randint(1, 8000):05d}" for _ in range(n_legit)],
    "merchant_cat":  np.random.choice(
                         ["grocery", "food", "recharge", "ecommerce", "fuel", "utilities"],
                         n_legit, p=[0.25, 0.20, 0.15, 0.20, 0.10, 0.10]),
    "city":          np.random.choice(
                         ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"],
                         n_legit),
    "is_new_device": np.random.choice([0, 1], n_legit, p=[0.90, 0.10]),
    "is_fraud":      0,
})
n_fraud = 300

fraud = pd.DataFrame({
    "txn_id":        [f"TXN{n_legit+i:06d}" for i in range(n_fraud)],
    "timestamp":     pd.to_datetime("2024-01-01") + pd.to_timedelta(
                         np.random.randint(0, 365*24*3600, n_fraud), unit="s"),
    "amount":        np.where(
                         np.random.rand(n_fraud) < 0.4,
                         np.random.uniform(1, 9, n_fraud),           # ₹1-9 probe txns
                         np.random.uniform(40000, 200000, n_fraud),  # large transfers
                     ),
    "sender_upi":    [f"user{np.random.randint(1, 200)}@upi" for _ in range(n_fraud)],
    "receiver_upi":  [f"fraud{np.random.randint(1, 50)}@upi" for _ in range(n_fraud)],
    "device_id":     [f"DEV{np.random.randint(8001, 9000):05d}" for _ in range(n_fraud)],
    "merchant_cat":  np.random.choice(
                         ["lottery", "crypto", "unknown", "recharge", "ecommerce"],
                         n_fraud, p=[0.30, 0.20, 0.25, 0.15, 0.10]),
    "city":          np.random.choice(
                         ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"],
                         n_fraud),
    "is_new_device": np.random.choice([0, 1], n_fraud, p=[0.30, 0.70]),
    "is_fraud":      1,
})
df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Check your data
print(df.shape)
print(df["is_fraud"].value_counts())
print(df.head())
# Sort by time first - important for velocity calculation later
df = df.sort_values("timestamp").reset_index(drop=True)

# Extract useful parts from the timestamp
df["hour"]         = df["timestamp"].dt.hour        # 0-23
df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)  # 1 if Sat/Sun
df["is_off_hours"] = df["hour"].between(1, 5).astype(int)  # 1 if 1am-5am

print(df[["timestamp", "hour", "is_weekend", "is_off_hours"]].head(8))
# log_amount: shrinks huge numbers (₹1 and ₹200000) into a smaller range
# log(1)=0, log(1000)=6.9, log(200000)=12.2  — model learns better this way
df["log_amount"]   = np.log1p(df["amount"])

# Two boolean flags that catch fraud patterns
df["is_micro_txn"] = (df["amount"] < 10).astype(int)     # ₹1-9 probe transactions
df["is_large_txn"] = (df["amount"] > 50000).astype(int)  # suspiciously large

print(df[["amount", "log_amount", "is_micro_txn", "is_large_txn"]].head(8))

# How many times has this sender transacted in the last 1 hour?
# Fraudsters send many rapid transactions — this catches them

df["ts_unix"] = df["timestamp"].astype(np.int64) // 10**9  # convert time to seconds

velocity_1h = []
for _, row in df.iterrows():
    past_1h = df[
        (df["sender_upi"] == row["sender_upi"]) &
        (df["ts_unix"] >= row["ts_unix"] - 3600) &
        (df["ts_unix"] <  row["ts_unix"])
    ]
    velocity_1h.append(len(past_1h))

df["velocity_1h"]   = velocity_1h
df["high_velocity"] = (df["velocity_1h"] > 5).astype(int)

print(df[["sender_upi", "velocity_1h", "high_velocity"]].sort_values("velocity_1h", ascending=False).head(8))
# Receivers who appear very rarely are suspicious (fraudsters use throwaway accounts)
receiver_counts          = df["receiver_upi"].value_counts()
df["receiver_txn_count"] = df["receiver_upi"].map(receiver_counts)
df["is_rare_receiver"]   = (df["receiver_txn_count"] < 3).astype(int)

# Directly encode domain knowledge: these categories are high risk
HIGH_RISK = {"lottery", "crypto", "unknown"}
df["is_high_risk_merchant"] = df["merchant_cat"].isin(HIGH_RISK).astype(int)

# How old is this device? Brand new device + large amount = red flag
device_first_seen    = df.groupby("device_id")["timestamp"].transform("min")
df["device_age_days"] = (df["timestamp"] - device_first_seen).dt.total_seconds() / 86400
# ML models only understand numbers, not strings like "Mumbai" or "grocery"
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["city_encoded"]          = le.fit_transform(df["city"])
df["merchant_cat_encoded"]  = le.fit_transform(df["merchant_cat"])

# Final check — see all new columns
print(df.shape)   # should now be (10000, 27) roughly
print(df.columns.tolist())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# These are the 17 features we engineered — the model sees ONLY these columns
FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "is_off_hours",
    "log_amount", "is_micro_txn", "is_large_txn",
    "velocity_1h", "high_velocity",
    "receiver_txn_count", "is_rare_receiver",
    "is_high_risk_merchant", "is_new_device", "device_age_days",
    "city_encoded", "merchant_cat_encoded",
]

X = df[FEATURE_COLS]   # inputs  (everything the model learns from)
y = df["is_fraud"]     # output  (what we want it to predict)

print("X shape:", X.shape)   # (10000, 16)
print("y counts:\n", y.value_counts())
# 80% of data to train the model, 20% kept aside to test it
# stratify=y makes sure both sets have the same 97/3 fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))   # 8000
print("Testing rows: ", len(X_test))    # 2000
# Problem: 9700 legit vs 300 fraud — model will just predict "legit" always
# Fix: tell the model to penalise missing a fraud case ~32x more than missing a legit one
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

print("Class weights:", class_weight_dict)
# You'll see something like {0: 0.51, 1: 16.6}
# Meaning: every fraud case counts as ~32x more important
model = RandomForestClassifier(
    n_estimators=200,       # build 200 decision trees and average their votes
    max_depth=12,           # each tree can ask at most 12 questions
    min_samples_leaf=5,     # each answer needs at least 5 transactions behind it
    max_features="sqrt",    # each tree sees only √16 ≈ 4 random features per split
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1               # use all CPU cores — runs faster
)

print("Training model... (takes ~10-20 seconds)")
model.fit(X_train, y_train)
print("Done! Model trained on", len(X_train), "transactions.")
# Test on the first 5 rows of test set
sample_preds = model.predict_proba(X_test[:5])[:, 1]
for i, prob in enumerate(sample_preds):
    label = "FRAUD" if prob > 0.5 else "legit"
    print(f"Transaction {i+1}: fraud probability = {prob:.1%}  →  {label}")
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # fraud probability for each row

print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc:.4f}")
# Anything above 0.90 is excellent
# 1.0 = perfect, 0.5 = random guessing
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted Legit   Predicted Fraud")
print(f"Actual Legit     {cm[0][0]:>10}        {cm[0][1]:>10}")
print(f"Actual Fraud     {cm[1][0]:>10}        {cm[1][1]:>10}")
print(f"\nCorrectly caught fraud: {cm[1][1]} out of {cm[1][0]+cm[1][1]}")
print(f"False alarms:           {cm[0][1]} legitimate transactions wrongly flagged")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("UPI Fraud Detection — Model Results", fontsize=14)

# Plot 1: Confusion matrix heatmap
ConfusionMatrixDisplay(cm, display_labels=["Legit","Fraud"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues"
)
axes[0].set_title("Confusion Matrix")

# Plot 2: Feature importance — what did the model learn mattered most?
importances = model.feature_importances_
feat_df = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1])
labels, values = zip(*feat_df)

axes[1].barh(labels, values, color="#378ADD")
axes[1].set_title("Feature Importance — what mattered most")
axes[1].set_xlabel("Importance score")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("fraud_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved as fraud_results.png")
import joblib

joblib.dump(model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl")

# To load it later in any other script:
# model = joblib.load("fraud_model.pkl")
def score_transaction(txn: dict) -> dict:
    """
    Pass in a single transaction as a dictionary.
    Returns fraud probability and risk level.
    """
    row = pd.DataFrame([txn])[FEATURE_COLS]
    prob = model.predict_proba(row)[0][1]

    if prob >= 0.75:
        risk = "HIGH — block"
    elif prob >= 0.45:
        risk = "MEDIUM — review"
    else:
        risk = "LOW — allow"

    return {"fraud_probability": f"{prob:.1%}", "decision": risk}


# Try a suspicious transaction
suspicious = {
    "hour": 3, "day_of_week": 6, "is_weekend": 1, "is_off_hours": 1,
    "log_amount": np.log1p(85000), "is_micro_txn": 0, "is_large_txn": 1,
    "velocity_1h": 8, "high_velocity": 1,
    "receiver_txn_count": 2, "is_rare_receiver": 1,
    "is_high_risk_merchant": 1, "is_new_device": 1, "device_age_days": 0.02,
    "city_encoded": 2, "merchant_cat_encoded": 3,
}

result = score_transaction(suspicious)
print(f"\nSuspicious transaction:")
print(f"  Fraud probability : {result['fraud_probability']}")
print(f"  Decision          : {result['decision']}")