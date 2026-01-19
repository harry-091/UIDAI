# ml_forecast.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# =============================
# Load Data
# =============================
df = pd.read_csv("unified_uidai.csv", parse_dates=["date"])

monthly = (
    df
    .set_index("date")
    .groupby("state")
    .resample("ME")
    .agg({
        "bio_total": "sum",
        "demo_total": "sum",
        "enrol_total": "sum"
    })
    .reset_index()
)

# =============================
# Feature Engineering
# =============================
def create_features(df, target):
    df = df.sort_values("date").copy()

    for lag in [1, 2, 3]:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    df["month"] = df["date"].dt.month
    df["trend"] = np.arange(len(df))

    return df.dropna().reset_index(drop=True)

# =============================
# Train Models (FIXED)
# =============================
targets = {
    "Biometric": "bio_total",
    "Demographic": "demo_total",
    "Enrolment": "enrol_total"
}

models = {k: {} for k in targets}

for dataset, target in targets.items():
    for state in monthly["state"].unique():
        raw = monthly[monthly["state"] == state]
        feats = create_features(raw, target)

        if len(feats) < 6:
            continue

        #  EXPLICIT FEATURE LIST (THE FIX)
        feature_cols = [
            f"{target}_lag_1",
            f"{target}_lag_2",
            f"{target}_lag_3",
            "month",
            "trend"
        ]

        X = feats[feature_cols]
        y = feats[target]

        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X, y)
        models[dataset][state] = model

joblib.dump(models, "models.pkl")
print("✅ Models trained correctly and saved")
