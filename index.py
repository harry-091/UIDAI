import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Helper: Load all CSV chunks
# =============================
def load_all_csvs(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# =============================
# Load Chunked Datasets
# =============================
bio = load_all_csvs("data/biometric/*.csv")
demo = load_all_csvs("data/demographic/*.csv")
enrol = load_all_csvs("data/enrolment/*.csv")

print("Biometric rows:", len(bio))
print("Demographic rows:", len(demo))
print("Enrolment rows:", len(enrol))


for df in [bio, demo, enrol]:
    df.dropna(inplace=True)
    df["date"] = pd.to_datetime(df["date"])

# =============================
# Standardize
# =============================
bio["bio_total"] = bio["bio_age_5_17"] + bio["bio_age_17_"]
print("Demographic columns:", demo.columns.tolist())

# =============================
# Feature Engineering (Schema-safe)
# =============================

# Biometric totals
bio["bio_total"] = bio["bio_age_5_17"] + bio["bio_age_17_"]

# Demographic totals (only 2 age groups exist)
demo["demo_total"] = demo["demo_age_5_17"] + demo["demo_age_17_"]

# Enrolment totals (3 age groups exist here)
enrol["enrol_total"] = (
    enrol["age_0_5"] +
    enrol["age_5_17"] +
    enrol["age_18_greater"]
)

enrol["enrol_total"] = enrol[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)

# =============================
# Merge All
# =============================
df = bio.merge(
    demo,
    on=["date", "state", "district", "pincode"],
    how="inner"
).merge(
    enrol,
    on=["date", "state", "district", "pincode"],
    how="inner"
)

print("Unified Dataset Shape:", df.shape)

# =============================
# 1️⃣ Biometric vs Demographic Imbalance
# =============================
df["bio_demo_ratio"] = df["bio_total"] / (df["demo_total"] + 1)

df["bio_demo_z"] = (
    df["bio_demo_ratio"] - df["bio_demo_ratio"].mean()
) / df["bio_demo_ratio"].std()

bio_demo_anomalies = df[df["bio_demo_z"].abs() > 2]

# =============================
# 2️⃣ Age-Shift Detection
# =============================
df["adult_share"] = df["age_18_greater"] / (df["demo_total"] + 1)
df["child_share"] = (df["age_0_5"] + df["age_5_17"]) / (df["demo_total"] + 1)

df["adult_share_z"] = (df["adult_share"] - df["adult_share"].mean()) / df["adult_share"].std()

age_shift_anomalies = df[df["adult_share_z"].abs() > 2]

# =============================
# 3️⃣ Enrolment ↔ Authentication Divergence
# =============================
df["auth_total"] = df["bio_total"] + df["demo_total"]
df["auth_enrol_ratio"] = df["auth_total"] / (df["enrol_total"] + 1)

df["aer_z"] = (
    df["auth_enrol_ratio"] - df["auth_enrol_ratio"].mean()
) / df["auth_enrol_ratio"].std()

dead_identity_zones = df[df["aer_z"] < -2]

# =============================
# 4️⃣ Temporal Shock Detection
# =============================
df.sort_values("date", inplace=True)

df["auth_delta"] = df.groupby(
    ["state", "district", "pincode"]
)["auth_total"].diff()

df["shock_z"] = (
    df["auth_delta"] - df["auth_delta"].mean()
) / df["auth_delta"].std()

system_shocks = df[df["shock_z"].abs() > 2.5]

# =============================
# 5️⃣ Aadhaar Health Index (AHI)
# =============================
df["stability"] = -df["shock_z"].abs().fillna(0)
df["security"] = df["bio_total"] / (df["auth_total"] + 1)
df["balance"] = 1 - df["child_share"]

df["AHI"] = (
    0.35 * df["stability"] +
    0.35 * df["security"] +
    0.30 * df["balance"]
)

ahi_region = df.groupby("state")["AHI"].mean().sort_values()

# =============================
# 6️⃣ Outputs
# =============================
print("\n🚨 Biometric–Demographic Anomalies:")
print(bio_demo_anomalies[["state", "district", "pincode", "bio_demo_ratio"]].head())

print("\n🚨 Age-Shift Anomalies:")
print(age_shift_anomalies[["state", "district", "adult_share"]].head())

print("\n🚨 Dead Identity Zones:")
print(dead_identity_zones[["state", "district", "auth_enrol_ratio"]].head())

print("\n🚨 System Shocks:")
print(system_shocks[["state", "district", "auth_delta"]].head())

print("\n📊 Aadhaar Health Index (Worst → Best):")
print(ahi_region)

# =============================
# Visualization: AHI
# =============================
plt.figure()
ahi_region.plot(kind="barh")
plt.title("Aadhaar Health Index by State")
plt.xlabel("AHI Score")
plt.tight_layout()
plt.show()
import os
print("Saving unified file to:", os.getcwd())
df.to_csv("unified_uidai.csv", index=False)
print("✅ unified_uidai.csv saved")

