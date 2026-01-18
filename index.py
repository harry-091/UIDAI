import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper: Load all CSV chunks

def load_all_csvs(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# Load Chunked Datasets

bio = load_all_csvs("data/biometric/*.csv")
demo = load_all_csvs("data/demographic/*.csv")
enrol = load_all_csvs("data/enrolment/*.csv")

print("Biometric rows:", len(bio))
print("Demographic rows:", len(demo))
print("Enrolment rows:", len(enrol))


# Clean + Standardize

for df in [bio, demo, enrol]:
    df.dropna(inplace=True)
    df["date"] = pd.to_datetime(df["date"])


# Feature Engineering


# Biometric
bio["bio_total"] = bio["bio_age_5_17"] + bio["bio_age_17_"]

# Demographic
demo["demo_total"] = demo["demo_age_5_17"] + demo["demo_age_17_"]

# Enrolment
enrol["enrol_total"] = enrol[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)


# Merge All (Unified Dataset)

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


# Anomaly & Index Computation


# 1️ Biometric vs Demographic
df["bio_demo_ratio"] = df["bio_total"] / (df["demo_total"] + 1)
df["bio_demo_z"] = (df["bio_demo_ratio"] - df["bio_demo_ratio"].mean()) / df["bio_demo_ratio"].std()

# 2️ Age Shift
df["adult_share"] = df["age_18_greater"] / (df["demo_total"] + 1)
df["child_share"] = (df["age_0_5"] + df["age_5_17"]) / (df["demo_total"] + 1)
df["adult_share_z"] = (df["adult_share"] - df["adult_share"].mean()) / df["adult_share"].std()

# 3️3 Enrolment Divergence
df["auth_total"] = df["bio_total"] + df["demo_total"]
df["auth_enrol_ratio"] = df["auth_total"] / (df["enrol_total"] + 1)
df["aer_z"] = (df["auth_enrol_ratio"] - df["auth_enrol_ratio"].mean()) / df["auth_enrol_ratio"].std()

# 4️4 Temporal Shocks
df.sort_values("date", inplace=True)
df["auth_delta"] = df.groupby(["state", "district", "pincode"])["auth_total"].diff()
df["shock_z"] = (df["auth_delta"] - df["auth_delta"].mean()) / df["auth_delta"].std()


df["stability"] = -df["shock_z"].abs().fillna(0)
df["security"] = df["bio_total"] / (df["auth_total"] + 1)
df["balance"] = 1 - df["child_share"]

df["AHI"] = (
    0.35 * df["stability"] +
    0.35 * df["security"] +
    0.30 * df["balance"]
)


# Save Unified Dataset

print("Saving unified file to:", os.getcwd())
df.to_csv("unified_uidai.csv", index=False)
print(" unified_uidai.csv saved")


# STATE-LEVEL MAP FILES



state_map = df.groupby("state").agg({
    "bio_total": "sum",
    "demo_total": "sum",
    "enrol_total": "sum",
    "AHI": "mean",
    "child_share": "mean",
    "adult_share": "mean"
}).reset_index()

state_map.to_csv("state_map.csv", index=False)
print("🗺️ state_map.csv saved")

#  Demographic
state_map_demo = demo.groupby("state").agg({
    "demo_total": "sum",
    "demo_age_5_17": "sum",
    "demo_age_17_": "sum"
}).reset_index()

state_map_demo.to_csv("state_map_demographic.csv", index=False)
print("🗺️ state_map_demographic.csv saved")

#  Enrolment
state_map_enrol = enrol.groupby("state").agg({
    "enrol_total": "sum",
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum"
}).reset_index()

state_map_enrol.to_csv("state_map_enrolment.csv", index=False)
print("🗺️ state_map_enrolment.csv saved")


# DISTRICT-LEVEL MAP FILES



district_map_bio = df.groupby(["state", "district"]).agg({
    "bio_total": "sum",
    "AHI": "mean",
    "child_share": "mean",
    "adult_share": "mean"
}).reset_index()

district_map_bio.to_csv("district_map_biometric.csv", index=False)
print("🏙️ district_map_biometric.csv saved")

# 🔹 Demographic
district_map_demo = demo.groupby(["state", "district"]).agg({
    "demo_total": "sum",
    "demo_age_5_17": "sum",
    "demo_age_17_": "sum"
}).reset_index()

district_map_demo.to_csv("district_map_demographic.csv", index=False)
print("🏙️ district_map_demographic.csv saved")

# 🔹 Enrolment
district_map_enrol = enrol.groupby(["state", "district"]).agg({
    "enrol_total": "sum",
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum"
}).reset_index()

district_map_enrol.to_csv("district_map_enrolment.csv", index=False)
print("🏙️ district_map_enrolment.csv saved")

print("\n ALL MAP FILES GENERATED YAYYYY")
