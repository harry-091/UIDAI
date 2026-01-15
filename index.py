import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("uidai.data.csv")

# -----------------------------
# Basic Cleaning
# -----------------------------
df.dropna(inplace=True)
df['Year'] = df['Year'].astype(int)

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# 1. State-wise Enrolments
# -----------------------------
state_enrol = df.groupby("State")["Enrolments"].sum().sort_values(ascending=False)

plt.figure()
state_enrol.plot(kind="bar")
plt.title("State-wise Aadhaar Enrolments")
plt.ylabel("Total Enrolments")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Urban vs Rural Comparison
# -----------------------------
ur_compare = df.groupby("Urban_Rural")[["Enrolments", "Updates"]].sum()

plt.figure()
ur_compare.plot(kind="bar")
plt.title("Urban vs Rural Comparison")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Update Type Distribution
# -----------------------------
update_types = df[[
    "Mobile_Updates",
    "Address_Updates",
    "DOB_Updates",
    "Biometric_Updates"
]].sum()

plt.figure()
update_types.plot(kind="pie", autopct="%1.1f%%")
plt.title("Distribution of Aadhaar Update Types")
plt.ylabel("")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Approval Rate Analysis
# -----------------------------
plt.figure()
df.boxplot(column="Approval_Rate", by="State")
plt.title("Approval Rate by State")
plt.suptitle("")
plt.ylabel("Approval Rate (%)")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Processing Time Analysis
# -----------------------------
plt.figure()
df.groupby("State")["Processing_Days"].mean().plot(kind="bar")
plt.title("Average Processing Days by State")
plt.ylabel("Days")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Year-wise Growth
# -----------------------------
yearly = df.groupby("Year")["Enrolments"].sum()
growth = yearly.pct_change() * 100

plt.figure()
yearly.plot(marker="o")
plt.title("Year-wise Aadhaar Enrolment Trend")
plt.ylabel("Enrolments")
plt.grid()
plt.tight_layout()
plt.show()

print("\nYear-on-Year Growth (%):")
print(growth.round(2))

# -----------------------------
# 7. Anomaly Detection (Z-Score)
# -----------------------------
df["Z_Score"] = (df["Enrolments"] - df["Enrolments"].mean()) / df["Enrolments"].std()
anomalies = df[df["Z_Score"].abs() > 2]

print("\nDetected Anomalies:")
print(anomalies[["State", "District", "Enrolments", "Z_Score"]])

# -----------------------------
# 8. Correlation Analysis
# -----------------------------
corr = df[[
    "Enrolments",
    "Updates",
    "Rejections",
    "Processing_Days",
    "Approval_Rate"
]].corr()

print("\nCorrelation Matrix:")
print(corr)

# -----------------------------
# 9. Simple Forecast (Linear)
# -----------------------------
x = np.arange(len(yearly))
y = yearly.values

coeff = np.polyfit(x, y, 1)
forecast = np.polyval(coeff, x)

plt.figure()
plt.plot(yearly.index, y, label="Actual")
plt.plot(yearly.index, forecast, linestyle="--", label="Forecast")
plt.title("Enrolment Forecast (Linear)")
plt.legend()
plt.tight_layout()
plt.show()
