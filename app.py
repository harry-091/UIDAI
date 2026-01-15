import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="UIDAI Data Analytics Dashboard",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("UIDAI Aadhaar Data Analytics Dashboard")
st.markdown("Yay!!!!!!!!!")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("uidai_data.csv")

df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

state_filter = st.sidebar.multiselect(
    "Select State(s)",
    options=df["State"].unique(),
    default=df["State"].unique()
)

year_filter = st.sidebar.multiselect(
    "Select Year(s)",
    options=sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)

filtered_df = df[
    (df["State"].isin(state_filter)) &
    (df["Year"].isin(year_filter))
]

# -----------------------------
# KPI Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Enrolments", f"{filtered_df['Enrolments'].sum():,}")
col2.metric("Total Updates", f"{filtered_df['Updates'].sum():,}")
col3.metric("Avg Approval Rate", f"{filtered_df['Approval_Rate'].mean():.2f}%")

st.divider()

# -----------------------------
# State-wise Enrolments
# -----------------------------
st.subheader("State-wise Aadhaar Enrolments")

state_enrol = filtered_df.groupby("State")["Enrolments"].sum()

fig1, ax1 = plt.subplots()
state_enrol.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Enrolments")
ax1.set_xlabel("State")

st.pyplot(fig1)

# -----------------------------
# Urban vs Rural
# -----------------------------
st.subheader("Urban vs Rural Comparison")

ur = filtered_df.groupby("Urban_Rural")[["Enrolments", "Updates"]].sum()

fig2, ax2 = plt.subplots()
ur.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Count")

st.pyplot(fig2)

# -----------------------------
# Update Type Distribution
# -----------------------------
st.subheader("Aadhaar Update Type Distribution")

update_types = filtered_df[
    ["Mobile_Updates", "Address_Updates", "DOB_Updates", "Biometric_Updates"]
].sum()

fig3, ax3 = plt.subplots()
ax3.pie(update_types, labels=update_types.index, autopct="%1.1f%%")
ax3.set_title("Update Types")

st.pyplot(fig3)

# -----------------------------
# Year-wise Trend
# -----------------------------
st.subheader("Year-wise Enrolment Trend")

yearly = filtered_df.groupby("Year")["Enrolments"].sum()

fig4, ax4 = plt.subplots()
ax4.plot(yearly.index, yearly.values, marker="o")
ax4.set_ylabel("Enrolments")
ax4.set_xlabel("Year")
ax4.grid()

st.pyplot(fig4)

# -----------------------------
# Raw Data
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df)

st.markdown("---")
st.caption("UIDAI Data Hackathon 2026 • Analytics Dashboard")
