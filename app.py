import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="UIDAI Intelligence Dashboard", layout="wide")
st.title("UIDAI Aadhaar System Intelligence Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("unified_uidai.csv", parse_dates=["date"])

df = load_data()

# =============================
# Sidebar
# =============================
st.sidebar.header("Filters")
states = st.sidebar.multiselect(
    "Select State(s)", df["state"].unique(), df["state"].unique()
)

df = df[df["state"].isin(states)]

# =============================
# KPIs
# =============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Authentications", int(df["auth_total"].sum()))
c2.metric("Total Enrolments", int(df["enrol_total"].sum()))
c3.metric("Avg Bio/Demo Ratio", round(df["bio_demo_ratio"].mean(), 2))
c4.metric("Avg AHI", round(df["AHI"].mean(), 2))

st.divider()

# =============================
# Aadhaar Health Index
# =============================
st.subheader("Aadhaar Health Index by State")
ahi = df.groupby("state")["AHI"].mean()

fig, ax = plt.subplots()
ahi.plot(kind="bar", ax=ax)
ax.set_ylabel("AHI Score")
st.pyplot(fig)

# =============================
# Anomaly Explorer
# =============================
st.subheader("Detected Anomalies")

tab1, tab2, tab3 = st.tabs([
    "Biometric/Demo Imbalance",
    "Age Shift",
    "Dead Identity Zones"
])

with tab1:
    st.dataframe(df[df["bio_demo_z"].abs() > 2])

with tab2:
    st.dataframe(df[df["adult_share_z"].abs() > 2])

with tab3:
    st.dataframe(df[df["aer_z"] < -2])

# =============================
# Temporal Shock View
# =============================
st.subheader("System Shock Events")
st.dataframe(df[df["shock_z"].abs() > 2.5])

st.caption("UIDAI Hackathon • Predictive Identity Intelligence Platform")

