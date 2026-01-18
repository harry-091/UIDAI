import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import joblib


# Page Config

st.set_page_config(
    page_title="UIDAI Intelligence Dashboard",
    layout="wide"
)

st.title("UIDAI Aadhaar System Intelligence Dashboard")


# Data Loaders

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_unified():
    return pd.read_csv("unified_uidai.csv", parse_dates=["date"])

@st.cache_resource
def load_models():
    return joblib.load("models.pkl")

df = load_unified()
models = load_models()

# MONTHLY PREP

@st.cache_data
def prepare_monthly(df):
    return (
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

def create_features(df, target):
    df = df.sort_values("date").copy()

    for lag in [1, 2, 3]:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    df["month"] = df["date"].dt.month
    df["trend"] = np.arange(len(df))

    return df.dropna().reset_index(drop=True)


# FORECAST 

def forecast_future(monthly, models, state, dataset, months=6):
    target_map = {
        "Biometric": "bio_total",
        "Demographic": "demo_total",
        "Enrolment": "enrol_total"
    }

    target = target_map[dataset]
    hist = monthly[monthly["state"] == state].sort_values("date")

    if len(hist) < 3:
        return None

    # ---------- ML PATH ----------
    if state in models.get(dataset, {}):
        model = models[dataset][state]
        feats = create_features(hist, target)
        trend_start = feats["trend"].iloc[-1]

        future = []

        for i in range(1, months + 1):
            last = feats.tail(3)

            row = {
                f"{target}_lag_1": last.iloc[-1][target],
                f"{target}_lag_2": last.iloc[-2][target],
                f"{target}_lag_3": last.iloc[-3][target],
                "month": ((feats["date"].iloc[-1].month + i - 1) % 12) + 1,
                "trend": trend_start + i
            }

           
            X_pred = pd.DataFrame([row])
            X_pred = X_pred[model.feature_names_in_]

        
            ml_pred = model.predict(X_pred)[0]

# ---- Linear trend slope (last 6 months) ----
            if len(hist) >= 6:
                recent = hist[target].tail(6).values
                slope = (recent[-1] - recent[0]) / 5
            else:
                slope = 0
# ---- Blend (THIS makes it slopey) ----
            pred = ml_pred + slope * 0.6

            new_date = hist["date"].iloc[-1] + pd.offsets.MonthEnd(i)

            hist = pd.concat([
                hist,
                pd.DataFrame([{
                    "date": new_date,
                    "state": state,
                    target: max(pred, 0)
                }])
            ], ignore_index=True)

            feats = create_features(hist, target)

            future.append({
                "date": new_date,
                target: max(pred, 0),
                "model": "ML"
            })

        return pd.DataFrame(future)

    # ---------- BASELINE PATH ----------
    baseline = hist[target].tail(3).mean()
    future = []

    for i in range(months):
        new_date = hist["date"].iloc[-1] + pd.offsets.MonthEnd(i + 1)
        future.append({
            "date": new_date,
            target: round(baseline, 2),
            "model": "Baseline (Moving Avg)"
        })

    return pd.DataFrame(future)


# SIDEBAR CONTROLS

st.sidebar.header("Controls")

dataset = st.sidebar.radio(
    "Select Dataset",
    ["Biometric", "Demographic", "Enrolment"]
)

# LOAD DATASET FILES

if dataset == "Biometric":
    state_df = load_csv("state_map.csv")
    district_df_all = load_csv("district_map_biometric.csv")

    metric_label_map = {
        "Total Biometric Updates": "bio_total",
        "Child Biometric Share": "child_share",
        "Adult Biometric Share": "adult_share"
    }

elif dataset == "Demographic":
    state_df = load_csv("state_map_demographic.csv")
    district_df_all = load_csv("district_map_demographic.csv")

    metric_label_map = {
        "Total Demographic Updates": "demo_total"
    }

else:
    state_df = load_csv("state_map_enrolment.csv")
    district_df_all = load_csv("district_map_enrolment.csv")

    metric_label_map = {
        "Total Enrolments": "enrol_total"
    }

# =============================
# STATE FILTER
# =============================
states = st.sidebar.multiselect(
    "Select State(s)",
    sorted(state_df["state"].unique()),
    sorted(state_df["state"].unique())
)

state_df = state_df[state_df["state"].isin(states)]
district_df_all = district_df_all[district_df_all["state"].isin(states)]

metric_label = st.selectbox("Select Metric", list(metric_label_map.keys()))
metric = metric_label_map[metric_label]

# =============================
# INDIA STATE MAP
# =============================
st.subheader(f"🗺️ State-wise {dataset} Map – India")

with open("india_states.geojson", "r") as f:
    india_geojson = json.load(f)

fig = px.choropleth(
    state_df,
    geojson=india_geojson,
    locations="state",
    featureidkey="properties.NAME_1",
    color=metric,
    color_continuous_scale="YlOrRd"
)

fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, width="stretch")

# =============================
# DISTRICT MAP
# =============================
st.sidebar.subheader("District Drill-down")

selected_state = st.sidebar.selectbox(
    "Select ONE State",
    sorted(district_df_all["state"].unique())
)

district_df = district_df_all[district_df_all["state"] == selected_state].copy()
district_df["district"] = district_df["district"].str.strip().str.title()

with open("india_district.geojson", "r") as f:
    district_geojson = json.load(f)

district_geojson["features"] = [
    f for f in district_geojson["features"]
    if f["properties"].get("NAME_1") == selected_state
]

for f in district_geojson["features"]:
    f["properties"]["NAME_2"] = f["properties"]["NAME_2"].strip().title()

fig = px.choropleth(
    district_df,
    geojson=district_geojson,
    locations="district",
    featureidkey="properties.NAME_2",
    color=metric,
    color_continuous_scale="YlOrRd"
)

fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, width="stretch")

# =============================
# FORECAST SECTION
# =============================
st.divider()
st.subheader("🔮 Future Trend Prediction")

forecast_dataset = st.selectbox(
    "Select Dataset for Forecast",
    ["Biometric", "Demographic", "Enrolment"]
)

forecast_state_name = st.selectbox(
    "Select State",
    sorted(df["state"].unique())
)

months = st.slider("Forecast Horizon (months)", 3, 12, 6)

monthly_df = prepare_monthly(df)

forecast_df = forecast_future(
    monthly_df,
    models,
    forecast_state_name,
    forecast_dataset,
    months
)

if forecast_df is None:
    st.warning("Not enough data to generate forecast.")
else:
    model_used = forecast_df["model"].iloc[0]
    st.info(f"Forecast generated using **{model_used} model**")

    target_map = {
        "Biometric": "bio_total",
        "Demographic": "demo_total",
        "Enrolment": "enrol_total"
    }

    target = target_map[forecast_dataset]

    history = monthly_df[
        monthly_df["state"] == forecast_state_name
    ][["date", target]]

    history["type"] = "Historical"
    forecast_df["type"] = "Forecast"

    plot_df = pd.concat([history, forecast_df])

    fig = px.line(
        plot_df,
        x="date",
        y=target,
        color="type",
        title=f"{forecast_dataset} Forecast – {forecast_state_name}"
    )

    st.plotly_chart(fig, width="stretch")

st.caption("UIDAI Hackathon • Predictive Identity Intelligence Platform")
