import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import joblib
import shap
import matplotlib.pyplot as plt

# Get current theme
current_theme = st._config.get_option("theme.base")  # 'light' or 'dark'

# Set Plotly template based on theme
plotly_template = 'plotly_white' if current_theme == 'light' else 'plotly_dark'

# =============================
# Custom CSS for Government-like UI with Dark Mode Support
# =============================
st.markdown(
    """
    <style>
    /* Base Variables */
    :root {
        --primary-color: #FF9933; /* Saffron */
        --secondary-color: #138808; /* Green */
        --accent-color: #000080; /* Navy Blue */
        --background-color: #FFFFFF; /* White */
        --text-color: #000000; /* Black */
        --font-family: 'Arial', sans-serif;
        --base-font-size: 16px;
    }

    /* Dark Mode Variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #1A1A1A; /* Dark gray */
            --text-color: #E0E0E0; /* Light gray */
            --accent-color: #87CEEB; /* Light blue for accents */
        }
    }

    /* Force colors with !important to override Streamlit themes */
    html, body, .stApp, [class*="css"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        font-family: var(--font-family);
        font-size: var(--base-font-size);
        line-height: 1.6;
    }

    /* Header and Titles */
    .stApp > header {
        background-color: var(--primary-color) !important;
        padding: 15px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stCaption {
        color: var(--secondary-color) !important;
        font-size: 14px;
    }
    p, li, div, .stMarkdown {
        font-size: var(--base-font-size);
        color: var(--text-color) !important;
        line-height: 1.6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F0F8FF !important; /* Light for light mode */
    }
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
            background-color: #2A2A2A !important; /* Dark for dark mode */
        }
    }
    .stSidebar .stRadio > label, .stSidebar .stSelectbox > label, .stSidebar .stMultiselect > label {
        color: var(--accent-color) !important;
        font-weight: bold;
        font-size: 16px;
    }
    .stSidebar .stToggle > label {
        font-size: 16px;
        color: var(--text-color) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
    }

    /* Charts */
    .plotly-chart {
        border: 1px solid #DDD;
        border-radius: 4px;
        background-color: transparent;
    }
    .element-container iframe {
        min-height: 400px;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--primary-color) !important;
        color: white !important;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }

    /* Logo */
    .logo {
        text-align: center;
        margin-bottom: 30px;
    }
    .logo img {
        width: 120px;
        height: auto;
    }
    @media (prefers-color-scheme: dark) {
        .logo img {
            filter: invert(1) brightness(2); /* Invert for visibility in dark mode */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="UIDAI Aadhaar System Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🇮🇳"
)


st.title("UIDAI Aadhaar System Intelligence Dashboard")
st.caption(
    "This dashboard provides a consolidated view of Aadhaar system activity across states and districts, "
    "along with short-term forecasts and analytical interpretations. Powered by official government datasets."
)

# =============================
# Data Loaders
# =============================
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

# =============================
# Anomaly Detection Engine
# =============================
@st.cache_data
def compute_anomalies(df):
    df = df.copy()
    df["total_activity"] = (
        df.get("bio_total", 0)
        + df.get("demo_total", 0)
        + df.get("enrol_total", 0)
    )
    df["z_score"] = (
        df.groupby("state")["total_activity"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    df["is_anomaly"] = df["z_score"].abs() > 4
    df["anomaly_score"] = df["z_score"].abs()
    return df

df_anomaly = compute_anomalies(df)

# =============================
# Dataset-specific anomaly view
# =============================
def get_dataset_anomalies(df_anomaly, dataset):
    if dataset == "Biometric":
        metric_col = "bio_total"
    elif dataset == "Demographic":
        metric_col = "demo_total"
    else:
        metric_col = "enrol_total"
    temp = df_anomaly.copy()
    temp["metric_value"] = temp[metric_col]
    temp["z_score_ds"] = (
        temp.groupby("state")["metric_value"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    temp["is_anomaly_ds"] = temp["z_score_ds"].abs() > 4
    temp["anomaly_score_ds"] = temp["z_score_ds"].abs()
    return temp

models = load_models()

# =============================
# Monthly Aggregation
# =============================
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

# =============================
# Time Series Interpretation Engine
# =============================
def analyze_time_series(ts):
    ts = ts.dropna()
    insights = []
    if len(ts) < 6:
        return ["Insufficient data to derive structural insights from this time series."]
    pct_change = (ts.iloc[-1] - ts.iloc[0]) / max(ts.iloc[0], 1)
    if abs(pct_change) < 0.02:
        insights.append(
            "Overall activity levels are stable over time, indicating steady system operations."
        )
    elif pct_change > 0:
        insights.append(
            "Activity shows a sustained upward trend, suggesting increasing usage or demand."
        )
    else:
        insights.append(
            "A gradual decline in activity is observed, which may indicate system saturation."
        )
    volatility = ts.pct_change().std()
    if volatility > 0.25:
        insights.append(
            "The series exhibits high volatility, indicating episodic surges or operational variability."
        )
    elif volatility < 0.05:
        insights.append(
            "Month-to-month variation is low, reflecting a stable operational environment."
        )
    recent_avg = ts.tail(3).mean()
    past_avg = ts.head(3).mean()
    if recent_avg > 1.2 * past_avg:
        insights.append(
            "Recent activity is significantly higher than historical levels, suggesting a possible regime shift."
        )
    elif recent_avg < 0.8 * past_avg:
        insights.append(
            "Recent activity is lower than earlier periods, indicating normalization after a prior peak."
        )
    return insights

# =============================
# Model Explanation Engine
# =============================
def explain_shap_contributions(explain_df, dataset, state):
    narrative = []
    negatives = explain_df[explain_df["Contribution"] < 0]
    positives = explain_df[explain_df["Contribution"] > 0]
    if negatives.abs().sum().values[0] > positives.sum().values[0]:
        narrative.append(
            f"For {state}, the forecast is primarily constrained by recent historical activity levels "
            "rather than driven by strong growth signals."
        )
    else:
        narrative.append(
            f"For {state}, recent activity patterns provide moderate support to the forecast."
        )
    if any("lag_3" in f for f in negatives.index):
        narrative.append(
            "Lower activity observed approximately three months ago continues to dampen the forecast, "
            "indicating incomplete recovery of recent momentum."
        )
    if "month" in positives.index:
        narrative.append(
            "Seasonal effects associated with the current period provide a modest upward influence."
        )
    if "trend" in negatives.index:
        narrative.append(
            "The long-term trend component slightly reduces expectations, suggesting a mature or stabilized system."
        )
    narrative.append(
        f"Taken together, the model indicates a near-term stabilization of {dataset.lower()} activity in {state}, "
        "with no strong signals of rapid expansion or contraction."
    )
    return narrative

# =============================
# Forecasting Function
# =============================
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
    if state in models.get(dataset, {}):
        model = models[dataset][state]
        feats = create_features(hist, target)
        trend_start = feats["trend"].iloc[-1]
        future = []
        explainer = shap.TreeExplainer(model)
        last_explain = None
        for i in range(1, months + 1):
            last = feats.tail(3)
            row = {
                f"{target}_lag_1": last.iloc[-1][target],
                f"{target}_lag_2": last.iloc[-2][target],
                f"{target}_lag_3": last.iloc[-3][target],
                "month": ((feats["date"].iloc[-1].month + i - 1) % 12) + 1,
                "trend": trend_start + i
            }
            X_pred = pd.DataFrame([row])[model.feature_names_in_]
            ml_pred = model.predict(X_pred)[0]
            if len(hist) >= 6:
                recent = hist[target].tail(6).values
                slope = (recent[-1] - recent[0]) / 5
            else:
                slope = 0
            pred = ml_pred + slope * 0.6
            shap_values = explainer.shap_values(X_pred)
            last_explain = dict(zip(X_pred.columns, shap_values[0]))
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
        df_future = pd.DataFrame(future)
        df_future["explain"] = [None] * (len(df_future) - 1) + [last_explain]
        return df_future
    baseline = hist[target].tail(3).mean()
    future = []
    for i in range(months):
        new_date = hist["date"].iloc[-1] + pd.offsets.MonthEnd(i + 1)
        future.append({
            "date": new_date,
            target: round(baseline, 2),
            "model": "Baseline (Moving Average)"
        })
    return pd.DataFrame(future)

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("---")

dataset = st.sidebar.radio(
    "Select Dataset",
    ["Biometric", "Demographic", "Enrolment"],
    help="Choose the type of Aadhaar activity to analyze."
)

show_anomalies = st.sidebar.toggle(
    "Highlight Anomalies",
    value=False,
    help="Highlights districts or states with unusually high Aadhaar activity spikes."
)

# Load Aggregated Map Data based on dataset
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

states = st.sidebar.multiselect(
    "Select States",
    sorted(state_df["state"].unique()),
    default=sorted(state_df["state"].unique()),
    help="Filter the map to specific states."
)

state_df = state_df[state_df["state"].isin(states)]
district_df_all = district_df_all[district_df_all["state"].isin(states)]

metric_label = st.sidebar.selectbox(
    "Select Metric",
    list(metric_label_map.keys()),
    help="Choose the key metric for visualization."
)
metric = metric_label_map[metric_label]

selected_state = st.sidebar.selectbox(
    "Select State for District Analysis",
    sorted(district_df_all["state"].unique()),
    help="Zoom into districts of a specific state."
)

st.sidebar.markdown("---")
st.sidebar.caption("UIDAI Hackathon – Predictive Identity Intelligence Platform")

# =============================
# Main Content Layout with Columns
# =============================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"State-wise {dataset} Activity Across India")
    with open("india_states.geojson", "r") as f:
        india_geojson = json.load(f)
    state_map_df = state_df.copy()
    if show_anomalies:
        df_ds_anomaly = get_dataset_anomalies(df_anomaly, dataset)
        anomaly_state = (
            df_ds_anomaly[df_ds_anomaly["is_anomaly_ds"]]
            .groupby("state")["anomaly_score_ds"]
            .max()
            .reset_index()
        )
        state_map_df = state_map_df.merge(
            anomaly_state,
            on="state",
            how="left"
        )
        fig = px.choropleth(
            state_map_df,
            geojson=india_geojson,
            locations="state",
            featureidkey="properties.NAME_1",
            color="anomaly_score_ds",
            color_continuous_scale="Reds",
            labels={"anomaly_score_ds": "Anomaly Intensity"},
            template=plotly_template
        )
    else:
        fig = px.choropleth(
            state_map_df,
            geojson=india_geojson,
            locations="state",
            featureidkey="properties.NAME_1",
            color=metric,
            color_continuous_scale="YlOrRd",
            template=plotly_template
        )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar={"title": metric_label},
        font={"size": 14}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    district_df = district_df_all[
        district_df_all["state"] == selected_state
    ].copy()
    district_df["district"] = district_df["district"].str.strip().str.title()
    with open("india_district.geojson", "r") as f:
        district_geojson = json.load(f)
    district_geojson["features"] = [
        f for f in district_geojson["features"]
        if f["properties"].get("NAME_1") == selected_state
    ]
    for f in district_geojson["features"]:
        f["properties"]["NAME_2"] = f["properties"]["NAME_2"].strip().title()

    st.subheader(f"District-wise {dataset} Activity in {selected_state}")
    district_map_df = district_df.copy()
    if show_anomalies:
        df_ds_anomaly = get_dataset_anomalies(df_anomaly, dataset)
        anomaly_district = (
            df_ds_anomaly[
                (df_ds_anomaly["state"] == selected_state) &
                (df_ds_anomaly["is_anomaly_ds"])
            ]
            .groupby("district")["anomaly_score_ds"]
            .max()
            .reset_index()
        )
        district_map_df = district_map_df.merge(
            anomaly_district,
            on="district",
            how="left"
        )
        fig = px.choropleth(
            district_map_df,
            geojson=district_geojson,
            locations="district",
            featureidkey="properties.NAME_2",
            color="anomaly_score_ds",
            color_continuous_scale="Reds",
            labels={"anomaly_score_ds": "Anomaly Intensity"},
            template=plotly_template
        )
    else:
        fig = px.choropleth(
            district_map_df,
            geojson=district_geojson,
            locations="district",
            featureidkey="properties.NAME_2",
            color=metric,
            color_continuous_scale="YlOrRd",
            template=plotly_template
        )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar={"title": metric_label},
        font={"size": 14}
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================
# About Section
# =============================
st.divider()
st.subheader("About the Government Datasets")
st.markdown(
    """
    This analysis is based on three primary datasets published through UIDAI and related government data portals. 
    These datasets capture **monthly Aadhaar activity** across states, districts, and pincodes.
    
    **1. Biometric Dataset**  
    - Records Aadhaar biometric update activity.  
    - Includes counts of biometric updates across age groups.  
    - Used to assess authentication load and update frequency.
    
    **2. Demographic Dataset**  
    - Captures demographic updates such as name, address, date of birth, and gender corrections.  
    - Reflects citizen-driven update behavior over time.
    
    **3. Enrolment Dataset**  
    - Records new Aadhaar enrolments across age groups.  
    - Used to study population onboarding and coverage expansion.
    
    These datasets are independently collected but share common geographic and temporal identifiers, 
    allowing them to be merged into a unified analytical dataset.
    """
)

# =============================
# Data Dictionary
# =============================
st.subheader("Dataset Column Definitions")
st.markdown(
    """
    Below is a description of the key columns used throughout this dashboard.
    
    ### Core Identifiers
    - **date**: Month-end date representing when the activity was recorded.
    - **state**: Indian state or union territory.
    - **district**: District within the state.
    - **pincode**: Postal code representing finer geographic granularity.
    
    ### Biometric Dataset Columns
    - **bio_age_5_17**: Biometric updates performed for individuals aged 5–17.
    - **bio_age_17_**: Biometric updates performed for individuals aged 18 and above.
    - **bio_total**: Total biometric updates in a given month (sum of age groups).
    
    ### Demographic Dataset Columns
    - **demo_age_5_17**: Demographic updates for individuals aged 5–17.
    - **demo_age_17_**: Demographic updates for individuals aged 18 and above.
    - **demo_total**: Total demographic updates in a given month.
    
    ### Enrolment Dataset Columns
    - **age_0_5**: New Aadhaar enrolments for children aged 0–5.
    - **age_5_17**: New Aadhaar enrolments for individuals aged 5–17.
    - **age_18_greater**: New Aadhaar enrolments for adults.
    - **enrol_total**: Total new enrolments in a given month.
    
    ### Derived Analytical Columns
    - **month**: Calendar month extracted from the date (used for seasonality).
    - **trend**: Sequential index representing long-term progression over time.
    - **{metric}_lag_1 / lag_2 / lag_3**: Activity values from the previous 1, 2, and 3 months. 
      These capture short-term momentum in Aadhaar activity.
    
    ### Forecast Output Columns
    - **Historical**: Actual recorded Aadhaar activity.
    - **Forecast**: Expected activity based on recent patterns and trends.
    """
)
st.markdown(
    """
    All forecasts and interpretations in this dashboard are based on **aggregated monthly counts**. 
    The Y-axis in all time-series charts therefore represents the **number of Aadhaar transactions 
    processed in a month**, not percentages or indices.
    """
)

# =============================
# Forecast Section
# =============================
st.divider()
st.subheader("Short-term Activity Forecast")

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
with forecast_col1:
    forecast_dataset = st.selectbox(
        "Dataset for Forecasting",
        ["Biometric", "Demographic", "Enrolment"]
    )
with forecast_col2:
    forecast_state_name = st.selectbox(
        "Select State",
        sorted(df["state"].unique())
    )
with forecast_col3:
    months = st.slider("Forecast Horizon (months)", 3, 12, 6)

monthly_df = prepare_monthly(df)
forecast_df = forecast_future(
    monthly_df,
    models,
    forecast_state_name,
    forecast_dataset,
    months
)

if forecast_df is not None:
    target_map = {
        "Biometric": "bio_total",
        "Demographic": "demo_total",
        "Enrolment": "enrol_total"
    }
    target = target_map[forecast_dataset]
    history = monthly_df[
        monthly_df["state"] == forecast_state_name
    ][["date", target]].copy()
    history["Series"] = "Historical"
    forecast_df["Series"] = "Forecast"
    plot_df = pd.concat([history.rename(columns={target: "value"}), forecast_df.rename(columns={target: "value"})])
    fig = px.line(
        plot_df,
        x="date",
        y="value",
        color="Series",
        title=f"{forecast_dataset} Activity Forecast for {forecast_state_name}",
        labels={"value": "Activity Count"},
        template=plotly_template
    )
    fig.update_layout(
        legend_title_text="Data Type",
        xaxis_title="Date",
        yaxis_title="Monthly Activity",
        font={"family": "Arial", "size": 14}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretation of Observed and Forecasted Trends")
    insights = analyze_time_series(pd.concat([history[target], forecast_df[target]]))
    for insight in insights:
        st.markdown(f"- {insight}")

    if "explain" in forecast_df.columns and forecast_df["explain"].notna().any():
        st.subheader("Factors Influencing the Forecast")
        latest_explain = forecast_df["explain"].dropna().iloc[-1]
        explain_df = (
            pd.DataFrame.from_dict(
                latest_explain,
                orient="index",
                columns=["Contribution"]
            )
            .sort_values("Contribution", key=abs, ascending=False)
        )
        # Set matplotlib style based on theme
        if current_theme == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=explain_df["Contribution"].values,
                base_values=0,
                feature_names=explain_df.index.tolist()
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Narrative Explanation")
        narratives = explain_shap_contributions(
            explain_df,
            dataset=forecast_dataset,
            state=forecast_state_name
        )
        for line in narratives:
            st.markdown(f"- {line}")
else:
    st.warning("Insufficient data to generate a forecast for the selected state.")

# =============================
# Footer
# =============================
st.markdown(
    """
    <div class="footer">
        © 2026 Government of India | UIDAI Aadhaar System Intelligence Dashboard | Data sourced from official APIs | For demonstration purposes only.
    </div>
    """,
    unsafe_allow_html=True
)
