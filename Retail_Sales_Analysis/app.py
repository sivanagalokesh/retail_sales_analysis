import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sales Revenue Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD MODEL ARTIFACTS (ROBUST WAY)
# =========================================================
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_path, "model.pkl")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    features_path = os.path.join(base_path, "feature_columns.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(features_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

# =========================================================
# TITLE & DESCRIPTION
# =========================================================
st.title("📊 Sales Revenue Intelligence Dashboard")
st.markdown(
    """
    This dashboard predicts **Sales Revenue** using a trained **Linear / Regularized Regression model**.
    It is designed for **business decision support**, interpretability, and scenario analysis.
    """
)

st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("🔧 Business Inputs")

marketing_spend = st.sidebar.slider(
    "Marketing Spend",
    min_value=0.0,
    max_value=1000.0,
    value=500.0,
    step=25.0
)

store_count = st.sidebar.slider(
    "Store Count",
    min_value=1,
    max_value=500,
    value=100
)

customer_rating = st.sidebar.slider(
    "Customer Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

seasonal_index = st.sidebar.slider(
    "Seasonal Demand Index",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05
)

competitor_price = st.sidebar.slider(
    "Competitor Price Index",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05
)

promotion = st.sidebar.selectbox(
    "Promotion Applied",
    ["No", "Yes"]
)

st.sidebar.divider()
st.sidebar.info(
    "Adjust inputs to simulate business scenarios and observe revenue impact."
)

# =========================================================
# BUILD INPUT DATAFRAME
# =========================================================
input_dict = {
    "MarketingSpend": marketing_spend,
    "StoreCount": store_count,
    "CustomerRating": customer_rating,
    "SeasonalDemandIndex": seasonal_index,
    "CompetitorPrice": competitor_price,
    "IsPromotionApplied_Yes": 1 if promotion == "Yes" else 0
}

input_df = pd.DataFrame([input_dict])

# Add missing one-hot encoded columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[feature_columns]

# Scale features
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]

# =========================================================
# KPI SECTION
# =========================================================
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="📈 Predicted Sales Revenue",
        value=f"{prediction:,.2f}"
    )

with kpi2:
    st.metric(
        label="🏪 Store Count",
        value=store_count
    )

with kpi3:
    st.metric(
        label="⭐ Customer Rating",
        value=customer_rating
    )

st.divider()

# =========================================================
# MODEL INTERPRETATION
# =========================================================
st.subheader("🔍 Key Revenue Drivers")

coef_df = pd.DataFrame({
    "Feature": feature_columns,
    "Impact": model.coef_
}).sort_values(by="Impact", key=abs, ascending=False)

st.write(
    "The table below shows the **relative impact of each feature** on sales revenue. "
    "Positive values increase revenue, while negative values reduce it."
)

st.dataframe(
    coef_df.head(10),
    use_container_width=True
)

st.divider()

# =========================================================
# SCENARIO ANALYSIS
# =========================================================
st.subheader("🔮 Scenario Analysis (What-If Simulation)")

sc1, sc2 = st.columns(2)

with sc1:
    marketing_increase = st.slider(
        "Increase Marketing Spend (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5
    )

with sc2:
    store_increase = st.slider(
        "Increase Store Count (%)",
        min_value=0,
        max_value=30,
        value=5,
        step=5
    )

scenario_df = input_df.copy()
scenario_df["MarketingSpend"] *= (1 + marketing_increase / 100)
scenario_df["StoreCount"] *= (1 + store_increase / 100)

scenario_scaled = scaler.transform(scenario_df)
scenario_prediction = model.predict(scenario_scaled)[0]

delta = scenario_prediction - prediction

st.metric(
    label="📊 Scenario Revenue",
    value=f"{scenario_prediction:,.2f}",
    delta=f"{delta:,.2f}"
)

st.divider()

# =========================================================
# BUSINESS INTERPRETATION
# =========================================================
st.subheader("💡 Business Interpretation")

st.markdown(
    f"""
    - Increasing **marketing spend by {marketing_increase}%** and **store count by {store_increase}%**
      results in an estimated revenue change of **{delta:,.2f}**.
    - Store expansion and customer satisfaction are the strongest revenue drivers.
    - Promotions should be applied strategically, as their impact on net revenue may be limited.
    """
)

st.divider()

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    **Disclaimer:**  
    This model is based on historical data and linear assumptions.
    Predictions should be used for **strategic guidance**, not exact forecasts.
    """
)
