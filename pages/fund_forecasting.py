import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fund Utilization Forecasting", layout="wide")

@st.cache_data
def get_master_data():
    return pd.read_csv('data/processed/master_df.csv')

@st.cache_resource
def get_fund_model():
    return joblib.load('models/fund_model.joblib')

df = get_master_data()
model = get_fund_model()

st.title("💰 Fund Utilization Forecasting")

# Predict Fund Ratios
numeric_features = ['Sanction Amount ( ₹ )', 'RECOMMENDED AMOUNT ( ₹ )', 'amount_diff', 'days_rec_to_sanc', 'mp_avg_utilization', 'state_delay_rate']
categorical_features = ['State', 'Work category']

X = df[numeric_features + categorical_features]
df['forecasted_utilization_ratio'] = model.predict(X)
df['forecasted_disbursement'] = df['Sanction Amount ( ₹ )'] * df['forecasted_utilization_ratio']

# Unified Sidebar Filters
st.sidebar.header("Filter Forecast Data")
selected_state = st.sidebar.selectbox("Select State", ['All'] + sorted(list(df['State'].dropna().unique())))
selected_category = st.sidebar.selectbox("Select Category", ['All'] + sorted(list(df['Work category'].dropna().unique())))

filtered_df = df.copy()
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['State'] == selected_state]
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Work category'] == selected_category]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual vs AI Forecasted Disbursement")
    fig_comp = px.scatter(
        filtered_df,
        x='total_disbursed',
        y='forecasted_disbursement',
        color='Work category',
        hover_data=['work_id', 'State', "Hon'ble Members of Parliament"],
        labels={'total_disbursed': 'Actual Disbursed (₹)', 'forecasted_disbursement': 'Forecasted Disbursed (₹)'}
    )
    # 45-degree Reference line
    max_val = max(filtered_df['total_disbursed'].max(), filtered_df['forecasted_disbursement'].max())
    fig_comp.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Perfect 1:1 Match', line=dict(color='red', dash='dash')))
    fig_comp.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_comp, width='stretch')

with col2:
    st.subheader("Forecasted Utilization Ratios by Category")
    fig_box = px.box(
        filtered_df,
        x='Work category',
        y='forecasted_utilization_ratio',
        color='Work category',
        points="outliers"
    )
    fig_box.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_box, width='stretch')

st.subheader("📋 Comprehensive Forecast Ledger")
st.dataframe(
    filtered_df[['work_id', 'State', 'Sanction Amount ( ₹ )', 'total_disbursed', 'forecasted_disbursement', 'forecasted_utilization_ratio']]
    .sort_values(by='forecasted_disbursement', ascending=False),
    width='stretch'
)