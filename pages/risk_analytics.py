import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Risk Analytics", layout="wide")

@st.cache_data
def get_master_data():
    return pd.read_csv('data/processed/master_df.csv')

@st.cache_resource
def get_risk_model():
    return joblib.load('models/risk_model.joblib')

df = get_master_data()
model = get_risk_model()

st.title("⚠️ High-Risk Anomaly & Delayed Project Detection")

# Predict Risk Probabilities
numeric_features = ['Sanction Amount ( ₹ )', 'RECOMMENDED AMOUNT ( ₹ )', 'amount_diff', 'days_rec_to_sanc', 'mp_avg_utilization', 'state_delay_rate']
categorical_features = ['State', 'Work category']

X = df[numeric_features + categorical_features]
df['predicted_risk_prob'] = model.predict_proba(X)[:, 1]

# Unified Sidebar Filters
st.sidebar.header("Filter Risk Analysis")
selected_state = st.sidebar.selectbox("Select State", ['All'] + sorted(list(df['State'].dropna().unique())))
selected_category = st.sidebar.selectbox("Select Category", ['All'] + sorted(list(df['Work category'].dropna().unique())))
threshold = st.sidebar.slider("Risk Threshold Sensitivity", 0.1, 0.9, 0.5, 0.05)

filtered_df = df.copy()
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['State'] == selected_state]
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Work category'] == selected_category]

filtered_df['risk_alert'] = filtered_df['predicted_risk_prob'] >= threshold
high_risk_df = filtered_df[filtered_df['risk_alert'] == True]

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Filtered Projects", f"{len(filtered_df):,}")
col_m2.metric("Flagged High Risk", f"{len(high_risk_df):,}")
col_m3.metric("Risk Percentage", f"{(len(high_risk_df)/len(filtered_df)*100 if len(filtered_df)>0 else 0):.1f}%")

st.markdown("---")

# Interactive Charts Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Risk Score Distribution")
    fig_dist = px.histogram(
        filtered_df, 
        x='predicted_risk_prob', 
        nbins=30, 
        color='risk_alert',
        color_discrete_map={False: '#1f77b4', True: '#d9534f'},
        labels={'predicted_risk_prob': 'Risk Probability Score', 'risk_alert': 'High Risk Alert'}
    )
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
    fig_dist.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_dist, width='stretch')

with col2:
    st.subheader("Risk Score vs Fund Utilization")
    fig_scatter = px.scatter(
        filtered_df,
        x='predicted_risk_prob',
        y='fund_utilization_ratio',
        color='risk_alert',
        size='Sanction Amount ( ₹ )',
        hover_data=['work_id', 'State', "Hon'ble Members of Parliament"],
        color_discrete_map={False: '#2ca02c', True: '#d9534f'},
        labels={'predicted_risk_prob': 'Risk Probability Score', 'fund_utilization_ratio': 'Utilization Ratio'}
    )
    fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_scatter, width='stretch')

st.subheader("📋 Actionable High-Risk Audit Cases")
st.dataframe(
    high_risk_df[['work_id', 'State', "Hon'ble Members of Parliament", 'Work category', 'Sanction Amount ( ₹ )', 'predicted_risk_prob']]
    .sort_values(by='predicted_risk_prob', ascending=False),
    width='stretch'
)