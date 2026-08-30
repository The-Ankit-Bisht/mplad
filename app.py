import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="MPLADS AI Portal", layout="wide", page_icon="🏛️")

@st.cache_data
def get_master_data():
    return pd.read_csv('data/processed/master_df.csv')

@st.cache_resource
def get_models():
    risk_model = joblib.load('models/risk_model.joblib')
    fund_model = joblib.load('models/fund_model.joblib')
    return risk_model, fund_model

try:
    df = get_master_data()
    risk_model, fund_model = get_models()
except Exception:
    st.error("Missing processed data or models! Please run `data_processing.py` and `train_models.py` first.")
    st.stop()

st.title("🏛️ MPLADS AI-Powered Monitoring & Analytics Platform")
st.markdown("Centralized AI intelligence hub for audit risk detection, fund utilization forecasting, and operational tracking.")

# Executive KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sanctioned Works", f"{len(df):,}")
col2.metric("Total Sanctioned Value", f"₹{df['Sanction Amount ( ₹ )'].sum()/1e7:,.2f} Cr")
col3.metric("High-Risk Projects Identified", f"{df['is_high_risk'].sum():,}")
col4.metric("Avg Utilization Rate", f"{df['fund_utilization_ratio'].mean()*100:.1f}%")

st.markdown("---")

# Executive Visualizations
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 State-Wise Sanctioned vs. Disbursed Funds")
    state_summary = df.groupby('State')[['Sanction Amount ( ₹ )', 'total_disbursed']].sum().reset_index()
    state_summary['Sanction (Cr)'] = state_summary['Sanction Amount ( ₹ )'] / 1e7
    state_summary['Disbursed (Cr)'] = state_summary['total_disbursed'] / 1e7
    state_summary = state_summary.sort_values(by='Sanction (Cr)', ascending=False).head(10)

    fig_state = go.Figure()
    fig_state.add_trace(go.Bar(x=state_summary['State'], y=state_summary['Sanction (Cr)'], name='Sanctioned (Cr)', marker_color='#1f77b4'))
    fig_state.add_trace(go.Bar(x=state_summary['State'], y=state_summary['Disbursed (Cr)'], name='Disbursed (Cr)', marker_color='#2ca02c'))
    fig_state.update_layout(barmode='group', height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
    st.plotly_chart(fig_state, width='stretch')

with col_b:
    st.subheader("⚠️ High Risk Breakdown by Sector")
    category_risk = df.groupby('Work category')['is_high_risk'].agg(['count', 'sum']).reset_index()
    category_risk.columns = ['Work category', 'Total Works', 'High Risk Count']
    
    fig_pie = px.pie(
        category_risk, 
        values='High Risk Count', 
        names='Work category', 
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(textinfo='percent+label', hovertemplate="<b>%{label}</b><br>High Risk Works: %{value}<br>Percentage: %{percent}")
    fig_pie.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_pie, width='stretch')