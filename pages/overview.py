import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Executive Overview", layout="wide")

@st.cache_data
def get_master_data():
    return pd.read_csv('data/processed/master_df.csv')

df = get_master_data()

st.title("📊 Executive Overview & Trend Analytics")

# Unified Sidebar Filters
st.sidebar.header("Filter Data")
selected_state = st.sidebar.selectbox("Select State", ['All'] + sorted(list(df['State'].dropna().unique())))
selected_category = st.sidebar.selectbox("Select Work Category", ['All'] + sorted(list(df['Work category'].dropna().unique())))

filtered_df = df.copy()
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['State'] == selected_state]
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Work category'] == selected_category]

st.markdown(f"Displaying **{len(filtered_df):,}** projects based on selected filters.")

# Row 1 Graphs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Work Categories by Sanction Volume")
    cat_summary = filtered_df.groupby('Work category')['Sanction Amount ( ₹ )'].sum().reset_index()
    cat_summary['Sanction Amount (Lakhs)'] = cat_summary['Sanction Amount ( ₹ )'] / 1e5
    cat_summary = cat_summary.sort_values(by='Sanction Amount (Lakhs)', ascending=True).tail(10)

    fig_cat = px.bar(
        cat_summary, 
        x='Sanction Amount (Lakhs)', 
        y='Work category', 
        orientation='h',
        color='Sanction Amount (Lakhs)',
        color_continuous_scale='Blues',
        hover_data=['Sanction Amount (Lakhs)']
    )
    fig_cat.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cat, width='stretch')

with col2:
    st.subheader("Sanctioned Amount vs Total Disbursed (With Regression)")
    
    clean_scatter = filtered_df[(filtered_df['Sanction Amount ( ₹ )'] > 0) & (filtered_df['total_disbursed'] >= 0)].copy()
    
    fig_reg = px.scatter(
        clean_scatter,
        x='Sanction Amount ( ₹ )',
        y='total_disbursed',
        color='is_high_risk',
        color_discrete_map={0: '#2b5c8f', 1: '#d9534f'},
        hover_data=['work_id', 'State', "Hon'ble Members of Parliament", 'Work category'],
        labels={'is_high_risk': 'High Risk Flag', 'total_disbursed': 'Total Disbursed (₹)', 'Sanction Amount ( ₹ )': 'Sanctioned Amount (₹)'}
    )
    
    # Add OLS Linear Regression Trendline manually
    x_val = clean_scatter['Sanction Amount ( ₹ )']
    y_val = clean_scatter['total_disbursed']
    if len(clean_scatter) > 1 and x_val.nunique() > 1:
        slope, intercept = np.polyfit(x_val, y_val, 1)
        x_trend = np.linspace(x_val.min(), x_val.max(), 100)
        y_trend = slope * x_trend + intercept
        fig_reg.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name='Regression Trend', line=dict(color='black', width=2, dash='dash')))

    fig_reg.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_reg, width='stretch')

# Row 2 Graphs
col3, col4 = st.columns(2)

with col3:
    st.subheader("Average Fund Utilization Ratio by Sector")
    util_cat = filtered_df.groupby('Work category')['fund_utilization_ratio'].mean().reset_index()
    util_cat['Utilization %'] = util_cat['fund_utilization_ratio'] * 100
    fig_util = px.bar(util_cat, x='Work category', y='Utilization %', color='Utilization %', color_continuous_scale='Greens')
    fig_util.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_util, width='stretch')

with col4:
    st.subheader("Project Active Days vs Rec-to-Sanction Lead Time")
    fig_bubble = px.scatter(
        filtered_df,
        x='days_rec_to_sanc',
        y='days_active',
        size='Sanction Amount ( ₹ )',
        color='State',
        hover_data=['work_id', 'Work category', "Hon'ble Members of Parliament"]
    )
    fig_bubble.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_bubble, width='stretch')