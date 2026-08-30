import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="AI Natural Language Assistant", layout="wide")

@st.cache_data
def get_master_data():
    return pd.read_csv('data/processed/master_df.csv')

df = get_master_data()

st.title("🤖 AI-Powered Natural Language Query Assistant (Groq)")
st.markdown("Ask natural language questions to query project records, expenditures, and risk patterns.")

# Sidebar Configuration & Filters
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Select Active Groq Model",
    ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
)

st.sidebar.header("Filter Data Context")
selected_state = st.sidebar.selectbox("Select State", ['All'] + sorted(list(df['State'].dropna().unique())))
filtered_df = df.copy()
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['State'] == selected_state]

# Visual Context Overview
st.subheader("Interactive Data Overview")
fig_quick = px.histogram(
    filtered_df, 
    x='Work category', 
    y='Sanction Amount ( ₹ )', 
    histfunc='sum', 
    color='Work category'
)
fig_quick.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
st.plotly_chart(fig_quick, use_container_width=True)

st.markdown("---")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ `GROQ_API_KEY` not found in your environment. Please add it to your `.env` file in the project root directory:\n\n`GROQ_API_KEY=gsk_your_api_key_here`")
else:
    llm = ChatGroq(
        temperature=0, 
        model_name=selected_model, 
        groq_api_key=api_key
    )

    query = st.text_input("Ask a question about your project data:")

    if query:
        with st.spinner(f"Analyzing dataset and generating report with {selected_model}..."):
            # Construct DataFrame context for the prompt
            data_summary = (
                f"Columns: {list(filtered_df.columns)}\n"
                f"Data types:\n{filtered_df.dtypes.to_string()}\n\n"
                f"Sample rows:\n{filtered_df.head(3).to_string()}"
            )

            prompt = ChatPromptTemplate.from_template(
                """You are an expert Python data analyst working with a pandas DataFrame named `df`.

Dataset Context:
{data_summary}

User Query: {query}

Instructions:
Write Python code using `pandas` and `plotly.express` (as `px`) that assigns three specific variables:
1. `explanation`: A string providing a clear, natural language summary/explanation of the findings and answering the query.
2. `result`: A pandas DataFrame or Series containing the filtered/aggregated records.
3. `fig`: A Plotly figure (`px.bar`, `px.pie`, `px.scatter`, etc.) visualizing `result` if visual representation makes sense; otherwise set `fig = None`.

Return ONLY an executable Python code block inside ```python ... ``` without any surrounding conversational text.
"""
            )

            chain = prompt | llm

            try:
                response = chain.invoke({"data_summary": data_summary, "query": query})
                raw_text = response.content

                # Extract python block from model response
                code_match = re.search(r"```python(.*?)```", raw_text, re.DOTALL)
                code_to_exec = code_match.group(1).strip() if code_match else raw_text.strip()

                # Execute generated code in local namespace
                local_vars = {"df": filtered_df, "pd": pd, "px": px}
                exec(code_to_exec, {}, local_vars)

                explanation = local_vars.get("explanation", None)
                result = local_vars.get("result", None)
                fig = local_vars.get("fig", None)

                st.markdown("## 📋 Query Analysis & Results")

                # 1. Content Explanation
                if explanation:
                    st.markdown("### 📝 Summary & Explanation")
                    st.info(explanation)

                # 2. Graph / Chart Visualization
                if fig is not None:
                    st.markdown("### 📊 Graphical Visualization")
                    st.plotly_chart(fig, use_container_width=True)

                # 3. Data Table
                if result is not None:
                    st.markdown("### 🔢 Detailed Data Table")
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.write(result)

                # Code view fallback
                with st.expander("🔍 View Generated Python Code"):
                    st.code(code_to_exec, language="python")

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")