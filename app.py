"""Streamlit interface for the Intelligent Data Cleaning Agent."""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

load_dotenv()

st.set_page_config(page_title="Data Cleaning Agent", layout="wide")
st.title("🧹 Intelligent Data Cleaning Agent")

st.markdown("""
This agent analyzes your dataset, identifies quality issues, and intelligently decides 
which cleaning steps are appropriate. It explains its reasoning before cleaning your data.
""")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df_raw = pd.read_csv(uploaded_file)
    
    # Display raw data info
    st.subheader("Raw Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df_raw.shape[0])
    with col2:
        st.metric("Columns", df_raw.shape[1])
    with col3:
        st.metric("Missing Values", df_raw.isna().sum().sum())
    
    st.dataframe(df_raw.head(), use_container_width=True)
    
    # Custom instructions
    with st.expander("📝 Optional: Add custom cleaning instructions"):
        custom_instructions = st.text_area(
            "Tell the agent about specific cleaning goals (e.g., 'Keep all customer feedback columns even if sparse')",
            height=100,
            placeholder="Leave blank for general intelligent cleaning..."
        )
    
    # Clean button
    if st.button("Analyze & Clean Data", type="primary"):
        with st.spinner("Analyzing dataset and generating cleaning code..."):
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                agent = LightweightDataCleaningAgent(model=llm, log=True)
                agent.invoke_agent(
                    data_raw=df_raw,
                    user_instructions=custom_instructions if custom_instructions else None
                )
                
                # ============ DISPLAY REASONING ============
                st.success("Analysis complete! Here's what the agent found:")
                
                # Data Quality Analysis
                st.subheader("📊 Data Quality Analysis")
                analysis = agent.get_data_quality_analysis()
                if analysis:
                    st.info(analysis)
                else:
                    st.warning("No analysis available")
                
                # Cleaning Decisions
                st.subheader("✅ Cleaning Decisions")
                decisions = agent.get_cleaning_decisions()
                if decisions:
                    st.info(decisions)
                else:
                    st.warning("No decisions documented")
                
                # ============ DISPLAY RESULTS ============
                st.divider()
                st.subheader("🎯 Cleaned Data")
                
                df_cleaned = agent.get_data_cleaned()
                
                if df_cleaned is not None:
                    # Show comparison
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df_cleaned.shape[0], delta=df_cleaned.shape[0] - df_raw.shape[0])
                    with col2:
                        st.metric("Columns", df_cleaned.shape[1], delta=df_cleaned.shape[1] - df_raw.shape[1])
                    with col3:
                        st.metric("Missing Values", df_cleaned.isna().sum().sum(), 
                                 delta=df_cleaned.isna().sum().sum() - df_raw.isna().sum().sum())
                    
                    # Display cleaned data
                    st.dataframe(df_cleaned.head(10), use_container_width=True)
                    
                    # Download button
                    csv = df_cleaned.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cleaned Data (CSV)",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Optional: Show the generated code
                    with st.expander("👨‍💻 View Generated Cleaning Code"):
                        code = agent.get_data_cleaner_function()
                        st.code(code, language="python")
                else:
                    st.error("Failed to clean data. Check the generated code above.")
                    
            except Exception as e:
                st.error(f"Error during cleaning: {str(e)}")
                st.info("Try adjusting your custom instructions or check the data format.")
