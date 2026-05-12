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


def display_data_overview(df, title="Data Overview"):
    """Display detailed statistics for raw or cleaned data."""
    st.subheader(title)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isna().sum().sum())
    with col4:
        st.metric("Duplicates", df.duplicated().sum())
    
    # Detailed column statistics
    st.markdown("#### Column Statistics")
    
    col_stats = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numerical column: min, mean, max
            col_stats.append({
                "Column": col,
                "Type": "Numeric",
                "Missing": f"{df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.1f}%)",
                "Min": f"{df[col].min():.2f}" if df[col].notna().any() else "N/A",
                "Mean": f"{df[col].mean():.2f}" if df[col].notna().any() else "N/A",
                "Max": f"{df[col].max():.2f}" if df[col].notna().any() else "N/A",
                "Unique": df[col].nunique()
            })
        else:
            # Categorical column: show top values with counts
            value_counts = df[col].value_counts()
            top_values = ", ".join([f"{val} ({cnt})" for val, cnt in value_counts.head(3).items()])
            col_stats.append({
                "Column": col,
                "Type": "Categorical",
                "Missing": f"{df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.1f}%)",
                "Distinct": df[col].nunique(),
                "Top Values": top_values if top_values else "N/A",
                "Min": "-",
                "Mean": "-",
                "Max": "-"
            })
    
    stats_df = pd.DataFrame(col_stats)
    # Reorder columns for better display
    display_cols = ["Column", "Type", "Missing", "Distinct", "Top Values", "Min", "Mean", "Max", "Unique"]
    display_cols = [col for col in display_cols if col in stats_df.columns]
    
    st.dataframe(stats_df[display_cols], width='stretch', hide_index=True)
    
    # Show sample data
    st.markdown("#### Sample Data")
    st.dataframe(df.head(10), width='stretch')


def format_analysis_text(text):
    """Format analysis text - add blank line before each numbered section."""
    if not text:
        return text
    
    lines = text.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        # Check if this is a numbered header (e.g., "1.", "2.", "1.1.")
        is_numbered_header = (line.strip() and 
                             line.strip()[0].isdigit() and 
                             '.' in line.strip().split()[0])
        
        # Add blank line before numbered headers (except at start)
        if is_numbered_header and i > 0 and result and result[-1].strip():
            result.append('')
        
        result.append(line)
    
    return '\n'.join(result)


# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df_raw = pd.read_csv(uploaded_file)
    
    # Display raw data overview
    display_data_overview(df_raw, "📊 Raw Data Overview")
    
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
                    st.write(format_analysis_text(analysis))
                else:
                    st.warning("No analysis available")
                
                # Cleaning Decisions
                st.subheader("✅ Cleaning Decisions")
                decisions = agent.get_cleaning_decisions()
                if decisions:
                    st.write(format_analysis_text(decisions))
                else:
                    st.warning("No decisions documented")
                
                # ============ DISPLAY RESULTS ============
                st.divider()
                st.subheader("🎯 Cleaning Results")
                
                df_cleaned = agent.get_data_cleaned()
                
                if df_cleaned is not None:
                    # Show before/after comparison
                    st.markdown("#### Before & After Comparison")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", df_cleaned.shape[0], delta=df_cleaned.shape[0] - df_raw.shape[0])
                    with col2:
                        st.metric("Columns", df_cleaned.shape[1], delta=df_cleaned.shape[1] - df_raw.shape[1])
                    with col3:
                        st.metric("Missing Values", df_cleaned.isna().sum().sum(), 
                                 delta=df_cleaned.isna().sum().sum() - df_raw.isna().sum().sum())
                    with col4:
                        st.metric("Duplicates", df_cleaned.duplicated().sum(),
                                 delta=df_cleaned.duplicated().sum() - df_raw.duplicated().sum())
                    
                    st.divider()
                    
                    # Create two columns for before/after detailed views
                    st.markdown("#### Detailed Statistics Comparison")
                    
                    # Side by side comparison using tabs
                    tab1, tab2 = st.tabs(["✨ After Cleaning", "📊 Before Cleaning"])
                    
                    with tab1:
                        display_data_overview(df_cleaned, "Cleaned Data")
                    
                    with tab2:
                        display_data_overview(df_raw, "Raw Data")
                    
                    st.divider()
                    
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
                
                