"""Streamlit interface for the Intelligent Data Cleaning Agent."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


_RAW_COLOR = "#4C78A8"      # muted blue
_CLEANED_COLOR = "#54A24B"  # muted green
_CHART_CONFIG = {"displayModeBar": False}


def _fmt(name: str) -> str:
    return name.replace("_", " ").title()


def _layout(title="", height=320, **extra):
    return dict(
        template="plotly_white",
        title=dict(text=title, font=dict(size=14, color="#374151"), x=0, xanchor="left"),
        font=dict(family="sans-serif", size=12, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=55, b=40, l=55, r=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
        ),
        **extra,
    )


def display_visualisations(df_raw, df_cleaned=None):
    """Display distribution and quality visualisations, optionally with before/after comparison."""
    comparison = df_cleaned is not None
    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        col for col in df_raw.columns
        if not pd.api.types.is_numeric_dtype(df_raw[col]) and df_raw[col].nunique() <= 20
    ]

    # --- Missing values (horizontal bars, sorted by % missing) ---
    missing_raw = (df_raw.isna().sum() / len(df_raw) * 100).sort_values()
    cols_with_missing = missing_raw[missing_raw > 0]
    if not cols_with_missing.empty:
        st.markdown("##### Missing Values")
        bar_height = max(260, len(cols_with_missing) * 36)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[_fmt(c) for c in cols_with_missing.index],
            x=cols_with_missing.values,
            orientation="h",
            name="Raw",
            marker=dict(color=_RAW_COLOR, opacity=0.85),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        if comparison:
            missing_cleaned = (
                df_cleaned.isna().sum() / len(df_cleaned) * 100
            ).reindex(cols_with_missing.index, fill_value=0)
            fig.add_trace(go.Bar(
                y=[_fmt(c) for c in missing_cleaned.index],
                x=missing_cleaned.values,
                orientation="h",
                name="Cleaned",
                marker=dict(color=_CLEANED_COLOR, opacity=0.85),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
        fig.update_layout(
            **_layout(height=bar_height),
            barmode="group",
            xaxis=dict(title="% Missing", range=[0, 100], ticksuffix="%"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)

    # --- Numeric distributions ---
    if numeric_cols:
        st.markdown("##### Numeric Distributions")
        for i in range(0, len(numeric_cols), 2):
            row_cols = st.columns(2)
            for j, col in enumerate(numeric_cols[i : i + 2]):
                with row_cols[j]:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df_raw[col].dropna(),
                        name="Raw",
                        nbinsx=25,
                        marker=dict(
                            color=_RAW_COLOR, opacity=0.75,
                            line=dict(color="white", width=0.5),
                        ),
                        hovertemplate="Range: %{x}<br>Count: %{y}<extra>Raw</extra>",
                    ))
                    if comparison and col in df_cleaned.columns:
                        fig.add_trace(go.Histogram(
                            x=df_cleaned[col].dropna(),
                            name="Cleaned",
                            nbinsx=25,
                            marker=dict(
                                color=_CLEANED_COLOR, opacity=0.65,
                                line=dict(color="white", width=0.5),
                            ),
                            hovertemplate="Range: %{x}<br>Count: %{y}<extra>Cleaned</extra>",
                        ))
                    fig.update_layout(
                        **_layout(title=_fmt(col), height=300),
                        barmode="overlay",
                        showlegend=comparison,
                        xaxis=dict(title=""),
                        yaxis=dict(title="Count"),
                    )
                    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)

    # --- Categorical value counts ---
    if categorical_cols:
        st.markdown("##### Categorical Distributions")
        for i in range(0, len(categorical_cols), 2):
            row_cols = st.columns(2)
            for j, col in enumerate(categorical_cols[i : i + 2]):
                with row_cols[j]:
                    counts_raw = df_raw[col].value_counts().head(10).sort_values()
                    labels = [str(v) for v in counts_raw.index]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=labels,
                        x=counts_raw.values,
                        orientation="h",
                        name="Raw",
                        marker=dict(color=_RAW_COLOR, opacity=0.85),
                        hovertemplate="%{y}: %{x}<extra>Raw</extra>",
                    ))
                    if comparison and col in df_cleaned.columns:
                        # Normalise to lowercase strings before matching — cleaning agents
                        # commonly strip/lowercase category values, breaking exact reindex.
                        raw_labels_norm = [str(v).strip().lower() for v in counts_raw.index]
                        cleaned_vc_norm = (
                            df_cleaned[col].astype(str).str.strip().str.lower().value_counts()
                        )
                        cleaned_vals = [int(cleaned_vc_norm.get(lbl, 0)) for lbl in raw_labels_norm]
                        fig.add_trace(go.Bar(
                            y=labels,
                            x=cleaned_vals,
                            orientation="h",
                            name="Cleaned",
                            marker=dict(color=_CLEANED_COLOR, opacity=0.85),
                            hovertemplate="%{y}: %{x}<extra>Cleaned</extra>",
                        ))
                    cat_height = max(280, len(labels) * 32 + 80)
                    fig.update_layout(
                        **_layout(title=_fmt(col), height=cat_height),
                        barmode="group",
                        showlegend=comparison,
                        xaxis=dict(title="Count"),
                        yaxis=dict(title=""),
                    )
                    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)

    # --- Correlation heatmap ---
    target_df = df_cleaned if comparison else df_raw
    numeric_target = target_df.select_dtypes(include="number")
    if len(numeric_target.columns) >= 2:
        label = "after cleaning" if comparison else "raw data"
        st.markdown(f"##### Correlation Heatmap ({label})")
        corr = numeric_target.corr()
        n = len(corr)
        cell_px = max(52, min(80, 420 // n))
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_traces(textfont=dict(size=max(9, 13 - n)))
        fig.update_layout(
            **_layout(height=n * cell_px + 80),
            xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
            coloraxis_colorbar=dict(thickness=14, len=0.8),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df_raw = pd.read_csv(uploaded_file)
    
    # Display raw data overview
    display_data_overview(df_raw, "📊 Raw Data Overview")

    with st.expander("📈 Raw Data Visualisations", expanded=False):
        display_visualisations(df_raw)

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
            logger.info("Starting data cleaning process...")
            try:
                logger.info("Initializing LLM model...")
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                agent = LightweightDataCleaningAgent(model=llm, log=True)
                
                logger.info(f"Invoking agent on dataset with shape {df_raw.shape}...")
                agent.invoke_agent(
                    data_raw=df_raw,
                    user_instructions=custom_instructions if custom_instructions else None
                )
                logger.info("Agent processing complete.")
                
                # ============ DISPLAY REASONING ============
                st.success("✅ Analysis complete! Here's what the agent found:")
                st.write("📊 Analyzing data quality and generating cleaning code...")
                
                # Data Quality Analysis
                logger.info("Retrieving data quality analysis...")
                st.subheader("📊 Data Quality Analysis")
                analysis = agent.get_data_quality_analysis()
                if analysis:
                    st.write(format_analysis_text(analysis))
                else:
                    st.warning("No analysis available")
                
                # Cleaning Decisions
                logger.info("Retrieving cleaning decisions...")
                st.subheader("✅ Cleaning Decisions")
                decisions = agent.get_cleaning_decisions()
                if decisions:
                    st.write(format_analysis_text(decisions))
                else:
                    st.warning("No decisions documented")
                
                # ============ DISPLAY RESULTS ============
                logger.info("Retrieving cleaned data...")
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

                    # Before/after visualisations
                    st.markdown("#### Before & After Visualisations")
                    display_visualisations(df_raw, df_cleaned)

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
                    logger.error("Cleaned data is None - cleaning failed")
                    
            except Exception as e:
                logger.error(f"Error during cleaning: {str(e)}", exc_info=True)
                st.error(f"Error during cleaning: {str(e)}")
                st.info("Try adjusting your custom instructions or check the data format.")
