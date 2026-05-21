# Utility functions for lightweight data cleaning agent

import re
import logging
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""
    
    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def extract_section(text: str, section_name: str) -> str:
    """
    Extract a named section from markdown response.
    
    Looks for markdown headers like "## SECTION NAME" and extracts
    the content until the next header or end of text.
    
    Parameters
    ----------
    text : str
        The full response text from the LLM.
    section_name : str
        The section header to find (e.g., "DATA QUALITY ANALYSIS").
    
    Returns
    -------
    str
        The content of that section, or empty string if not found.
    """
    pattern = rf"##\s*{re.escape(section_name)}(.*?)(?=##|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    
    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])
    
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    
    # Get shape info
    shape_info = f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    
    # Get duplicate info
    duplicates = df.duplicated().sum()
    duplicate_info = f"Duplicate rows: {duplicates}"
    
    # Get constant column info
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    constant_info = f"Constant columns (all same value): {', '.join(constant_cols) if constant_cols else 'None'}"
    
    summary = f"""
        Dataset Summary:
        ----------------
        {shape_info}
        {duplicate_info}
        
        Column Data Types:
        {column_types}

        Missing Value Percentage (top columns):
        {missing_summary}
        
        {constant_info}"""

    return summary.strip()


def _round_to_original_precision(df_cleaned: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    for col in df_cleaned.select_dtypes(include='number').columns:
        source = df_original[col] if col in df_original.columns else df_cleaned[col]
        decimals = detect_decimal_places(source)
        df_cleaned[col] = df_cleaned[col].round(decimals)
    return df_cleaned


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    
    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key) or {}
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # Execute the LLM-generated code in isolated namespace
    # Note: exec() can be risky - only use with trusted LLM-generated code
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Get the function from executed code
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")
    
    # Run the function and handle errors
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = _round_to_original_precision(result, df)
            result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"
    
    return {result_key: result, error_key: agent_error}


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }


def detect_decimal_places(series: pd.Series) -> int:
    """
    Detect the number of decimal places in a numeric series.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to analyze.
    
    Returns
    -------
    int
        Number of decimal places (0 if all integers, otherwise max decimal places found).
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return 0
    
    max_decimals = 0
    for val in non_null:
        try:
            # Convert to string and count decimal places
            str_val = str(float(val))
            if '.' in str_val:
                decimals = len(str_val.split('.')[-1])
                max_decimals = max(max_decimals, decimals)
        except (ValueError, TypeError):
            continue
    
    return max_decimals
