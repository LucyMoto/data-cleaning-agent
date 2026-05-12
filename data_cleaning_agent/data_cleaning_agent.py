# Libraries
from typing import TypedDict
import os
import logging

import pandas as pd

from langchain.prompts import PromptTemplate
from langgraph.types import Checkpointer
from langgraph.graph import StateGraph, END

from .utils import (
    PythonOutputParser,
    get_dataframe_summary,
    execute_agent_code,
    fix_agent_code,
    extract_section,
)

# Setup
logger = logging.getLogger(__name__)
AGENT_NAME = "lightweight_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class LightweightDataCleaningAgent:
    """
    LLM-powered agent that analyzes datasets and generates intelligent cleaning code.
    
    The agent reasons about data quality issues, decides on appropriate cleaning steps,
    and explains its decisions. Uses an LLM to analyze and clean pandas DataFrames.
    The agent automatically retries with error correction if the generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for analyzing and cleaning data (e.g., ChatOpenAI).
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving agent state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_agent() is called, including reasoning.
    """
    
    def __init__(
        self, 
        model, 
        log: bool = False, 
        log_path: str | None = None, 
        file_name: str = "data_cleaner.py", 
        function_name: str = "data_cleaner",
        checkpointer: Checkpointer | None = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.file_name = file_name
        self.function_name = function_name
        self.checkpointer = checkpointer
        self.response = None
        # Build the LangGraph workflow with code generation, execution, and error fixing nodes
        self._compiled_graph = make_lightweight_data_cleaning_agent(
            model=model,
            log=log,
            log_path=log_path,
            file_name=file_name,
            function_name=function_name,
            checkpointer=checkpointer
        )
    
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str | None = None, max_retries: int = 3, retry_count: int = 0, **kwargs):
        """
        Analyze the dataset and generate cleaning code based on intelligent reasoning.
        
        Parameters
        ----------
        data_raw : pd.DataFrame
            Raw dataset to analyze and clean.
        user_instructions : str, optional
            Custom cleaning goals or dataset-specific instructions. 
            If None, applies general-purpose intelligent cleaning.
        max_retries : int, default=3
            Maximum number of retry attempts if generated code fails.
        retry_count : int, default=0
            Starting retry count (typically left at 0).
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response  # Store full workflow response for getter methods
        return None
    
    def get_data_cleaned(self):
        """
        Retrieves the cleaned data after running invoke_agent.
        
        Returns
        -------
        pd.DataFrame or None
            The cleaned dataset.
        """
        if self.response:
            cleaned = self.response.get("data_cleaned")
            if cleaned:
                return pd.DataFrame(cleaned)
        return None
        
    def get_data_raw(self):
        """
        Retrieves the original raw data.
        
        Returns
        -------
        pd.DataFrame or None
            The raw dataset.
        """
        if self.response:
            raw = self.response.get("data_raw")
            if raw:
                return pd.DataFrame(raw)
        return None
    
    def get_data_cleaner_function(self):
        """
        Retrieves the agent's cleaning function code.
        
        Returns
        -------
        str or None
            The Python code of the generated cleaning function.
        """
        if self.response:
            return self.response.get("data_cleaner_function")
    
    def get_data_quality_analysis(self):
        """
        Retrieves the agent's analysis of data quality issues.
        
        Shows which problems were identified in the dataset (missing values,
        wrong types, duplicates, constants, etc.).
        
        Returns
        -------
        str or None
            The agent's data quality analysis.
        """
        if self.response:
            return self.response.get("data_quality_analysis")
    
    def get_cleaning_decisions(self):
        """
        Retrieves the agent's reasoning about cleaning decisions.
        
        Explains why each cleaning step was chosen and how it will improve data quality.
        
        Returns
        -------
        str or None
            The agent's justification of cleaning decisions.
        """
        if self.response:
            return self.response.get("cleaning_decisions")


# Agent Factory Function

def make_lightweight_data_cleaning_agent(
    model, 
    log: bool = False, 
    log_path: str | None = None, 
    file_name: str = "data_cleaner.py",
    function_name: str = "data_cleaner",
    checkpointer: Checkpointer | None = None
):
    """
    Factory function that creates a compiled LangGraph workflow for intelligent data cleaning.
    
    Builds a state graph where the agent analyzes data quality, reasons about cleaning steps,
    generates code, executes it, and fixes errors if needed.
    
    Parameters
    ----------
    model : LLM
        Language model for data analysis and code generation.
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process cleaning requests.
    """
    # Setup Log Directory
    log_path_resolved: str = log_path if log_path is not None else LOG_PATH
    if log:
        if not os.path.exists(log_path_resolved):
            os.makedirs(log_path_resolved)    

    # Define state schema for the workflow graph
    class GraphState(TypedDict, total=False):
        user_instructions: str | None
        data_raw: dict
        data_cleaned: dict | None
        data_cleaner_function: str
        data_cleaner_function_path: str | None
        data_cleaner_function_name: str
        data_cleaner_error: str | None
        data_quality_analysis: str      # Agent's analysis of quality issues
        cleaning_decisions: str          # Agent's reasoning about decisions
        max_retries: int
        retry_count: int

    
    def create_data_cleaner_code(state: GraphState):
        """
        Analyze the dataset and generate cleaning code with reasoning.
        
        The agent examines data quality issues and decides on appropriate cleaning steps,
        then generates the code to implement those decisions.
        """
        logger.info("Creating data cleaner code with analysis")
        
        data_raw = state.get("data_raw") or {}
        df = pd.DataFrame.from_dict(data_raw)

        dataset_summary = get_dataframe_summary(df)
        
        # Prompt that asks for reasoning + code
        data_cleaning_prompt = PromptTemplate(
            template="""
            You are an intelligent Data Cleaning Agent. Analyze the dataset and decide on 
            appropriate cleaning steps based on data quality issues you observe.

            === YOUR TASK ===
            1. Analyze the data: identify quality issues (missing values, wrong types, 
               duplicates, constants, outliers, etc.)
            2. Decide: determine which cleaning steps are necessary and appropriate
            3. Justify: explain your decisions clearly
            4. Generate: write Python code to implement your decisions

            === ANALYSIS GUIDELINES ===
            - Remove a column only if it has low information (too sparse, constant, or all-null)
            - Impute missing values only if the column is important enough to keep
            - **CRITICAL**: When imputing numeric values, ALWAYS round to match the precision of existing values
            - Handle outliers only if they appear to be errors, not legitimate extremes
            - Standardize formats (dtypes, names, spacing) for consistency
            - Respect the data: don't over-clean or lose important information
            - AVOID chained assignment with inplace=True (df['col'].method(inplace=True) causes FutureWarnings)
            - Use direct assignment instead: df['col'] = df['col'].method() or df = df.method()

            === OUTPUT FORMAT ===
            Return your response in THREE sections (use exact markdown headers):

            ## DATA QUALITY ANALYSIS
            [Describe the issues you found in the dataset]

            ## CLEANING DECISIONS
            [Explain which cleaning steps you chose and why]
            [IMPORTANT: For EVERY numeric column you impute, explicitly state: "Round to X decimal places to match existing values"]

            ## CLEANING CODE
            [Python code in ```python``` blocks below]

            === DATASET SUMMARY ===
            {all_datasets_summary}

            === USER INSTRUCTIONS (custom goals) ===
            {user_instructions}

            === IMPLEMENTATION NOTES ===
            Return Python code with a single function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                # Your cleaning code here
                # Based on analysis and decisions above
                
                # CRITICAL: Avoid chained assignment with inplace=True
                # ❌ DON'T DO THIS: df['col'].fillna(value, inplace=True)
                # ✅ DO THIS INSTEAD: df['col'] = df['col'].fillna(value)
                
                # MANDATORY: ALWAYS round imputed numeric values to match existing precision
                # DO NOT generate long floats like 42.66666666666667
                # INSTEAD: Detect the decimal precision of each column and round to match
                
                # Helper function to detect decimal places
                def detect_decimals(series):
                    non_null = series.dropna()
                    if len(non_null) == 0:
                        return 0
                    max_decimals = 0
                    for val in non_null:
                        try:
                            str_val = str(float(val))
                            if '.' in str_val:
                                decimals = len(str_val.split('.')[-1].rstrip('0'))
                                max_decimals = max(max_decimals, decimals)
                        except (ValueError, TypeError):
                            pass
                    return max_decimals
                
                # Example: For EVERY numeric column that needs imputation:
                #   decimals = detect_decimals(df['salary'])
                #   df['salary'] = df['salary'].fillna(df['salary'].median()).round(decimals)
                # Result: salary values like 45000.00 stay 45000, imputed values like 45000.000001 become 45000
                # Result: salary values like 45000.50 stay 45000.50, imputed values like 45000.666667 become 45000.67
                
                return data_cleaned
            """,
            input_variables=["user_instructions", "all_datasets_summary", "function_name"]
        )

        data_cleaning_agent = data_cleaning_prompt | model
        
        response = data_cleaning_agent.invoke({
            "user_instructions": state.get("user_instructions") or "Apply intelligent cleaning based on data quality analysis.",
            "all_datasets_summary": dataset_summary,
            "function_name": function_name
        })
        
        # Parse response to extract reasoning and code
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        analysis = extract_section(response_text, "DATA QUALITY ANALYSIS")
        decisions = extract_section(response_text, "CLEANING DECISIONS")
        code = PythonOutputParser().parse(response_text)
        
        # Simple logging if enabled
        file_path = None
        if log:
            file_path = os.path.join(log_path_resolved, file_name)
            with open(file_path, 'w') as f:
                f.write(code)
            logger.info(f"Code saved to: {file_path}")
   
        return {
            "data_cleaner_function": code,
            "data_quality_analysis": analysis,      # Store reasoning
            "cleaning_decisions": decisions,         # Store decisions
            "data_cleaner_function_path": file_path,
            "data_cleaner_function_name": function_name,
        }
        
    def execute_data_cleaner_code(state):
        """
        Execute the generated cleaning code on the data.
        """
        return execute_agent_code(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name")
        )
        
    def fix_data_cleaner_code(state: GraphState):
        """
        Fix errors in the generated data cleaning code using the LLM.
        """
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Fix the broken {function_name}() function.
        
        The function had this error:
        {error}

        Analyze the error and provide corrected code that avoids this problem.
        
        Return Python code in ```python``` format with the corrected function definition.
        
        Broken code: 
        {code_snippet}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=model,  
            prompt_template=data_cleaner_prompt,
            function_name=state.get("data_cleaner_function_name"),
        )

    # Build the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    
    # Set entry point
    workflow.set_entry_point("create_data_cleaner_code")
    
    # Add edges
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("fix_data_cleaner_code", "execute_data_cleaner_code")
    
    # Conditional routing: retry with fixes if error occurs and retries remain
    def should_retry(state):
        has_error = state.get("data_cleaner_error") is not None
        can_retry = (
            state.get("retry_count") is not None
            and state.get("max_retries") is not None
            and state["retry_count"] < state["max_retries"]
        )
        return "fix_code" if (has_error and can_retry) else "end"
    
    workflow.add_conditional_edges(
        "execute_data_cleaner_code",
        should_retry,
        {
            "fix_code": "fix_data_cleaner_code",
            "end": END,
        },
    )
    
    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
    
    return app
