# Agent Behavior & Decision-Making

This document explains how the Data Cleaning Agent reasons about data and makes decisions.

## Overview: Agent vs Workflow

This project is a **workflow orchestrated by an intelligent agent**:

- **Workflow** (`data_cleaning_agent.py`, LangGraph): Fixed structure with 3 nodes
  - Create cleaning code
  - Execute cleaning code
  - Fix errors if needed

- **Agent** (LLM + prompt): Reasoning entity that analyzes data and decides on actions
  - Analyzes data quality issues
  - Decides which cleaning steps are appropriate
  - Explains its reasoning
  - Generates implementation code

The **workflow provides structure and error handling**. The **agent provides intelligence**.

---

## The Agent's Decision-Making Process

### Step 1: Analyze (What's wrong with this data?)

The agent receives a dataset summary with:
- Column names and data types
- Missing value percentages per column
- Number of rows, columns, duplicates
- Which columns have constant values

Example analysis:
```
Dataset: 1000 rows, 15 columns
- 'customer_id': int64, 0% missing (good, primary key)
- 'feedback': object, 42% missing (sparse, but valuable)
- 'is_active': bool, 0% missing, only 2 unique values (constant)
- 'age': int64, 8% missing, contains values -5 to 200 (suspicious range)
- 'signup_date': object, 0% missing, but in different formats
- 'email_verified': bool, 100% missing (completely empty column)
```

### Step 2: Decide (Which steps should I take?)

The agent reasons:
- **'customer_id'** → Keep (primary key, no issues)
- **'feedback'** → Keep despite 42% missing (rich information in 58% of rows)
- **'is_active'** → Keep (legitimate binary feature)
- **'age'** → Keep but fix (negative values are data entry errors)
- **'signup_date'** → Keep but standardize (parse to datetime)
- **'email_verified'** → Remove (100% missing, no information)

### Step 3: Explain (Why am I doing this?)

The agent documents reasoning:
```
DATA QUALITY ANALYSIS:
- Column 'email_verified' is completely empty (100% missing)
- Column 'feedback' has 42% missing values but contains valuable text
- Column 'age' has invalid values (negative numbers, max 200)
- Column 'is_active' is binary (only True/False values)

CLEANING DECISIONS:
1. Remove 'email_verified' — adds no signal, completely empty
2. Keep 'feedback' — sparse but valuable; will impute nulls with "no_feedback"
3. Fix 'age' — cap negative values at 0, likely data entry errors
4. Impute 'signup_date' — parse to datetime, handle missing dates
5. Standardize 'is_active' — ensure boolean type
```

### Step 4: Generate Code

The agent writes Python code implementing its decisions:
```python
def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(data_raw)
    
    # Remove completely empty column
    df = df.drop(columns=['email_verified'])
    
    # Fix age: cap negative values (data entry errors)
    df['age'] = df['age'].clip(lower=0)
    
    # Parse signup_date to datetime
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    
    # Impute feedback with "no_feedback" for missing values
    df['feedback'] = df['feedback'].fillna('no_feedback')
    
    # Ensure is_active is boolean
    df['is_active'] = df['is_active'].astype(bool)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df
```

### Step 5: Execute & Retry

The workflow:
1. Executes the code
2. If successful → returns cleaned data
3. If error → LLM sees error, regenerates code with fixes, retries

Example failure + fix:
```
FIRST ATTEMPT:
Error: "Cannot convert 'signup_date' to datetime — invalid format"

SECOND ATTEMPT (fixed):
df['signup_date'] = pd.to_datetime(
    df['signup_date'], 
    format='mixed',  # Handle multiple date formats
    errors='coerce'  # Invalid dates become NaT
)

Result: Success ✓
```

---

## Key Decision Principles

### 1. Context-Dependent Thresholds

The agent doesn't use fixed rules. It reasons contextually:

**Fixed Rule (Bad):**
```
IF missing% > 40% THEN remove_column
```

**Intelligent Reasoning (Good):**
```
IF missing% > 40% THEN
    - IF column_name contains ['feedback', 'comment', 'note'] → keep (valuable text)
    - ELIF column_has_high_variance → keep (has information)
    - ELSE → remove (sparse, low information)
```

### 2. Preserve Valuable Information

The agent is biased toward **keeping data**, not removing it:
- Sparse columns with rich information are kept
- Missing values are imputed, not deleted
- Outliers are capped/fixed, not removed (unless clearly errors)

### 3. Handle Edge Cases

The agent documents assumptions:
```
ASSUMPTIONS:
- Negative ages are data entry errors → capped at 0
- 'unknown' and 'N/A' in status column are equivalent → standardized to 'unknown'
- Email addresses with 'test@' are test data → could be removed (but kept for now)
```

### 4. Respect User Intent

User instructions override defaults:
```
agent.invoke_agent(
    data_raw=df,
    user_instructions="""
    - Keep all experimental columns (don't remove even if sparse)
    - Convert price to float, don't remove negative values (they mean refunds)
    - Preserve all date formats (don't standardize)
    """
)
```

The agent respects these instructions in its reasoning.

---

## Transparency: The Three Outputs

### 1. Data Quality Analysis

**What**: Problems found in the data  
**Why**: So users understand what the agent is working with  
**Example**:
```
ISSUES IDENTIFIED:
- 'customer_age': 8% missing, contains values from -5 to 234
- 'order_status': Inconsistent values ('PENDING', 'pending', 'Pending')
- 'feedback': 45% missing but contains qualitative insights
- 'flag_processing_error': 99% identical values (True for all rows)
- 2,341 exact duplicate rows (same values across all columns)
```

### 2. Cleaning Decisions

**What**: Which steps the agent chose and why  
**Why**: So users can verify the reasoning and request changes  
**Example**:
```
DECISIONS:
1. Remove 'flag_processing_error' — no variance, all True (constant column)
2. Remove duplicate rows — 2,341 rows with identical values
3. Keep 'feedback' despite 45% missing — qualitative data is valuable
4. Impute 'customer_age' missing values — use median (robust to outliers)
5. Cap 'customer_age' negative values at 0 — likely data entry errors
6. Standardize 'order_status' — convert all to lowercase ('pending', 'shipped', etc.)
```

### 3. Cleaned Data

**What**: The processed dataset ready for analysis/modeling  
**Why**: The actual output users need  

---

## User Control: Custom Instructions

Users can guide the agent with instructions:

### Example 1: Domain Knowledge
```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="""
    In this domain:
    - Negative prices are refunds, not errors — keep them
    - 'test_user_ID' values are for testing — remove rows
    - Location code '00' means 'unknown' — standardize to 'UNKNOWN'
    """
)
```

### Example 2: Preserve Specific Columns
```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="""
    These columns are important for your use case:
    - comments: Keep even if mostly empty (qualitative feedback)
    - experimental_feature_X: Keep, we're testing this
    - budget_approved: Keep, needed for stakeholder reports
    """
)
```

### Example 3: Aggressive Cleaning
```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="""
    For ML modeling:
    - Remove any column with >10% missing (we need complete data)
    - Remove low-variance columns (variance < 0.001)
    - Convert all text to lowercase
    """
)
```

---

## Error Recovery: Self-Healing

When code generation fails, the agent sees the error and fixes it:

### Example: Type Mismatch Error

**First Attempt:**
```python
df['date'] = pd.to_datetime(df['date'])
# Error: ValueError: mixed dayfirst formats
```

**Agent Reasoning:** "The format parameter wasn't specified. Different rows have different date formats."

**Second Attempt:**
```python
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
# Success: Invalid dates become NaT, valid dates are parsed
```

### Example: Column Not Found

**First Attempt:**
```python
df['age_category'] = pd.cut(df['age'], bins=[0, 18, 65, 120])
# Error: KeyError: 'age' — column doesn't exist
```

**Agent Reasoning:** "I assumed 'age' exists, but it doesn't. The column might be named 'customer_age' or 'age_years'."

**Second Attempt:**
```python
# Check if age-like column exists
age_col = None
for col in df.columns:
    if 'age' in col.lower():
        age_col = col
        break

if age_col:
    df['age_category'] = pd.cut(df[age_col], bins=[0, 18, 65, 120])
else:
    # No age column found, skip categorization
    pass
```

---

## When NOT to Use This Agent

This agent is best for **exploratory data cleaning**. It may not be ideal for:

1. **Highly regulated data** where decisions must follow strict rules
2. **Real-time pipelines** where you need deterministic, not adaptive, behavior
3. **Sensitive data** where you want to control every decision precisely
4. **Domain-specific cleaning** with complex, context-dependent rules

For these, consider a **rule-based workflow** (fixed steps, no reasoning).

---

## Design Philosophy

### Principle 1: Transparency Over Automation

> The agent explains itself, not just executes.

Users see analysis + decisions before data changes.

### Principle 2: Preserve Data, Don't Obliterate It

> The agent errs on the side of keeping data.

- Sparse columns are kept (impute or handle)
- Outliers are capped, not removed
- Duplicates are removed, but only exact matches

### Principle 3: Adaptive, Not Prescriptive

> The agent reasons, not executes instructions.

No hardcoded thresholds ("remove if >40% missing"). Instead: "Is this column worth keeping?"

### Principle 4: Self-Healing

> The agent learns from failures and retries.

If code fails, the agent sees the error and fixes it (up to max retries).

---

## Example: The Complete Cycle

**User Action:**
```python
agent = LightweightDataCleaningAgent(model=llm)
agent.invoke_agent(
    data_raw=messy_df,
    user_instructions="Keep all customer feedback columns"
)
```

**Agent's Analysis:**
```
DATASET SUMMARY:
- 5000 rows, 23 columns
- Missing values: feedback (45%), comments (38%), notes (52%)
- Duplicates: 42 rows
- Data types: mostly object, a few int64, one float64
```

**Agent's Reasoning:**
```
ANALYSIS:
- Three text columns with high missingness (38-52%)
- These are feedback/comment columns → qualitative data
- 42 exact duplicate rows

DECISIONS:
1. Keep all three feedback columns (per user instruction + valuable data)
2. Remove 42 duplicate rows
3. Impute feedback nulls with "no_response" marker
4. Standardize text (trim whitespace, lowercase)
5. Convert all columns to appropriate dtypes
```

**Generated Code:**
```python
def data_cleaner(data_raw):
    df = pd.DataFrame(data_raw)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Impute text columns with "no_response"
    for col in ['feedback', 'comments', 'notes']:
        df[col] = df[col].fillna('no_response')
    
    # Standardize text
    for col in ['feedback', 'comments', 'notes']:
        df[col] = df[col].str.strip().str.lower()
    
    # Standardize dtypes
    df['customer_id'] = df['customer_id'].astype(int)
    df['purchase_amount'] = df['purchase_amount'].astype(float)
    
    return df
```

**Output:**
```
✓ Analysis: [agent's findings]
✓ Decisions: [agent's reasoning]
✓ Cleaned Data: [5000 - 42 = 4958 rows]
```

---

## Questions & Answers

**Q: Why does the agent keep sparse columns?**  
A: Sparse data with rich information (like customer feedback) is more valuable than complete data with no signal. The agent preserves it.

**Q: How does the agent decide between imputing and removing?**  
A: If the column has <50% missing and good variance, impute. If >70% missing AND low information, remove. User instructions override.

**Q: Can the agent handle complex cleaning logic?**  
A: Yes, through user instructions. Example: "If age < 0, treat as refusal to answer (not error) — use separate 'unknown' value."

**Q: What if the agent makes a bad decision?**  
A: That's what user instructions are for. Re-run with updated guidance.

**Q: Is the agent deterministic?**  
A: Mostly yes (same data → same analysis). But the LLM might phrase reasoning differently on different runs.

---

## Further Reading

- `README.md` — Usage and workflows
- `data_cleaning_agent.py` — Architecture and implementation
- `utils.py` — Helper functions
- `app.py` — Streamlit interface example
