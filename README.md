# Intelligent Data Cleaning Agent

An AI-powered data cleaning agent that **analyzes your dataset, reasons about quality issues, and intelligently decides which cleaning steps are appropriate**. Unlike prescriptive tools, this agent adapts to your specific data and explains its decisions.


## How It Works

The agent follows an intelligent reasoning cycle:

1. **Analyze**: Examines your dataset structure and identifies quality issues
   - Missing values (and their patterns)
   - Wrong data types
   - Duplicate rows
   - Constant/low-variance columns
   - Potential outliers

2. **Reason**: Decides which cleaning steps are necessary and appropriate
   - Not all datasets need all steps
   - Thresholds are chosen based on actual data characteristics
   - User instructions can override default behavior

3. **Explain**: Documents the reasoning before cleaning
   - Data Quality Analysis: what problems were found
   - Cleaning Decisions: why each step was chosen
   - Generated Code: implementation of those decisions

4. **Execute**: Runs the generated code to clean your data

5. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

## Key Features

- **Transparent reasoning** — See why the agent made each decision
- **Intelligent adaptation** — Different datasets get different cleaning steps
- **Custom instructions** — Override defaults with your specific goals
- **Self-healing** — Automatically fixes code errors and retries
- **Audit trail** — Full record of analysis, decisions, and code

## Setup

### Prerequisites

- **Python 3.9 to 3.13 inclusive** (3.9, 3.10, 3.11, 3.12, or 3.13)
  - **Note**: Python 3.9.7 is not supported due to a Streamlit compatibility issue
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.9, python3.10, python3.12, etc.

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Streamlit Web Interface (Recommended)

The easiest way to use the agent is through the interactive web interface:

```bash
poetry run streamlit run app.py
```

Then:
1. Upload your CSV file
2. (Optional) Add custom cleaning instructions
3. Click "Analyze & Clean Data"
4. Review the agent's reasoning
5. Download the cleaned dataset

**Example workflow:**
```
Upload: messy_customer_data.csv
↓
Agent analyzes: "Found 42% missing values in 'feedback' column, 
                 customers are either active/inactive (binary), 
                 age has outliers (negative values)"
↓
Agent decides: "Keep feedback despite 42% missing (useful data), 
               fix negative ages (data entry error), 
               remove 'email_invalid' column (all False, no signal)"
↓
You download: cleaned_customer_data.csv
```

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(data_raw=df)

# Access results
analysis = agent.get_data_quality_analysis()
decisions = agent.get_cleaning_decisions()
cleaned_df = agent.get_data_cleaned()

# Review reasoning before proceeding
print("Analysis:", analysis)
print("Decisions:", decisions)

# Save cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: Provide custom instructions**

```python
# Give the agent specific goals
agent.invoke_agent(
    data_raw=df,
    user_instructions="""
    - Keep all customer feedback columns even if sparse
    - Standardize date formats to YYYY-MM-DD
    - Do NOT remove 'experimental' columns (we're testing these)
    """
)
```

## Output: Three Key Artifacts

After cleaning, you get:

### 1. Data Quality Analysis
What problems the agent found:
```
- Column 'age': 8% missing, contains negative values (likely data entry errors)
- Column 'feedback': 42% missing, but contains valuable customer insights
- Column 'is_active': only 2 unique values (True/False), binary feature
- 127 exact duplicate rows detected
```

### 2. Cleaning Decisions
Why each step was chosen:
```
- Removing 'email_invalid' (constant column, all False values)
- Keeping 'feedback' despite 42% missing (valuable for analysis)
- Imputing 'age' with median (more robust to outliers than mean)
- Capping negative ages at 0 (assuming data entry errors)
- Standardizing 'status' to lowercase ('ACTIVE' → 'active')
```

### 3. Cleaned Data
The processed DataFrame, ready for analysis or modeling.

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Main agent orchestration & workflow
│   └── utils.py                # Helper functions (parsing, execution, fixing)
├── app.py                      # Streamlit interface
├── AGENTS.md                   # Agent behavior documentation
├── pyproject.toml              # Dependencies configuration
├── poetry.lock                 # Locked dependency versions
└── README.md                   # This file
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.

## Architecture: Agents vs Workflows

This project demonstrates the distinction between **agents** (which reason and adapt) and **workflows** (which follow fixed paths):

- **Workflow Structure** (`agent.py`): Fixed sequence of nodes (analyze → execute → fix if error)
- **Agent Behavior** (LLM prompt): Reasons about data quality, decides on steps, explains reasoning
- **Result**: Adaptive workflow guided by intelligent agent decision-making

Learn more in `AGENTS.md`.

## Advanced Configuration

### Logging Generated Code

Save the generated cleaning code to a file for inspection:

```python
agent = LightweightDataCleaningAgent(model=llm, log=True, log_path="./my_logs/")
agent.invoke_agent(data_raw=df)
# Generated code saved to ./my_logs/data_cleaner.py
```

### Custom Function Names

Use a different function name if desired:

```python
agent = LightweightDataCleaningAgent(model=llm, function_name="clean_survey_data")
agent.invoke_agent(data_raw=df)
```

### Adjusting Retry Behavior

Control how many times the agent retries if code fails:

```python
agent.invoke_agent(data_raw=df, max_retries=5)  # More aggressive
agent.invoke_agent(data_raw=df, max_retries=1)  # Fail fast
```

## Common Workflows

### Workflow 1: Quick Clean (Default)
```python
agent = LightweightDataCleaningAgent(model=llm)
agent.invoke_agent(data_raw=df)
df_clean = agent.get_data_cleaned()
```

### Workflow 2: Inspect Reasoning
```python
agent.invoke_agent(data_raw=df)
print("Analysis:\n", agent.get_data_quality_analysis())
print("\nDecisions:\n", agent.get_cleaning_decisions())
df_clean = agent.get_data_cleaned()
```

### Workflow 3: Custom Requirements
```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="Preserve all date columns. Convert 'price' to float if not already."
)
df_clean = agent.get_data_cleaned()
```

### Workflow 4: Audit Trail
```python
agent = LightweightDataCleaningAgent(model=llm, log=True)
agent.invoke_agent(data_raw=df)

# Now you have:
# - Analysis & decisions (in console or Streamlit UI)
# - Generated code (in ./logs/data_cleaner.py)
# - Cleaned data (returned)
# - Full audit trail for compliance/review
```

## Troubleshooting

**Q: The agent removes columns I want to keep**
```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="Keep 'feedback' and 'notes' columns even if sparse"
)
```

**Q: The generated code fails with an error**
The agent automatically retries (up to 3 times) to fix errors. If it still fails, check `agent.get_data_quality_analysis()` for data issues.

**Q: How do I understand what the agent decided?**
Always call `agent.get_data_quality_analysis()` and `agent.get_cleaning_decisions()` to see the reasoning.

**Q: Can I see the generated code?**
```python
print(agent.get_data_cleaner_function())
```

## Design Patterns

This project demonstrates key AI engineering patterns:

| Pattern | Purpose | Example |
|---------|---------|---------|
| **Reasoning + Execution** | Agent thinks, then acts | LLM analyzes data, generates code, executes it |
| **Error Recovery** | Self-healing workflows | If code fails, LLM sees error and fixes it |
| **Transparent Decision-Making** | Audit trail for trust | Agent explains reasoning before cleaning |
| **State Management** | Immutable state tracking | LangGraph TypedDict holds analysis, decisions, code, results |

## Next Steps

- Explore `AGENTS.md` to understand the agent's decision-making process
- Try the Streamlit interface with your own datasets
- Extend the agent with custom validation rules
- Integrate into your data pipeline
