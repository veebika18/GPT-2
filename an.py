# [Unchanged imports]
from pyparsing import col
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import io
import pdfplumber
import json
import re
import ast
import traceback

from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Optional: format code with black or autopep8 if installed
try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import autopep8
    AUTOPEP8_AVAILABLE = True
except ImportError:
    AUTOPEP8_AVAILABLE = False

try:
    import rope
    ROPE_AVAILABLE = True
except ImportError:
    ROPE_AVAILABLE = False

st.set_page_config(page_title="🧠 AI Analyst with Memory, LangChain & SQL", layout="wide")
st.title("🧠 AI Analyst Chatbot with Code Validation")

if "history" not in st.session_state:
    st.session_state.history = ConversationBufferMemory(return_messages=True)

# LangChain + LLM Setup
llm = Ollama(model="llama3:latest")

# STEP 1: Code Generation Prompt
code_generation_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are an expert Python data analyst. Use the conversation history to maintain context.

Previous conversation:
{history}

Current request:
{input}

CRITICAL RULES:
1. Use ONLY real column names in the dataset (case-sensitive!).
2. Write ONLY executable Python code - no explanations, no comments, no markdown.
3. Use ONLY these imports: pandas as pd, matplotlib.pyplot as plt, numpy as np
4. The DataFrame is already loaded as 'df' - do not read files again.
5. Always store your final answer in a variable called 'result'.
6. ALWAYS convert date/time columns to datetime **before** using .dt, resample, or time-based operations:
Use: df['column'] = pd.to_datetime(df['column'], errors='coerce')
7. Use proper Python syntax - no undefined variables.
8. Do NOT make up column names. Use ONLY those listed in the schema. If unsure, use df.columns to inspect.
9. If you need to use a column in a function, ensure it exists in df.columns.
10. If you don't know a column name exactly, stop and use df.columns to check. Never guess.
Only generate Python code now:
"""
)

# STEP 2: Code Validation and Correction Prompt
code_validation_prompt = PromptTemplate(
    input_variables=["original_query", "schema_info", "generated_code", "error_info"],
    template="""
You are a Python code validator and corrector. Your job is to analyze the generated code and fix any errors.

ORIGINAL USER QUERY: {original_query}

DATASET SCHEMA:
{schema_info}

GENERATED CODE TO VALIDATE:
{generated_code}

ERROR INFORMATION (if any):
{error_info}

Your task:
1. Check if the code uses correct column names (case-sensitive)
2. Verify proper data type handling - if doing arithmetic on string columns, convert them first
3. Ensure all variables are defined
4. Fix any syntax errors
5. Make sure the code addresses the original query
6. Ensure 'result' variable is properly set

COMMON FIXES NEEDED:
- For arithmetic operations on string columns: df['col'] = pd.to_numeric(df['col'], errors='coerce')
- For datetime operations: df['col'] = pd.to_datetime(df['col'], errors='coerce')
- For string operations that should be numeric: check data types first

RULES:
- Return ONLY the corrected Python code
- No explanations, comments, or markdown
- Use exact column names from the schema
- If no errors found, return the original code unchanged
- Handle data type conversions properly (datetime, numeric)
- Add data type conversions BEFORE using columns in operations

Corrected code:
"""
)

# Create two separate chains
code_generator = LLMChain(llm=llm, prompt=code_generation_prompt, memory=st.session_state.history)
code_validator = LLMChain(llm=llm, prompt=code_validation_prompt)

def clean_llm_code(text: str) -> str:
    """Clean and extract Python code from LLM response"""
    text = text.strip()
    
    # Extract code blocks first
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    code = "\n".join(code_blocks).strip() if code_blocks else text

    # Remove common prefixes/explanations
    patterns = [
        r"Here's the code:.*?\n", r"The code is:.*?\n",
        r"```python\n", r"```\n", r"Corrected code:.*?\n",
        r"Here's the corrected code:.*?\n", r"The corrected code is:.*?\n",
        r"Fixed code:.*?\n", r"Updated code:.*?\n"
    ]
    for pattern in patterns:
        code = re.sub(pattern, "", code, flags=re.IGNORECASE)

    # Fix smart quotes
    code = code.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
    
    # Extract only lines that look like Python code
    lines = code.split("\n")
    clean_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started and stripped.startswith(("df", "result", "import", "pd", "np", "plt", "#")):
            started = True
        if started:
            # Skip lines that look like explanations
            if not (stripped.startswith(("This", "The", "Here", "Now", "Finally", "Note")) and not stripped.startswith(("df", "result", "import", "pd", "np", "plt"))):
                clean_lines.append(line)

    code = "\n".join(clean_lines)

    # Optional formatting
    if BLACK_AVAILABLE:
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception:
            pass
    elif AUTOPEP8_AVAILABLE:
        try:
            code = autopep8.fix_code(code)
        except Exception:
            pass
    
    return code.strip()

def validate_code_syntax(code: str, df: pd.DataFrame) -> tuple:
    """
    Validate code for syntax errors and column name issues
    Returns: (is_valid, error_message)
    """
    try:
        # Check syntax
        ast.parse(code)
        
        # Check for result variable
        if "result" not in code:
            return False, "Code does not contain 'result' variable"
        
        # Check column names
        actual_cols = set(df.columns)
        used_cols = set(re.findall(r"df\[['\"](.*?)['\"]\]", code))
        unknown_cols = used_cols - actual_cols
        if unknown_cols:
            return False, f"Invalid column names used: {list(unknown_cols)}"
        
        return True, "No errors found"
    
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def execute_with_auto_fixes(code: str, df: pd.DataFrame) -> tuple:
    """
    Execute code with automatic data type fixes
    Returns: (success, result, error_message)
    """
    try:
        df_copy = df.copy()
        
        # Auto-fix datetime usage
        dt_cols = re.findall(r"df\[['\"](.*?)['\"]\]\.dt", code)
        for col in dt_cols:
            if col in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

        # Auto-fix object columns used with arithmetic operations
        arithmetic_patterns = [
            r"df\[['\"](.*?)['\"]\]\s*[-+*/]",  # df['col'] - or + or * or /
            r"[-+*/]\s*df\[['\"](.*?)['\"]\]",  # - df['col'] or + or * or /
            r"df\[['\"](.*?)['\"]\]\.sum\(\)",   # sum()
            r"df\[['\"](.*?)['\"]\]\.mean\(\)",  # mean()
            r"df\[['\"](.*?)['\"]\]\.std\(\)",   # std()
            r"df\[['\"](.*?)['\"]\]\.var\(\)",   # var()
            r"df\[['\"](.*?)['\"]\]\.median\(\)", # median()
            r"df\[['\"](.*?)['\"]\]\.min\(\)",   # min()
            r"df\[['\"](.*?)['\"]\]\.max\(\)",   # max()
        ]
        
        numeric_cols = set()
        for pattern in arithmetic_patterns:
            matches = re.findall(pattern, code)
            numeric_cols.update(matches)
        
        # Convert identified columns to numeric
        for col in numeric_cols:
            if col in df_copy.columns and df_copy[col].dtype == "object":
                # Try to convert to numeric
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Auto-fix groupby operations with numeric aggregations
        groupby_patterns = re.findall(r"\.groupby\(['\"]([^'\"]+)['\"]\)(?:\[['\"]([^'\"]+)['\"]\])?\.(?:sum|mean|std|var|median|min|max)\(\)", code)
        for match in groupby_patterns:
            group_col, agg_col = match
            if agg_col and agg_col in df_copy.columns and df_copy[agg_col].dtype == "object":
                df_copy[agg_col] = pd.to_numeric(df_copy[agg_col], errors='coerce')
        
        # Auto-detect columns that look numeric but are stored as strings
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                # Check if column contains numeric-like strings
                sample = df_copy[col].dropna().astype(str).str.strip()
                if len(sample) > 0:
                    # Check if most values look like numbers (including decimals, negatives)
                    numeric_pattern = r'^-?\d+\.?\d*$'
                    numeric_count = sample.str.match(numeric_pattern).sum()
                    if numeric_count / len(sample) > 0.8:  # 80% numeric-like
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "plt": plt,
            "np": np,
            "df": df_copy,
        }
        exec_locals = {}
        
        exec(code, exec_globals, exec_locals)
        result = exec_locals.get("result", exec_globals.get("result"))
        
        return True, result, None
    
    except Exception as e:
        return False, None, str(e)

# File upload and dataset handling
uploaded_files = st.file_uploader("📁 Upload datasets", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)
all_dfs = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1].lower()
        key = uploaded_file.name
        try:
            if ext in ["xlsx", "xls"]:
                excel_data = pd.ExcelFile(uploaded_file)
                sheet = st.selectbox(f"Sheet in {key}", excel_data.sheet_names, key=f"sheet_{key}")
                df = pd.read_excel(excel_data, sheet_name=sheet)
            elif ext == "csv":
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            elif ext == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "\n".join(
                        page.extract_text() for page in pdf.pages if page.extract_text()
                    )
                df = pd.DataFrame({"pdf_text": text.split("\n")})
            else:
                st.error(f"❌ Unsupported format: {key}")
                continue
            all_dfs[key] = df
            if st.checkbox(f"👁 View - {key}", key=f"view_{key}"):
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape}")
        except Exception as e:
            st.error(f"❌ Error loading {key}: {str(e)}")

# 🔗 Merge datasets
if len(all_dfs) >= 2 and st.checkbox("🔗 Merge Datasets"):
    keys = list(all_dfs.keys())
    left = st.selectbox("Left dataset", keys)
    right = st.selectbox("Right dataset", keys, index=1)
    on = st.text_input("Merge on column:")
    if st.button("🚀 Merge"):
        if on in all_dfs[left].columns and on in all_dfs[right].columns:
            merged = pd.merge(all_dfs[left], all_dfs[right], on=on)
            all_dfs[f"merged_{left}_{right}"] = merged
            st.success("✅ Merged!")
            st.dataframe(merged.head())
        else:
            st.error("❌ Column not found in both datasets.")

# 📊 Select dataset
if all_dfs:
    selected_key = st.selectbox("📊 Select dataset for analysis", list(all_dfs.keys()))
    df = all_dfs[selected_key]

    with st.expander("📋 Dataset Info"):
        st.write("**Columns:**", list(df.columns))
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
else:
    df = None

query = st.text_input("🔍 Ask your data question")

if df is not None and query:
    # Collect exact, case-sensitive schema information
    schema_lines = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).unique()[:3]
        dtype = df[col].dtype
        schema_lines.append(f"- '{col}' (dtype={dtype}, sample={list(sample_values)})")

    schema_context = (
        f"DATASET SCHEMA:\n"
        f"Dataset Name: {selected_key}\n"
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
        f"Columns:\n" + "\n".join(schema_lines)
    )

    # Final prompt with user query + strict schema
    full_prompt_input = (
        f"{query}\n\n"
        f"{schema_context}\n\n"
        f"IMPORTANT: Use exact column names (case-sensitive) and respect data types."
    )

    try:
        # STEP 1: Generate initial code
        with st.spinner("🧠 Step 1: Generating code..."):
            initial_response = code_generator.run(input=full_prompt_input)
            initial_code = clean_llm_code(initial_response)

        st.subheader("Step 1: Initially Generated Code")
        st.code(initial_code, language="python")

        # STEP 2: Validate and correct the code
        with st.spinner("🔍 Step 2: Validating and correcting code..."):
            is_valid, error_msg = validate_code_syntax(initial_code, df)
            
            if not is_valid:
                st.warning(f"⚠️ Issues found: {error_msg}")
                
                # Use LLM to correct the code
                correction_response = code_validator.run(
                    original_query=query,
                    schema_info=schema_context,
                    generated_code=initial_code,
                    error_info=error_msg
                )
                
                # ✅ ENHANCEMENT: Clean the corrected code too!
                final_code = clean_llm_code(correction_response)
                st.subheader("Step 2: Corrected Code")
                st.code(final_code, language="python")
                
                # Validate corrected code
                is_corrected_valid, corrected_error = validate_code_syntax(final_code, df)
                if not is_corrected_valid:
                    st.error(f"❌ Correction failed: {corrected_error}")
                    final_code = initial_code  # Fallback to original
            else:
                st.success("✅ Initial code passed validation!")
                final_code = initial_code

        # STEP 3: Execute the final code
        if final_code:
            with st.spinner("⚡ Executing code..."):
                success, result, exec_error = execute_with_auto_fixes(final_code, df)
                
                if success and result is not None:
                    st.success("✅ Code executed successfully!")
                    
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        with st.expander("⬇️ Download Result"):
                            fmt = st.selectbox("Format", ["CSV", "Excel"])
                            if fmt == "CSV":
                                st.download_button("Download CSV", result.to_csv(index=False), "result.csv")
                            else:
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                                    result.to_excel(writer, index=False)
                                st.download_button("Download Excel", output.getvalue(), "result.xlsx")

                    elif isinstance(result, (pd.Series, list, tuple, str, int, float, bool)):
                        st.write(result)

                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        with st.expander("📷 Download Plot"):
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png", dpi=300)
                            st.download_button("Download Plot", buf.getvalue(), "plot.png", "image/png")
                        plt.close("all")
                else:
                    st.error(f"❌ Execution Error: {exec_error}")
                    st.text_area("🔍 Code that caused error:", final_code)

    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        st.text(traceback.format_exc())

# Sidebar with tips
with st.sidebar:
    st.header("💡 Tips for Better Results")
    st.markdown("""
- "Show me top 10 products by revenue"
- "Group by city and get average sales"
- "Plot histogram of repair cost"
""")
    if df is not None:
        st.header("📊 Current Dataset")
        st.write(f"**Name:** {selected_key}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
    
    st.header("🔄 Process Flow")
    st.markdown("""
    **Step 1:** Generate initial code using Ollama
    
    **Step 2:** Validate & correct code with second LLM call
    
    **Step 3:** Execute the final validated code
    """)