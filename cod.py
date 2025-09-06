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

# Optional: format code with black if installed
try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

st.set_page_config(page_title="🧠 AI Analyst with Memory, LangChain & SQL", layout="wide")
st.title("🧠 AI Analyst Chatbot")

if 'history' not in st.session_state:
    st.session_state.history = ConversationBufferMemory(return_messages=True)

# LangChain + LLM Setup
llm = Ollama(model="llama3:latest")
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are an expert Python data analyst. Use the conversation history to maintain context.

Previous conversation:
{history}

Current request:
{input}

CRITICAL RULES:
1. Write ONLY executable Python code - no explanations, no comments, no markdown
2. Use ONLY these imports: pandas as pd, matplotlib.pyplot as plt, numpy as np
3. The DataFrame is already loaded as 'df' - do not read files again
4. Always store your final answer in a variable called 'result'
5. For date columns, use: pd.to_datetime(df['column'], errors='coerce')
6. Use proper Python syntax - no undefined variables
7. Test your logic before outputting

Example format:
```
# Your analysis code here
result = df.groupby('column').sum()
```

Generate clean, executable Python code only:
"""
)
langchain_chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.history)

def clean_llm_code(text: str) -> str:
    """Extract and clean code from LLM response"""
    # Remove any text before and after code blocks
    text = text.strip()
    
    # Extract code blocks if present
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks).strip()
    else:
        # If no code blocks, take the whole text but remove common LLM phrases
        code = text
        # Remove common LLM response patterns
        patterns_to_remove = [
            r"Here's the code:.*?\n",
            r"Here is the code:.*?\n",
            r"The code is:.*?\n",
            r"```python\n",
            r"```\n",
            r"Here's how.*?:\n",
            r"To.*?, here's the code:\n"
        ]
        for pattern in patterns_to_remove:
            code = re.sub(pattern, "", code, flags=re.IGNORECASE)
    
    # Clean up quotes and common issues
    code = code.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
    
    # Remove any lines that are just explanations or comments at the start
    lines = code.split('\n')
    clean_lines = []
    code_started = False
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and pure comment/explanation lines at the beginning
        if not code_started:
            if (stripped and 
                not stripped.startswith('#') and 
                not stripped.lower().startswith(('here', 'this', 'the code', 'to solve', 'first', 'let me'))):
                code_started = True
                clean_lines.append(line)
            elif stripped.startswith(('import ', 'from ', 'df', 'result', 'plt', 'pd', 'np')):
                code_started = True
                clean_lines.append(line)
        else:
            clean_lines.append(line)
    
    code = '\n'.join(clean_lines)
    
    # Format with black if available
    if BLACK_AVAILABLE:
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception:
            pass
    
    return code.strip()

def validate_and_fix_code(code: str) -> str:
    """Validate and attempt to fix common issues in generated code"""
    try:
        # Try to parse the code
        ast.parse(code)
        return code
    except SyntaxError as e:
        st.warning(f"⚠️ Syntax issue detected: {e}. Attempting to fix...")
        
        # Common fixes
        fixed_code = code
        
        # Fix common variable name issues
        fixed_code = re.sub(r'\bPython\b', 'python', fixed_code)
        fixed_code = re.sub(r'\bDataFrame\b', 'df', fixed_code)
        
        # Fix common quote issues
        fixed_code = fixed_code.replace("'", "'").replace("'", "'")
        fixed_code = fixed_code.replace(""", '"').replace(""", '"')
        
        # Try parsing again
        try:
            ast.parse(fixed_code)
            return fixed_code
        except:
            return code  # Return original if fixes don't work

# 📁 File uploader
uploaded_files = st.file_uploader("📁 Upload datasets", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)
all_dfs = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1].lower()
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
                    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
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
if len(all_dfs) >= 2:
    if st.checkbox("🔗 Merge Datasets"):
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
                st.error("❌ Column not found in both datasets")

# 📊 Dataset selector
if all_dfs:
    selected_key = st.selectbox("📊 Select dataset for query", list(all_dfs.keys()))
    df = all_dfs[selected_key]
    
    with st.expander("📋 Dataset Info"):
        st.write("**Columns:**", list(df.columns))
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
else:
    df = None

# 🔍 Ask question
query = st.text_input("🔍 Ask a data-related question")

if df is not None and query:
    # Prepare data summary for LLM
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().head(3).tolist()
        column_info.append(f"{col} ({dtype}): {sample_values}")
    
    data_summary = f"""
Dataset: {selected_key}
Shape: {df.shape}
Columns: {column_info}
"""

    try:
        with st.spinner("🧠 Generating code..."):
            # Generate code using LangChain
            response = langchain_chain.run(input=f"{query}\n\nData Info: {data_summary}")
            
            # Clean and validate the code
            code_output = clean_llm_code(response)
            code_output = validate_and_fix_code(code_output)
            
            st.subheader("Generated Code:")
            st.code(code_output, language="python")

            if not code_output or "result" not in code_output:
                st.error("❌ Generated code does not contain a `result` variable.")
                st.text_area("🔍 Raw LLM Response", value=response, height=200)
            else:
                try:
                    # Create execution environment
                    exec_globals = {
                        "__builtins__": __builtins__,
                        "pd": pd,
                        "plt": plt,
                        "np": np,
                        "df": df.copy(),  # Work with a copy to avoid modifying original
                    }
                    
                    exec_locals = {}

                    # Execute the code
                    exec(code_output, exec_globals, exec_locals)
                    
                    # Get the result
                    result = exec_locals.get("result")
                    if result is None:
                        result = exec_globals.get("result")

                    if result is not None:
                        st.success("✅ Code executed successfully!")
                        
                        if isinstance(result, pd.DataFrame):
                            st.subheader("📊 Result DataFrame:")
                            st.dataframe(result)
                            
                            with st.expander("⬇️ Download Result"):
                                format_choice = st.selectbox("Format", ["CSV", "Excel"])
                                if format_choice == "CSV":
                                    csv_data = result.to_csv(index=False)
                                    st.download_button("Download CSV", csv_data, "result.csv", "text/csv")
                                else:
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result.to_excel(writer, index=False)
                                    st.download_button("Download Excel", output.getvalue(), "result.xlsx")
                        
                        elif isinstance(result, pd.Series):
                            st.subheader("📈 Result Series:")
                            st.write(result)
                        
                        elif isinstance(result, (int, float, str, bool)):
                            st.subheader("🎯 Result:")
                            st.metric("Answer", result)
                        
                        elif isinstance(result, (list, tuple)):
                            st.subheader("📋 Result:")
                            st.write(result)
                        
                        else:
                            st.subheader("📤 Result:")
                            st.write(result)

                        # Handle matplotlib plots
                        if plt.get_fignums():
                            st.subheader("📊 Visualization:")
                            st.pyplot(plt.gcf())
                            
                            with st.expander("📷 Download Plot"):
                                buf = io.BytesIO()
                                plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                st.download_button("Download Plot", buf.getvalue(), "plot.png", "image/png")
                            plt.close('all')  # Clean up

                    else:
                        st.warning("⚠️ Code executed but no `result` variable was found.")

                except Exception as ex:
                    st.error(f"❌ Runtime Error: {str(ex)}")
                    st.subheader("🔍 Error Details:")
                    st.text(traceback.format_exc())
                    st.text_area("Generated Code (for debugging)", value=code_output, height=300)

    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        st.text(traceback.format_exc())

# Sidebar with tips
with st.sidebar:
    st.header("💡 Tips for Better Results")
    st.markdown("""
    **Good questions:**
    - "Show me the top 10 customers by sales"
    - "Create a bar chart of monthly revenue"
    - "What's the average price by category?"
    - "Find correlations between numeric columns"
    
    **Data operations:**
    - "Group by category and sum sales"
    - "Filter data where price > 100"
    - "Create pivot table of sales by region and month"
    
    **Visualizations:**
    - "Create scatter plot of price vs quantity"
    - "Show distribution of ages as histogram"
    - "Plot time series of daily sales"
    """)
    
    if df is not None:
        st.header("📊 Current Dataset")
        st.write(f"**Name:** {selected_key}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")