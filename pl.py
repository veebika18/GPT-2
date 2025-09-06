# [Unchanged imports]
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
import datetime  # ✅ Added datetime

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

if "history" not in st.session_state:
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
1. Use ONLY real column names in the dataset (case-sensitive!).
2. Write ONLY executable Python code - no explanations, no comments, no markdown.
3. Use ONLY these imports: pandas as pd, matplotlib.pyplot as plt, numpy as np
4. The DataFrame is already loaded as 'df' - do not read files again.
5. Always store your final answer in a variable called 'result'.
6. For date columns, use: pd.to_datetime(df['column'], errors='coerce')
7. Use proper Python syntax - no undefined variables.
8. Avoid made-up column names or renaming columns.

Only generate Python code now:
"""
)

langchain_chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.history)

def clean_llm_code(text: str) -> str:
    text = text.strip()
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    code = "\n".join(code_blocks).strip() if code_blocks else text

    patterns = [
        r"Here's the code:.*?\n", r"The code is:.*?\n",
        r"```python\n", r"```\n"
    ]
    for pattern in patterns:
        code = re.sub(pattern, "", code, flags=re.IGNORECASE)

    code = code.replace("“", "\"").replace("”", "\"").replace("’", "'")
    lines = code.split("\n")
    clean_lines = []
    started = False
    for line in lines:
        if not started and line.strip().startswith(("df", "result", "import", "pd", "np", "plt")):
            started = True
        if started:
            clean_lines.append(line)
    code = "\n".join(clean_lines)

    if BLACK_AVAILABLE:
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception:
            pass
    return code.strip()

def validate_and_fix_code(code: str) -> str:
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        try:
            fixed_code = re.sub(r"\bPython\b", "python", code)
            fixed_code = fixed_code.replace("“", "\"").replace("”", "\"")
            ast.parse(fixed_code)
            return fixed_code
        except Exception:
            return code

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

    full_prompt_input = (
        f"{query}\n\n"
        f"{schema_context}\n\n"
        f"IMPORTANT: Use exact column names (case-sensitive) and respect data types."
    )

    try:
        with st.spinner("🧠 Generating code..."):
            response = langchain_chain.run(input=full_prompt_input)
            code_output = validate_and_fix_code(clean_llm_code(response))

            st.subheader("Generated Code")
            st.code(code_output, language="python")

            actual_cols = set(df.columns)
            used_cols = set(re.findall(r"df\[['\"](.*?)['\"]\]", code_output))
            unknown_cols = used_cols - actual_cols

            if unknown_cols:
                st.error(f"❌ Generated code uses invalid column names: {unknown_cols}")
                st.text_area("🔍 Raw LLM Response", response)

            elif not code_output or "result" not in code_output:
                st.error("❌ Generated code does not contain a `result` variable.")
                st.text_area("🔍 Raw LLM Response", response)
            else:
                exec_globals = {
                    "__builtins__": __builtins__,
                    "pd": pd,
                    "plt": plt,
                    "np": np,
                    "df": df.copy(),
                    "datetime": datetime  # ✅ Inject datetime into execution scope
                }
                exec_locals = {}

                try:
                    exec(code_output, exec_globals, exec_locals)
                    result = exec_locals.get("result", exec_globals.get("result"))

                    if result is not None:
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
                        st.warning("⚠️ Code ran but no `result` found.")

                except Exception as ex:
                    st.error(f"❌ Runtime Error: {str(ex)}")
                    st.text_area("🔍 Code that caused error:", code_output)
                    st.text(traceback.format_exc())

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
