import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import io
import pdfplumber
import json
import re
import ast

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
You are a Python data analyst. Maintain memory of previous context.

Conversation so far:
{history}

New user question:
{input}

👉 Instructions:
- Always convert date columns to datetime using `pd.to_datetime`
- Store final output in variable `result`
- Use pandas & matplotlib
- Output only executable code, no explanation
"""
)
langchain_chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.history)

# ✅ Updated: Function to clean and extract only valid Python code
def clean_llm_code(text: str) -> str:
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    code = "\n".join(code_blocks).strip() if code_blocks else text

    # Remove non-code/explanation lines
    lines = code.splitlines()
    filtered_lines = []
    for line in lines:
        line_strip = line.strip()
        if (
            not line_strip
            or line_strip.startswith("#")
            or re.match(r"^(import|from|df|plt|result|print|for|if|def|class|with|return|pd|[a-zA-Z0-9_]+\s*=)", line_strip)
        ):
            filtered_lines.append(line)

    # Handle incomplete 'for' or 'if' statements
    cleaned = []
    skip_next = False
    for i, line in enumerate(filtered_lines):
        if re.match(r"^\s*(for|if|while|with|def|class).*:\s*$", line.strip()):
            if i + 1 >= len(filtered_lines) or filtered_lines[i + 1].strip() == "":
                continue  # skip incomplete block
        cleaned.append(line)

    code = "\n".join(cleaned)
    code = code.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    code = re.sub(r"\\\s*\n", "", code)

    if BLACK_AVAILABLE:
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception:
            pass

    return code

# 📁 File uploader
uploaded_files = st.file_uploader("📁 Upload datasets", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)
all_dfs = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1].lower()
        key = uploaded_file.name

        if ext in ["xlsx", "xls"]:
            excel_data = pd.ExcelFile(uploaded_file)
            sheet = st.selectbox(f"Sheet in {key}", excel_data.sheet_names)
            df = pd.read_excel(excel_data, sheet_name=sheet)
        elif ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif ext == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            df = pd.DataFrame({"pdf_text": text.split("\n")})
        else:
            st.error(f"❌ Unsupported format: {key}")
            continue

        all_dfs[key] = df
        if st.checkbox(f"👁 View - {key}"):
            st.dataframe(df)

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
                st.dataframe(merged)
            else:
                st.error("❌ Column not found in both")

# 📊 Dataset selector
if all_dfs:
    selected_key = st.selectbox("📊 Select dataset for query", list(all_dfs.keys()))
    df = all_dfs[selected_key]
else:
    df = None

# 🔍 Ask question
query = st.text_input("🔍 Ask a data-related question")

if df is not None and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding='utf-8') as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)

    column_names = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")
    langchain_input = f"\nColumns: {column_names}\nSamples: {sample_data}\nFile path: {tmp_csv.name}"

    try:
        with st.spinner("🧠 Generating code..."):
            response = langchain_chain.run(input=f"{query}\n{langchain_input}")
            code_output = clean_llm_code(response)

            if not code_output or "result" not in code_output:
                st.error("❌ Code does not contain a `result` variable.")
                st.text_area("🔍 Generated Code", value=response, height=300)
            else:
                try:
                    ast.parse(code_output)  # Validate syntax
                    st.code(code_output, language="python")

                    exec_locals = {"pd": pd, "plt": plt, "df": pd.read_csv(tmp_csv.name)}
                    exec(code_output, {}, exec_locals)
                    result = exec_locals.get("result")

                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            st.success("✅ Executed successfully.")
                            st.dataframe(result)
                            with st.expander("⬇️ Download Result"):
                                format = st.selectbox("Format", ["CSV", "Excel"])
                                if format == "CSV":
                                    st.download_button("Download CSV", result.to_csv(index=False), "result.csv")
                                else:
                                    output = io.BytesIO()
                                    result.to_excel(output, index=False, engine="openpyxl")
                                    st.download_button("Download Excel", output.getvalue(), "result.xlsx")
                        else:
                            st.write("Result:", result)

                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        with st.expander("📷 Download Plot"):
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png")
                            st.download_button("Download Plot", buf.getvalue(), "plot.png")

                except SyntaxError as se:
                    st.error(f"❌ Syntax Error: {se}")
                    st.text_area("⚠️ Generated Code (Raw)", value=code_output, height=300)
                except Exception as ex:
                    st.error(f"❌ Runtime Error: {ex}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")

    try:
        os.unlink(tmp_csv.name)
    except:
        pass
