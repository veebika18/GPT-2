import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import pdfplumber
import json
import io
import re
import traceback
import autopep8

from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="🧠 AI Data Analyst", layout="wide")
st.title("🧠 AI Data Analyst with RAG, Ollama, and Transformers")

# Initialize LLaMA
llm = Ollama(model="llama3:latest", temperature=0.3)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Define prompt to enforce using `df`
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a Python data analysis expert.

Use the pandas DataFrame named `df` to answer the following question.
Do not use placeholder variables like `data = [...]`.
Avoid mock data and generate only valid executable Python code.

Here is the dataset schema and a sample:
{history}

User query:
{input}

Only output valid Python code using `df`. Do not include markdown or explanations.
"""
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Upload files
uploaded_files = st.file_uploader("Upload CSV, Excel, or PDF files", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)

df_combined = pd.DataFrame()
raw_text = ""

# Read all uploaded files
for uploaded_file in uploaded_files or []:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "csv":
        df = pd.read_csv(uploaded_file, encoding='cp1252', on_bad_lines='skip')
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    elif ext == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() + "\n"

if not df_combined.empty:
    st.success("Files uploaded successfully.")
    st.dataframe(df_combined.head())

# Extract schema
if not df_combined.empty:
    dtypes_info = df_combined.dtypes.apply(str).to_dict()
    schema_text = "\n".join([f"{col}: {dtype}" for col, dtype in dtypes_info.items()])
else:
    schema_text = ""

# Chat input
query = st.text_input("Ask your data question", placeholder="e.g., What is the average sales by product category?")

if query:
    with st.spinner("Generating Python code using LLaMA..."):
        sample_data = df_combined.head(10).to_dict(orient="records") if not df_combined.empty else {}
        context = f"Schema:\n{schema_text}\n\nSample Data:\n{json.dumps(sample_data, indent=2)}"
        full_input = f"{context}\n\nQuestion: {query}"
        llm_response = chain.run(full_input)

    # Extract only Python code
    code_match = re.findall(r"```(?:python)?(.*?)```", llm_response, re.DOTALL)
    raw_code = code_match[0].strip() if code_match else llm_response.strip()

    # Remove any lines that define fake data
    raw_code = re.sub(r"(sample_)?data\s*=\s*\[.*?\]", "# Removed placeholder data", raw_code, flags=re.DOTALL)
    raw_code = re.sub(r"(sample_)?data\s*=.*?df\.head\(\d+\)", "# Removed extra assignment", raw_code)

    # Force use of actual `df` variable
    raw_code = raw_code.replace("data", "df").replace("sample_data", "df")

    # Auto-fix formatting
    formatted_code = autopep8.fix_code(raw_code)

    # Show generated code
    st.markdown("#### 🧠 Generated Python Code:")
    st.code(formatted_code, language="python")

    # Save combined dataset to temp CSV
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_combined.to_csv(temp_csv.name, index=False)

    # Prepare isolated exec environment
    exec_locals = {"pd": pd, "plt": plt}
    exec_locals["df"] = pd.read_csv(temp_csv.name)

    # Execute code
    fig = None
    try:
        with st.spinner("Executing Python code..."):
            exec(formatted_code, exec_locals)

            # Show plot if generated
            if plt.get_fignums():
                fig = plt.gcf()
                st.pyplot(fig)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                    fig.savefig(f.name)
                    st.download_button("📥 Download Plot as PNG", f.read(), file_name="plot.png")

            # Show result
            result = exec_locals.get("result", None)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
                st.download_button("📥 Download Result as CSV", result.to_csv(index=False), "result.csv")
                st.download_button("📥 Download Result as Excel", result.to_excel(index=False), "result.xlsx")
            elif result is not None:
                st.write("Result:")
                st.write(result)

    except Exception:
        st.error("⚠️ Error executing generated Python code:")
        st.text(traceback.format_exc())

# Clean up
plt.close("all")
