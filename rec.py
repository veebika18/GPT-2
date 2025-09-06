import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import io
import pdfplumber
import json
import re
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="🧠 AI Analyst with Ollama", layout="wide")
st.title("🧠 AI Analyst Chatbot (No Timeout 🔄)")

# -- Chat Memory
if 'history' not in st.session_state:
    st.session_state.history = ConversationBufferMemory(return_messages=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# -- Ollama LLM via LangChain (no subprocess)
llm = Ollama(model="gemma3:latest", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a Python data analyst working with a Pandas DataFrame called `df`.

Conversation so far:
{history}

New user question:
{input}

👉 Instructions:
- Use the provided dataframe `df` (do not make up data)
- Use pandas & matplotlib
- Convert date columns using pd.to_datetime() if needed
- Store the final result in variable `result`
- Do not print or explain
- Only return clean executable Python code
"""
)

langchain_chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.history)

# -- Upload & Parse Files
uploaded_files = st.file_uploader("📁 Upload CSV, Excel, PDF (multiple allowed)", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)
all_dfs = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1].lower()
        base_name = uploaded_file.name

        if ext in ["xlsx", "xls"]:
            excel = pd.ExcelFile(uploaded_file)
            for sheet in excel.sheet_names:
                df = pd.read_excel(excel, sheet_name=sheet)
                all_dfs[f"{base_name} - {sheet}"] = df
        elif ext == "csv":
            df = pd.read_csv(uploaded_file)
            all_dfs[base_name] = df
        elif ext == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            df = pd.DataFrame({"pdf_text": text.split("\n")})
            all_dfs[base_name] = df

# -- Show datasets
if all_dfs:
    for name, df in all_dfs.items():
        if st.checkbox(f"👁 Show data - {name}"):
            st.dataframe(df)

# -- Dataset merging
if len(all_dfs) >= 2:
    st.markdown("### 🔗 Merge Datasets")
    keys = list(all_dfs.keys())
    left = st.selectbox("Left dataset", keys, key="merge_left")
    right = st.selectbox("Right dataset", keys, key="merge_right")
    on_col = st.text_input("Column to merge on")
    if st.button("🔄 Merge Now"):
        if on_col in all_dfs[left].columns and on_col in all_dfs[right].columns:
            merged_df = pd.merge(all_dfs[left], all_dfs[right], on=on_col)
            merged_key = f"merged_{left}_{right}"
            all_dfs[merged_key] = merged_df
            st.success(f"Datasets merged as {merged_key}")
            st.dataframe(merged_df)
        else:
            st.error("❌ Merge column not found in both datasets")

# -- Select dataset for query
selected_df_key = st.selectbox("📊 Select dataset for analysis", list(all_dfs.keys()))
df = all_dfs.get(selected_df_key)

user_query = st.text_input("🔍 Ask your data-related question:")

if df is not None and user_query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding='utf-8') as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)

    sample = df.head(3).to_dict(orient="records")
    columns = df.columns.tolist()

    prompt_input = f"""
The CSV file is saved at: {tmp_csv.name}
The DataFrame `df` has columns: {columns}
Sample rows:
{json.dumps(sample, indent=2)}

Question: {user_query}
"""

    if st.button("▶️ Run Code"):
        try:
            with st.spinner("🤖 Thinking... generating Python code..."):
                langchain_response = langchain_chain.run(input=prompt_input)

                code = re.sub(r"(?i)```python|```|>>>", "", langchain_response).strip()
                if "result" not in code:
                    raise Exception("❌ 'result' variable not found in response.")
                
                # Show the code
                st.code(code, language="python")

                # Execute
                local_vars = {"pd": pd, "plt": plt, "df": pd.read_csv(tmp_csv.name)}
                exec(code, {}, local_vars)
                result = local_vars.get("result")

                # Display result
                if isinstance(result, pd.DataFrame):
                    st.success("✅ DataFrame result:")
                    st.dataframe(result)
                    with st.expander("⬇️ Download Result"):
                        fmt = st.selectbox("Choose format", ["CSV", "Excel"])
                        if fmt == "CSV":
                            st.download_button("Download CSV", result.to_csv(index=False), file_name="result.csv")
                        else:
                            buf = io.BytesIO()
                            result.to_excel(buf, index=False, engine="openpyxl")
                            st.download_button("Download Excel", buf.getvalue(), file_name="result.xlsx")
                else:
                    st.success("✅ Output:")
                    st.write(result)

                # Plot if any
                if plt.get_fignums():
                    st.pyplot(plt.gcf())
                    with st.expander("📷 Download Plot"):
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        st.download_button("Download Plot", buf.getvalue(), file_name="plot.png")

                st.session_state.messages.append(("🧑", user_query))
                st.session_state.messages.append(("🤖", code))

        except Exception as e:
            st.error(str(e))
        finally:
            try:
                os.unlink(tmp_csv.name)
            except:
                pass

# -- Chat History
st.sidebar.markdown("## 💬 Chat History")
for speaker, msg in st.session_state.messages:
    st.sidebar.markdown(f"**{speaker}**: {msg}")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history.clear()
    st.session_state.messages.clear()
    st.experimental_rerun()
