import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import io
import base64
import pdfplumber
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import re
import traceback

st.set_page_config(page_title="📊 Data Analyst Chatbot", layout="wide")
st.title("📊 Intelligent Data Analyst using Ollama")

uploaded_file = st.file_uploader("Upload CSV, Excel, or PDF file", type=["csv", "xlsx", "xls", "pdf"])

df = None
if uploaded_file:
    file_type = uploaded_file.type

    try:
        if "csv" in uploaded_file.name:
            df = pd.read_csv(uploaded_file)
        elif "excel" in uploaded_file.name or "spreadsheet" in file_type:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            selected_sheet = st.selectbox("Select a sheet", sheet_names)
            df = pd.read_excel(excel_file, sheet_name=selected_sheet)
        elif "pdf" in uploaded_file.name:
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                st.text_area("Extracted PDF Text", text, height=300)
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(5))

    user_query = st.text_area("💬 Ask a question about the dataset", key="query")

    if st.button("🔍 Analyze with Ollama"):
        with st.spinner("Thinking..."):

            # Prepare context
            col_info = df.dtypes.to_dict()
            col_info_str = "\n".join(f"{k}: {v}" for k, v in col_info.items())
            context = f"The dataset has these columns:\n{col_info_str}"

            # Prompt Template
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are a data analyst. Given the dataset info:\n{context}\n\n"
                    "Generate Python code to answer:\n{question}\n"
                    "Use the dataframe named 'df'. Don't include explanation, only valid Python code."
                )
            )

            memory = ConversationBufferMemory()
            chain = LLMChain(
                llm=Ollama(model="codellama:7b-python", temperature=0),
                prompt=prompt,
                memory=memory
            )

            # Run model
            response = chain.run(context=context, question=user_query)

            # Extract Python code
            python_code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
            python_code = python_code_blocks[0].strip() if python_code_blocks else response.strip()

            st.subheader("🧠 Generated Python Code")
            st.code(python_code, language="python")

            # Run the generated code safely
            try:
                local_vars = {"df": df, "plt": plt, "pd": pd}
                exec(python_code, {}, local_vars)

                # Try to get a result from local vars
                results = [v for k, v in local_vars.items() if isinstance(v, (pd.DataFrame, pd.Series, plt.Figure))]
                for r in results:
                    if isinstance(r, pd.DataFrame) or isinstance(r, pd.Series):
                        st.subheader("📈 Resulting Output")
                        st.dataframe(r)

                        # Download option
                        with st.expander("⬇️ Download Result"):
                            download_type = st.radio("Select format", ["CSV", "Excel"])
                            if st.button("Download Data"):
                                buffer = io.BytesIO()
                                if download_type == "CSV":
                                    csv = r.to_csv(index=False)
                                    st.download_button("Download CSV", csv, file_name="result.csv", mime="text/csv")
                                else:
                                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                                        r.to_excel(writer, index=False)
                                    st.download_button("Download Excel", buffer.getvalue(), file_name="result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    elif isinstance(r, plt.Figure):
                        st.pyplot(r)
                        # Download plot
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                            r.savefig(tmpfile.name)
                            with open(tmpfile.name, "rb") as f:
                                img_bytes = f.read()
                            st.download_button("⬇️ Download Plot as PNG", data=img_bytes, file_name="plot.png", mime="image/png")
            except Exception as e:
                st.error("❌ Error running generated code:")
                st.text(traceback.format_exc())
