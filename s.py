import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import io
import traceback

# Streamlit setup
st.set_page_config(page_title="🧠 NLP Data Analyst", layout="wide")
st.title("🧠 ChatGPT-Like Data Analyst with Ollama")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Initialize LLM
llm = Ollama(model="mistral")  # or "gemma", "llama3", etc.

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df.head())

        question = st.text_input("Ask a question about your dataset (e.g., which asset had most repairs)")

        if question:
            # Convert dataset to schema description
            column_names = ", ".join(df.columns)
            sample_rows = df.head(3).to_dict(orient="records")

            # Prompt Template
            template = """
You are a data analyst. Given this dataset with columns: {columns}
and these example rows: {rows}
Write pandas code in Python to answer the following question:
{question}
Only return the code. Do not explain it.
            """

            prompt = PromptTemplate(
                input_variables=["columns", "rows", "question"],
                template=template
            )

            # LangChain Chain
            chain = LLMChain(llm=llm, prompt=prompt)

            # Run Chain
            result = chain.run({
                "columns": column_names,
                "rows": sample_rows,
                "question": question
            })

            st.code(result, language="python")

            # Safe execution
            try:
                local_vars = {"df": df}
                exec(result, {}, local_vars)
                result_val = local_vars.get("result", None)
                if result_val is not None:
                    st.success("✅ Result:")
                    st.write(result_val)
                else:
                    st.warning("⚠️ No variable 'result' returned in code.")
            except Exception as e:
                st.error("❌ Error running the generated code.")
                st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
