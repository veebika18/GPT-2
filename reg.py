import streamlit as st
import pandas as pd
import re
import ast
import traceback
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Streamlit UI setup
st.set_page_config(page_title="🧠 AI Data Analyst", layout="wide")
st.title("🧠 Dynamic Data Analyst with Mistral + Ollama")

# Step 1: File upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Dataset")
    st.dataframe(df.head())

    # Step 2: Show and extract column names
    all_columns = df.columns.tolist()
    st.write("📊 Columns in dataset:", all_columns)

    # Step 3: User query input
    query = st.text_input("💬 Ask a data analysis question (e.g., Predict cost overrun by pbscode):")

    if query:
        # Step 4: LLM Setup (Mistral via Ollama)
        llm = Ollama(model="mistral")
        template = """
You are a Python data analyst. Generate clean, executable Python code using pandas to answer this question:
Question: {question}

Columns: {columns}

Make sure to assign the output to a variable named `result` or show visual output using matplotlib/seaborn.
Do not include explanations, only give the Python code block.
"""
        prompt = PromptTemplate(input_variables=["question", "columns"], template=template)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

        try:
            # Step 5: Call LLM with question and columns
            response = chain.run({
                "question": query,
                "columns": ", ".join(all_columns)
            })

            # Step 6: Extract only python code block from response
            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            if not code_match:
                st.error("⚠️ Failed to extract code from LLM response.")
                st.text(response)
            else:
                code = code_match.group(1).strip()
                st.subheader("🧠 Generated Python Code")
                st.code(code, language="python")

                # Step 7: Validate code using AST
                try:
                    tree = ast.parse(code)
                    safe = any(
                        isinstance(node, (ast.Assign, ast.Expr))
                        for node in tree.body
                    )
                    if not safe:
                        raise ValueError("No valid Python statements found.")
                except Exception as e:
                    st.error(f"❌ Invalid code: {e}")
                    st.stop()

                # Step 8: Execute safely in local scope
                try:
                    local_vars = {"df": df}
                    exec(code, {}, local_vars)

                    # Step 9: Display result or plots
                    if "result" in local_vars:
                        st.subheader("📊 Result")
                        st.dataframe(local_vars["result"])
                except Exception as exec_err:
                    st.error("❌ Error during code execution:")
                    st.text(traceback.format_exc())
        except Exception as e:
            st.error("❌ Error during code generation or execution:")
            st.text(str(e))
