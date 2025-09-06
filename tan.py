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
from datetime import datetime

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

# WhatsApp-like CSS styling
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Chat container */
    .chat-container {
        background: #f0f0f0;
        border-radius: 20px;
        padding: 20px;
        max-height: 70vh;
        overflow-y: auto;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Message bubbles */
    .user-message {
        background: #dcf8c6;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: #ffffff;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .system-message {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 8px auto;
        max-width: 60%;
        text-align: center;
        font-size: 0.9em;
        color: #856404;
    }
    
    /* Message timestamps */
    .message-time {
        font-size: 0.7em;
        color: #999;
        text-align: right;
        margin-top: 4px;
    }
    
    /* Input area styling */
    .input-container {
        background: white;
        border-radius: 25px;
        padding: 8px 20px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
    
    /* Header styling */
    .whatsapp-header {
        background: #075e54;
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 0 0;
        text-align: center;
        margin-bottom: 0;
    }
    
    .header-title {
        font-size: 1.5em;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.9em;
        opacity: 0.8;
        margin: 5px 0 0 0;
    }
    
    /* File upload area */
    .upload-area {
        background: #e8f5e8;
        border: 2px dashed #4caf50;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Dataset info cards */
    .dataset-card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #25d366;
    }
    
    /* Code blocks */
    .code-block {
        background: #2d3748;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        overflow-x: auto;
    }
    
    /* Status indicators */
    .status-success {
        color: #25d366;
        font-weight: bold;
    }
    
    .status-error {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .status-warning {
        color: #f39c12;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: #25d366;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #128c7e;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = ConversationBufferMemory(return_messages=True)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "datasets_loaded" not in st.session_state:
    st.session_state.datasets_loaded = False

# Add initial welcome message
if not st.session_state.chat_messages:
    st.session_state.chat_messages.append({
        "type": "bot",
        "content": "👋 Hi! I'm your AI Data Analyst. Upload your datasets and ask me anything about your data!",
        "timestamp": datetime.now().strftime("%H:%M")
    })

# LangChain + LLM Setup (unchanged)
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

# Helper functions (unchanged)
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

    code = code.replace(""", "\"").replace(""", "\"").replace("'", "'")
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
            fixed_code = fixed_code.replace(""", "\"").replace(""", "\"")
            ast.parse(fixed_code)
            return fixed_code
        except Exception:
            return code

# WhatsApp-style header
st.markdown("""
<div class="whatsapp-header">
    <div class="header-title">🧠 AI Data Analyst</div>
    <div class="header-subtitle">Powered by LangChain & Ollama</div>
</div>
""", unsafe_allow_html=True)

# Chat display function
def display_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.chat_messages:
        if message["type"] == "user":
            st.markdown(f"""
            <div class="user-message">
                {message["content"]}
                <div class="message-time">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
        elif message["type"] == "bot":
            st.markdown(f"""
            <div class="bot-message">
                {message["content"]}
                <div class="message-time">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
        elif message["type"] == "system":
            st.markdown(f"""
            <div class="system-message">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# File upload section with WhatsApp-style design
st.markdown("""
<div class="upload-area">
    <h3>📁 Upload Your Datasets</h3>
    <p>Drag and drop or click to upload CSV, Excel, or PDF files</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True, label_visibility="collapsed")
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
                continue
                
            all_dfs[key] = df
            
            # Add system message for successful upload
            if not st.session_state.datasets_loaded:
                st.session_state.chat_messages.append({
                    "type": "system",
                    "content": f"📊 Dataset '{key}' uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.session_state.datasets_loaded = True
            
            # Dataset card
            st.markdown(f"""
            <div class="dataset-card">
                <h4>📊 {key}</h4>
                <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
                <p><strong>Columns:</strong> {', '.join(list(df.columns)[:5])}{'...' if len(df.columns) > 5 else ''}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.session_state.chat_messages.append({
                "type": "bot",
                "content": f"❌ Error loading {key}: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M")
            })

# Merge datasets section
if len(all_dfs) >= 2:
    with st.expander("🔗 Merge Datasets"):
        keys = list(all_dfs.keys())
        left = st.selectbox("Left dataset", keys)
        right = st.selectbox("Right dataset", keys, index=1)
        on = st.text_input("Merge on column:")
        if st.button("🚀 Merge"):
            if on in all_dfs[left].columns and on in all_dfs[right].columns:
                merged = pd.merge(all_dfs[left], all_dfs[right], on=on)
                all_dfs[f"merged_{left}_{right}"] = merged
                st.session_state.chat_messages.append({
                    "type": "system",
                    "content": f"✅ Successfully merged {left} and {right} on '{on}' column",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()
            else:
                st.session_state.chat_messages.append({
                    "type": "bot",
                    "content": f"❌ Column '{on}' not found in both datasets",
                    "timestamp": datetime.now().strftime("%H:%M")
                })

# Dataset selection
if all_dfs:
    selected_key = st.selectbox("📊 Select dataset for analysis", list(all_dfs.keys()))
    df = all_dfs[selected_key]
else:
    df = None

# Display chat messages
display_chat()

# Chat input with WhatsApp-style design
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Create columns for input and button
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input("💬 Type your message...", placeholder="Ask me about your data...", label_visibility="collapsed")

with col2:
    send_button = st.button("📤", help="Send message")

st.markdown('</div>', unsafe_allow_html=True)

# Process query when button is clicked or Enter is pressed
if (query and send_button) or (query and query != st.session_state.get('last_query', '')):
    st.session_state.last_query = query
    
    # Add user message to chat
    st.session_state.chat_messages.append({
        "type": "user",
        "content": query,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    if df is not None:
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
            # Add thinking message
            thinking_msg = {
                "type": "bot",
                "content": "🧠 Analyzing your data and generating code...",
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.chat_messages.append(thinking_msg)
            
            response = langchain_chain.run(input=full_prompt_input)
            code_output = validate_and_fix_code(clean_llm_code(response))

            # Remove thinking message
            st.session_state.chat_messages.pop()

            # Add code to chat
            st.session_state.chat_messages.append({
                "type": "bot",
                "content": f"📝 **Generated Code:**\n```python\n{code_output}\n```",
                "timestamp": datetime.now().strftime("%H:%M")
            })

            # Check for invalid column names in code
            actual_cols = set(df.columns)
            used_cols = set(re.findall(r"df\[['\"](.*?)['\"]\]", code_output))
            unknown_cols = used_cols - actual_cols
            
            if unknown_cols:
                st.session_state.chat_messages.append({
                    "type": "bot",
                    "content": f"❌ **Error:** Generated code uses invalid column names: {unknown_cols}",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            elif not code_output or "result" not in code_output:
                st.session_state.chat_messages.append({
                    "type": "bot",
                    "content": "❌ **Error:** Generated code does not contain a `result` variable.",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            else:
                exec_globals = {
                    "__builtins__": __builtins__,
                    "pd": pd,
                    "plt": plt,
                    "np": np,
                    "df": df.copy(),
                }
                exec_locals = {}

                try:
                    exec(code_output, exec_globals, exec_locals)
                    result = exec_locals.get("result", exec_globals.get("result"))

                    if result is not None:
                        st.session_state.chat_messages.append({
                            "type": "bot",
                            "content": "✅ **Success!** Code executed successfully.",
                            "timestamp": datetime.now().strftime("%H:%M")
                        })

                        # Display results in chat-friendly format
                        if isinstance(result, pd.DataFrame):
                            result_preview = result.head(10).to_string()
                            st.session_state.chat_messages.append({
                                "type": "bot",
                                "content": f"📊 **DataFrame Result:**\n```\n{result_preview}\n```\nShape: {result.shape}",
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                        elif isinstance(result, (pd.Series, list, tuple, str, int, float, bool)):
                            st.session_state.chat_messages.append({
                                "type": "bot",
                                "content": f"📈 **Result:** {result}",
                                "timestamp": datetime.now().strftime("%H:%M")
                            })

                        if plt.get_fignums():
                            st.session_state.chat_messages.append({
                                "type": "bot",
                                "content": "📊 **Chart generated successfully!** Check below the chat for visualization.",
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                            st.pyplot(plt.gcf())
                            plt.close("all")
                    else:
                        st.session_state.chat_messages.append({
                            "type": "bot",
                            "content": "⚠️ Code ran but no `result` found.",
                            "timestamp": datetime.now().strftime("%H:%M")
                        })

                except Exception as ex:
                    st.session_state.chat_messages.append({
                        "type": "bot",
                        "content": f"❌ **Runtime Error:** {str(ex)}",
                        "timestamp": datetime.now().strftime("%H:%M")
                    })

        except Exception as e:
            st.session_state.chat_messages.append({
                "type": "bot",
                "content": f"❌ **Unexpected Error:** {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M")
            })
    else:
        st.session_state.chat_messages.append({
            "type": "bot",
            "content": "⚠️ Please upload a dataset first before asking questions about your data.",
            "timestamp": datetime.now().strftime("%H:%M")
        })
    
    st.rerun()

# Sidebar with WhatsApp-style tips
with st.sidebar:
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 20px; margin: 10px 0;">
        <h3>💡 Quick Tips</h3>
        <ul style="list-style-type: none; padding: 0;">
            <li>🔍 "Show me top 10 products by revenue"</li>
            <li>📊 "Group by city and get average sales"</li>
            <li>📈 "Plot histogram of repair cost"</li>
            <li>🗓️ "Show trend over time"</li>
            <li>🔢 "Calculate correlation between columns"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        st.markdown(f"""
        <div class="dataset-card">
            <h4>📊 Current Dataset</h4>
            <p><strong>Name:</strong> {selected_key}</p>
            <p><strong>Rows:</strong> {len(df):,}</p>
            <p><strong>Columns:</strong> {len(df.columns)}</p>
        </div>
        """, unsafe_allow_html=True)

# Auto-scroll to bottom of chat
st.markdown("""
<script>
    var chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)