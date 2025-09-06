import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import re
import subprocess
import sys
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        margin-right: 20%;
    }
    
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background-color: white;
        padding: 15px;
        border-radius: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 10px 15px;
    }
    
    .download-button {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
    }
    
    .code-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
    }
    
    .error-message {
        color: #dc3545;
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success-message {
        color: #155724;
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalysisChatbot:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3.2:3b"  # You can change this to your preferred model
        self.current_data = None
        self.data_description = ""
        self.execution_globals = {}
        
    def load_dataset(self, uploaded_files) -> Tuple[pd.DataFrame, str]:
        """Load and combine multiple datasets"""
        try:
            all_data = []
            descriptions = []
            
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == 'pdf':
                    st.error("PDF processing requires additional libraries. Please convert to CSV/Excel.")
                    continue
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    continue
                
                all_data.append(df)
                descriptions.append(f"File: {uploaded_file.name}\n{self.describe_dataset(df)}")
            
            if not all_data:
                return None, "No valid datasets loaded"
            
            # Combine all datasets
            combined_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
            combined_description = "\n\n".join(descriptions)
            
            return combined_df, combined_description
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, str(e)
    
    def describe_dataset(self, df: pd.DataFrame) -> str:
        """Generate comprehensive dataset description"""
        description = f"""
Dataset Overview:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Column Information:
"""
        
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100
            
            description += f"\n• {col} ({dtype}): {null_count} nulls ({null_percent:.1f}%)"
            
            if dtype in ['int64', 'float64']:
                description += f" | Range: {df[col].min():.2f} to {df[col].max():.2f}"
            elif dtype == 'object':
                unique_count = df[col].nunique()
                description += f" | {unique_count} unique values"
                if unique_count <= 10:
                    description += f" | Values: {list(df[col].unique())}"
        
        # Statistical summary
        description += f"\n\nStatistical Summary:\n{df.describe()}"
        
        return description
    
    def query_ollama(self, prompt: str) -> str:
        """Query Ollama API for code generation"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(self.ollama_url, json=payload)
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def extract_python_code(self, text: str) -> List[str]:
        """Extract Python code blocks from text"""
        # Pattern to match code blocks
        code_patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'<code>(.*?)</code>',
            r'`([^`]*)`'
        ]
        
        code_blocks = []
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if any(keyword in match for keyword in ['import', 'def', 'print', 'df', 'plt', 'sns']):
                    code_blocks.append(match.strip())
        
        # If no code blocks found, try to extract lines that look like code
        if not code_blocks:
            lines = text.split('\n')
            current_block = []
            
            for line in lines:
                line = line.strip()
                if (line.startswith(('import ', 'from ', 'df.', 'plt.', 'sns.', 'print(', 'fig', '=')) or
                    '=' in line or line.startswith('#')):
                    current_block.append(line)
                elif current_block and line == '':
                    continue
                elif current_block:
                    if len(current_block) > 0:
                        code_blocks.append('\n'.join(current_block))
                    current_block = []
            
            if current_block:
                code_blocks.append('\n'.join(current_block))
        
        return code_blocks
    
    def install_missing_packages(self, error_message: str) -> bool:
        """Automatically install missing packages"""
        try:
            # Common package mappings
            package_mappings = {
                'sklearn': 'scikit-learn',
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'bs4': 'beautifulsoup4'
            }
            
            # Extract package name from error
            import_patterns = [
                r"No module named '([^']*)'",
                r"ModuleNotFoundError: No module named '([^']*)'",
                r"ImportError: No module named ([^\s]*)"
            ]
            
            for pattern in import_patterns:
                match = re.search(pattern, error_message)
                if match:
                    package = match.group(1)
                    package = package_mappings.get(package, package)
                    
                    st.info(f"Installing missing package: {package}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    st.success(f"Successfully installed {package}")
                    return True
            
            return False
            
        except Exception as e:
            st.error(f"Failed to install package: {str(e)}")
            return False
    
    def execute_code(self, code: str, max_retries: int = 3) -> Tuple[bool, str, Any]:
        """Execute Python code with error handling and auto-installation"""
        for attempt in range(max_retries):
            try:
                # Setup execution environment
                exec_globals = {
                    'df': self.current_data,
                    'pd': pd,
                    'np': np,
                    'plt': plt,
                    'sns': sns,
                    'px': px,
                    'go': go,
                    'make_subplots': make_subplots,
                    'st': st,
                    '__builtins__': __builtins__
                }
                exec_globals.update(self.execution_globals)
                
                # Create a StringIO object to capture output
                output_buffer = io.StringIO()
                
                # Execute the code
                exec(code, exec_globals)
                
                # Update stored globals
                self.execution_globals.update(exec_globals)
                
                # Get any printed output
                output = output_buffer.getvalue()
                
                return True, output, exec_globals
                
            except ImportError as e:
                error_msg = str(e)
                if attempt < max_retries - 1 and self.install_missing_packages(error_msg):
                    continue
                else:
                    return False, f"Import Error: {error_msg}", None
                    
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    # Try to get corrected code from Ollama
                    correction_prompt = f"""
                    The following Python code produced an error:
                    
                    Code:
                    {code}
                    
                    Error:
                    {error_msg}
                    
                    Please provide a corrected version of this code that fixes the error.
                    Make sure to only return the corrected Python code without any explanations.
                    
                    Dataset columns: {list(self.current_data.columns) if self.current_data is not None else 'No data loaded'}
                    """
                    
                    corrected_response = self.query_ollama(correction_prompt)
                    corrected_codes = self.extract_python_code(corrected_response)
                    
                    if corrected_codes:
                        code = corrected_codes[0]
                        continue
                
                return False, f"Execution Error: {error_msg}", None
        
        return False, "Maximum retry attempts reached", None
    
    def create_download_link(self, df: pd.DataFrame, filename: str, file_format: str = 'csv') -> str:
        """Create download link for dataframe"""
        if file_format == 'csv':
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">Download CSV</a>'
        elif file_format == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False)
            output.seek(0)
            b64 = base64.b64encode(output.read()).decode()
            return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" class="download-button">Download Excel</a>'
    
    def create_plot_download_link(self, fig, filename: str) -> str:
        """Create download link for plots"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        b64 = base64.b64encode(img_buffer.read()).decode()
        return f'<a href="data:image/png;base64,{b64}" download="{filename}.png" class="download-button">Download Plot</a>'

def main():
    st.title("📊 Data Analysis Chatbot")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DataAnalysisChatbot()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize show_code preference
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Dataset(s)",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload CSV or Excel files"
        )
        
        # Code visibility toggle
        st.session_state.show_code = st.checkbox("Show Generated Code", value=st.session_state.show_code)
        
        # Ollama settings
        st.subheader("🤖 Ollama Settings")
        ollama_model = st.selectbox(
            "Model",
            ["llama3.2:3b", "llama3.1:8b", "codellama:7b", "mistral:7b"],
            help="Select Ollama model for code generation"
        )
        st.session_state.chatbot.model_name = ollama_model
        
        # Dataset info
        if st.session_state.chatbot.current_data is not None:
            st.subheader("📋 Dataset Info")
            st.write(f"Shape: {st.session_state.chatbot.current_data.shape}")
            st.write(f"Columns: {len(st.session_state.chatbot.current_data.columns)}")
            
            if st.button("Clear Dataset"):
                st.session_state.chatbot.current_data = None
                st.session_state.chatbot.data_description = ""
                st.success("Dataset cleared!")
    
    # Load datasets if uploaded
    if uploaded_files and st.session_state.chatbot.current_data is None:
        with st.spinner("Loading datasets..."):
            df, description = st.session_state.chatbot.load_dataset(uploaded_files)
            if df is not None:
                st.session_state.chatbot.current_data = df
                st.session_state.chatbot.data_description = description
                st.success(f"Loaded {len(uploaded_files)} file(s) successfully!")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'message': f"Dataset loaded successfully!\n\n{description}"
                })
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for chat in st.session_state.chat_history:
            if chat['type'] == 'user':
                st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{chat["message"]}</div>', unsafe_allow_html=True)
                
                # Display any additional content
                if 'code' in chat:
                    if st.session_state.show_code:
                        st.code(chat['code'], language='python')
                
                if 'dataframe' in chat:
                    st.dataframe(chat['dataframe'])
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_link = st.session_state.chatbot.create_download_link(
                            chat['dataframe'], 'analysis_result', 'csv'
                        )
                        st.markdown(csv_link, unsafe_allow_html=True)
                    with col2:
                        excel_link = st.session_state.chatbot.create_download_link(
                            chat['dataframe'], 'analysis_result', 'excel'
                        )
                        st.markdown(excel_link, unsafe_allow_html=True)
                
                if 'plot' in chat:
                    st.pyplot(chat['plot'])
                    
                    # Plot download option
                    plot_link = st.session_state.chatbot.create_plot_download_link(
                        chat['plot'], 'analysis_plot'
                    )
                    st.markdown(plot_link, unsafe_allow_html=True)
    
    # Input area (positioned at bottom)
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
    
    # Create columns for input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask me anything about your data...",
            placeholder="e.g., 'Show me the correlation between sales and profit'",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    # Process user query
    if (send_button or user_query) and user_query:
        if st.session_state.chatbot.current_data is None:
            st.error("Please upload a dataset first!")
            return
        
        # Add user message to history
        st.session_state.chat_history.append({
            'type': 'user',
            'message': user_query
        })
        
        # Generate response
        with st.spinner("Analyzing and generating code..."):
            # Create prompt for Ollama
            prompt = f"""
            Dataset Description:
            {st.session_state.chatbot.data_description}
            
            User Query: {user_query}
            
            Please provide Python code to analyze this dataset and answer the user's query.
            Use pandas (df), matplotlib (plt), seaborn (sns), and plotly as needed.
            The dataset is already loaded as 'df'.
            
            Make sure to:
            1. Handle any missing values appropriately
            2. Create visualizations when relevant
            3. Provide clear, informative output
            4. Use appropriate statistical methods
            
            Return only executable Python code without any explanations or markdown formatting.
            """
            
            # Get response from Ollama
            ollama_response = st.session_state.chatbot.query_ollama(prompt)
            
            # Extract Python code
            code_blocks = st.session_state.chatbot.extract_python_code(ollama_response)
            
            if code_blocks:
                # Execute the code
                success, output, exec_globals = st.session_state.chatbot.execute_code(code_blocks[0])
                
                if success:
                    response_data = {
                        'type': 'bot',
                        'message': f"Analysis completed successfully!\n\nOutput:\n{output}",
                        'code': code_blocks[0]
                    }
                    
                    # Check for dataframes in execution globals
                    for var_name, var_value in exec_globals.items():
                        if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                            response_data['dataframe'] = var_value
                            break
                    
                    # Check for plots
                    if plt.get_fignums():
                        response_data['plot'] = plt.gcf()
                    
                    st.session_state.chat_history.append(response_data)
                    
                else:
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'message': f"Error executing code: {output}",
                        'code': code_blocks[0]
                    })
            else:
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'message': f"I couldn't generate proper code for your query. Here's what I got:\n\n{ollama_response}"
                })
        
        # Clear input and rerun
        st.session_state.user_input = ""
        st.rerun()

if __name__ == "__main__":
    main()