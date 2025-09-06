title: Data Analyst Chatbot
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
  - streamlit
pinned: false
short_description: chatbot

# 🧠 AI Analyst Chatbot with Memory, LangChain & SQL

An intelligent data analysis chatbot that combines Streamlit, LangChain, and Ollama LLM to provide conversational data analysis with memory capabilities.

## Features

- **Multi-format file support**: CSV, Excel (XLSX/XLS), and PDF files
- **Conversational AI**: Uses LangChain with Ollama for natural language data queries
- **Memory retention**: Maintains conversation context across interactions
- **Dataset merging**: Combine multiple datasets for comprehensive analysis
- **Code generation**: Automatically generates and executes Python code for data analysis
- **Visual outputs**: Supports matplotlib plots and data visualizations
- **Export capabilities**: Download results as CSV/Excel and plots as PNG
- **Schema-aware**: Automatically detects and uses correct column names and data types

## Prerequisites

Before running this application, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed and running locally
3. **Llama3 model** pulled in Ollama (`ollama pull llama3:latest`)

### Installing Ollama

Visit [Ollama's website](https://ollama.ai) and follow the installation instructions for your operating system.

After installation, pull the required model:
```bash
ollama pull llama3:latest
```

## Installation

1. Clone or download the application files

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Start Ollama** (if not already running):
```bash
ollama serve
```

2. **Run the Streamlit app**:
```bash
streamlit run app.py
```

3. **Open your browser** to `http://localhost:8501`

## How to Use

### 1. Upload Data
- Click "📁 Upload datasets" to upload CSV, Excel, or PDF files
- For Excel files, select the specific sheet you want to analyze
- View your data using the checkboxes to ensure it loaded correctly

### 2. Merge Datasets (Optional)
- If you have multiple datasets, enable "🔗 Merge Datasets"
- Select the datasets to merge and specify the common column
- The merged dataset will be available for analysis

### 3. Ask Questions
- Select your dataset from the dropdown
- Type natural language questions about your data
- Examples:
  - "Show me the top 10 products by revenue"
  - "Create a histogram of sales data"
  - "Group by region and calculate average profit"
  - "What's the correlation between price and quantity?"

### 4. Review Results
- The AI generates Python code automatically
- Code is executed safely in an isolated environment
- Results are displayed as tables, charts, or text
- Download results and visualizations as needed

## Supported Query Types

- **Aggregations**: Sum, average, count, min, max
- **Filtering**: Show data based on conditions
- **Grouping**: Group by categories and calculate metrics
- **Visualizations**: Histograms, scatter plots, line charts, bar charts
- **Statistical analysis**: Correlations, distributions, trends
- **Data cleaning**: Handle missing values, data types

## File Format Support

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Automatically detects encoding |
| Excel | `.xlsx`, `.xls` | Multi-sheet support |
| PDF | `.pdf` | Extracts text content |

## Technical Details

### Architecture
- **Frontend**: Streamlit for the web interface
- **AI Engine**: LangChain + Ollama (Llama3)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Memory**: ConversationBufferMemory for context retention

### Security Features
- Code execution in isolated environment
- Schema validation to prevent column name errors
- Syntax validation before execution
- Safe import restrictions

### Code Quality
- Optional Black formatting for generated code
- AST parsing for syntax validation
- Error handling and user-friendly error messages

## Troubleshooting

### Common Issues

1. **"Connection refused" error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the Llama3 model is installed: `ollama list`

2. **"Column not found" errors**:
   - The AI uses exact column names from your dataset
   - Check the "Dataset Info" section to see available columns

3. **Memory issues with large files**:
   - Consider sampling your data for initial exploration
   - Use filtering to work with subsets of large datasets

4. **Slow response times**:
   - Large datasets may take longer to process
   - Complex queries require more computation time
   - Consider using a more powerful machine or GPU acceleration

### Performance Tips

- Start with smaller datasets to test queries
- Use specific, clear questions for better code generation
- Check the dataset schema before asking questions
- Use the conversation memory - refer to previous results

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages include:

- **streamlit**: Web app framework
- **langchain**: LLM orchestration
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **pdfplumber**: PDF text extraction
- **openpyxl**: Excel file support

## Contributing

This is a demonstration application. For production use, consider:

- Adding authentication and user management
- Implementing data validation and sanitization
- Adding support for more file formats
- Implementing caching for better performance
- Adding logging and monitoring

## License

[Specify your license here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify Ollama and Llama3 are working independently

---

**Note**: This application requires a local Ollama installation and does not send data to external services, ensuring your data privacy and security.