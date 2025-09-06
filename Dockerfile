FROM ollama/ollama

# Pull llama3 model during build
RUN ollama pull llama3:latest

# Install required system dependencies
RUN apt-get update && apt-get install -y python3-pip python3-dev

# Copy the app
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose default Streamlit port
EXPOSE 7860

# Run Ollama server and your Streamlit app
CMD ollama serve & sleep 10 && streamlit run app.py --server.port 7860 --server.enableCORS false
