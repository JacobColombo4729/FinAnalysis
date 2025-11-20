FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db temp_charts

# Expose Chainlit port
EXPOSE 8000

# Set environment variables
ENV CHAINLIT_HOST=0.0.0.0
ENV PORT=8000

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Run the application (use startup script to handle knowledge base)
CMD ["./start.sh"]

