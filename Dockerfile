FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for better performance
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Using --timeout to prevent hanging, and installing in one go for better caching
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db temp_charts

# Expose Chainlit port (Render will set PORT env var)
EXPOSE 8000

# Set environment variables
# Render will override PORT, but we set defaults
ENV CHAINLIT_HOST=0.0.0.0
ENV PORT=8000

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Run the application (use startup script to handle knowledge base)
CMD ["./start.sh"]

