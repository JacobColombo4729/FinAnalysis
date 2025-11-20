#!/bin/bash
# Startup script for deployment
# Checks if ChromaDB exists, builds if needed

set -e

echo "Starting Financial Analysis Application..."

# Check if chroma_db exists and has data
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo "ChromaDB not found or empty. Building knowledge base..."
    echo "This may take several minutes..."
    python utils/embeddings.py
    echo "Knowledge base built successfully!"
else
    echo "ChromaDB found. Skipping knowledge base build."
fi

# Start the application
echo "Starting Chainlit application..."
# Use PORT from environment (Fly.io provides this)
exec chainlit run app.py --port ${PORT:-8000} --host ${CHAINLIT_HOST:-0.0.0.0}

