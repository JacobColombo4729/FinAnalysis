#!/bin/bash
# Startup script for deployment
# Starts app immediately, builds knowledge base in background if needed

echo "Starting Financial Analysis Application..."

# Check if chroma_db exists and has data
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo "ChromaDB not found or empty. Will build knowledge base in background..."
    echo "App will start immediately. Knowledge base will be available once built."
    # Build knowledge base in background (non-blocking)
    nohup python utils/embeddings.py > /tmp/kb_build.log 2>&1 &
    KB_BUILD_PID=$!
    echo "Knowledge base build started in background (PID: $KB_BUILD_PID)"
else
    echo "ChromaDB found. Skipping knowledge base build."
fi

# Start the application immediately (don't wait for knowledge base)
echo "Starting Chainlit application..."
# Use PORT from environment (Fly.io provides this)
exec chainlit run app.py --port ${PORT:-8000} --host ${CHAINLIT_HOST:-0.0.0.0}

