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
# Use PORT from environment (Fly.io/Render provides this)
# Default to 8000 if not set, but Render will set PORT automatically
PORT=${PORT:-8000}
CHAINLIT_HOST=${CHAINLIT_HOST:-0.0.0.0}
echo "Binding to ${CHAINLIT_HOST}:${PORT}"
echo "PORT environment variable: ${PORT}"
echo "CHAINLIT_HOST environment variable: ${CHAINLIT_HOST}"

# Export PORT so Chainlit can read it
export PORT
export CHAINLIT_HOST

# Start Chainlit with explicit port and host binding
# Use --no-open to prevent opening browser, and ensure it binds properly
exec chainlit run app.py --port ${PORT} --host ${CHAINLIT_HOST} --no-open

