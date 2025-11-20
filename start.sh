#!/bin/bash
# Startup script for deployment
# Starts app immediately, builds knowledge base in background if needed

echo "Starting Financial Analysis Application..."

# Use CHROMA_DB_PATH env var if set (for persistent volumes), otherwise use ./chroma_db
CHROMA_DB_PATH=${CHROMA_DB_PATH:-./chroma_db}
export CHROMA_DB_PATH

# Ensure the directory exists
mkdir -p "${CHROMA_DB_PATH}"

# Check if chroma_db exists and has data
if [ ! -d "${CHROMA_DB_PATH}" ] || [ -z "$(ls -A "${CHROMA_DB_PATH}" 2>/dev/null)" ]; then
    echo "ChromaDB not found or empty at ${CHROMA_DB_PATH}. Will build knowledge base in background..."
    echo "App will start immediately. Knowledge base will be available once built."
    # Build knowledge base in background (non-blocking)
    nohup python utils/embeddings.py > /tmp/kb_build.log 2>&1 &
    KB_BUILD_PID=$!
    echo "Knowledge base build started in background (PID: $KB_BUILD_PID)"
else
    echo "ChromaDB found at ${CHROMA_DB_PATH}. Skipping knowledge base build."
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
# exec replaces the shell process for proper signal handling
exec chainlit run app.py --port ${PORT} --host ${CHAINLIT_HOST} --no-open

