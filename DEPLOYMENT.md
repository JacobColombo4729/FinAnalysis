# Deployment Guide for Beta Testing

This guide covers multiple deployment options for the Financial Analysis application.

## Prerequisites

1. **Environment Variables**: You'll need a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Knowledge Base**: The ChromaDB knowledge base must be populated before deployment. Run locally:
   ```bash
   python utils/embeddings.py
   ```
   Then include the `chroma_db/` directory in your deployment.

## Option 1: Railway (Recommended for Beta)

Railway is easy to use and has a free tier.

### Steps:

1. **Install Railway CLI** (optional):
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy via Railway Dashboard**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo" (or upload files)
   - Railway will auto-detect the Dockerfile

3. **Set Environment Variables**:
   - In Railway dashboard, go to Variables
   - Add `GROQ_API_KEY=your_key_here`

4. **Configure Port**:
   - Railway automatically handles port mapping
   - The app will be available at `your-app.railway.app`

### Railway Configuration:

Create `railway.json` (optional):
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "chainlit run app.py --port $PORT --host 0.0.0.0",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

## Option 2: Render

Render offers free tier with automatic deployments.

### Steps:

1. **Connect GitHub Repository**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**:
   - **Name**: `finanalysis` (or your choice)
   - **Environment**: `Docker`
   - **Region**: Choose closest to users
   - **Branch**: `main` (or your branch)
   - **Root Directory**: `/` (root)

3. **Environment Variables**:
   - Add `GROQ_API_KEY` in the Environment section

4. **Deploy**:
   - Render will build and deploy automatically
   - Your app will be at `your-app.onrender.com`

### Render Configuration:

Create `render.yaml`:
```yaml
services:
  - type: web
    name: finanalysis
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: GROQ_API_KEY
        sync: false
    healthCheckPath: /
```

## Option 3: Fly.io

Fly.io offers global deployment with good free tier.

### Steps:

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   fly auth login
   ```

2. **Initialize Fly App**:
   ```bash
   fly launch
   ```
   - Follow prompts
   - Don't deploy yet (we'll configure first)

3. **Create `fly.toml`**:
   ```toml
   app = "your-app-name"
   primary_region = "iad"

   [build]
     dockerfile = "Dockerfile"

   [env]
     PORT = "8000"

   [[services]]
     internal_port = 8000
     protocol = "tcp"

     [[services.ports]]
       handlers = ["http"]
       port = 80

     [[services.ports]]
       handlers = ["tls", "http"]
       port = 443

     [services.concurrency]
       type = "connections"
       hard_limit = 25
       soft_limit = 20

   [[services.http_checks]]
     interval = "10s"
     timeout = "2s"
     grace_period = "5s"
     method = "GET"
     path = "/"
   ```

4. **Set Secrets**:
   ```bash
   fly secrets set GROQ_API_KEY=your_key_here
   ```

5. **Deploy**:
   ```bash
   fly deploy
   ```

## Option 4: Docker + Any Cloud Platform

### Build and Test Locally:

```bash
# Build image
docker build -t finanalysis:latest .

# Run locally to test
docker run -p 8000:8000 --env-file .env finanalysis:latest
```

### Deploy to Cloud:

- **AWS ECS/Fargate**: Use AWS Console or CLI
- **Google Cloud Run**: `gcloud run deploy`
- **Azure Container Instances**: Use Azure Portal
- **DigitalOcean App Platform**: Connect GitHub and deploy

## Option 5: Chainlit Cloud (If Available)

Chainlit may offer cloud hosting. Check [chainlit.io](https://chainlit.io) for cloud deployment options.

## Important Notes for Beta Testing

### 1. Knowledge Base Setup

The `chroma_db/` directory must be included in deployment. Options:

**Option A: Include in Git** (not recommended for large files):
- Add `chroma_db/` to repository
- Note: This may make repo large

**Option B: Build on First Deploy**:
- Add startup script that runs `python utils/embeddings.py` if `chroma_db/` is empty
- Slower first startup but keeps repo small

**Option C: Use External Storage**:
- Store ChromaDB on S3/cloud storage
- Download on container startup

### 2. Persistent Storage

For production, consider:
- Using external database for ChromaDB
- Storing `chroma_db/` in persistent volume
- Using cloud vector database (Pinecone, Weaviate, etc.)

### 3. Environment Variables

Always set:
- `GROQ_API_KEY`: Required for LLM
- `CHAINLIT_HOST`: Usually `0.0.0.0` for containers
- `CHAINLIT_PORT`: Usually `8000` or `$PORT`

### 4. Resource Requirements

Recommended:
- **Memory**: At least 2GB (for embeddings and models)
- **CPU**: 2+ cores recommended
- **Storage**: 5GB+ (for ChromaDB and dependencies)

### 5. Security

For beta testing:
- Consider adding authentication (Chainlit supports this)
- Rate limiting
- API key protection
- HTTPS only

## Quick Start: Railway (Easiest)

1. Push code to GitHub
2. Go to railway.app
3. New Project → Deploy from GitHub
4. Add `GROQ_API_KEY` environment variable
5. Deploy!

Your app will be live in minutes.

## Troubleshooting

### ChromaDB Issues:
- Ensure `chroma_db/` directory exists and has data
- Check file permissions
- Verify ChromaDB version compatibility

### Port Issues:
- Some platforms use `$PORT` environment variable
- Update Dockerfile CMD to use `$PORT` if needed

### Memory Issues:
- Increase container memory limit
- Consider using lighter embedding models

## Beta Testing Checklist

- [ ] Knowledge base populated and included
- [ ] Environment variables set
- [ ] App builds successfully
- [ ] Can access app URL
- [ ] Stock analysis works
- [ ] RAG queries work
- [ ] Charts generate correctly
- [ ] No memory/performance issues
- [ ] Error handling works
- [ ] Share URL with beta testers

## Support

For deployment issues, check:
- Platform-specific logs
- Container logs
- Environment variable configuration
- Port and networking settings

