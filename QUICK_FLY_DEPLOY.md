# Quick Fly.io Deployment

## Prerequisites

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

## Deploy Steps

1. **Set your API key**:
   ```bash
   fly secrets set GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Deploy**:
   ```bash
   fly deploy
   ```

3. **Open your app**:
   ```bash
   fly open
   ```

## If App Already Exists

If you've already created the app on Fly.io:

1. **Link to existing app**:
   ```bash
   fly apps open finanalysis
   ```

2. **Or update fly.toml** with your actual app name:
   ```toml
   app = "your-actual-app-name"
   ```

3. **Then deploy**:
   ```bash
   fly deploy
   ```

## Troubleshooting Build Failures

If you see "Generate requirements for build" failed:

1. **Check Dockerfile** - Make sure it's valid
2. **Check requirements.txt** - All dependencies listed
3. **View build logs**:
   ```bash
   fly logs
   ```

4. **Try building locally first**:
   ```bash
   docker build -t finanalysis:test .
   ```

## Important Notes

- **First deployment** will take longer (builds knowledge base)
- **Memory**: App needs at least 2GB RAM
- **Cost**: ~$5-10/month for shared-cpu-2x with 2GB RAM
- **URL**: Your app will be at `https://finanalysis.fly.dev`

For detailed instructions, see `FLY_DEPLOY.md`.

