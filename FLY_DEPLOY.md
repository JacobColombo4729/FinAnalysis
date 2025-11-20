# Fly.io Deployment Guide

## Step 1: Install Fly CLI

### macOS:
```bash
curl -L https://fly.io/install.sh | sh
```

### Or using Homebrew:
```bash
brew install flyctl
```

### Verify installation:
```bash
fly version
```

## Step 2: Login to Fly.io

```bash
fly auth login
```

This will open a browser for authentication.

## Step 3: Initialize Fly App (if not already done)

If you haven't created the app yet:

```bash
fly launch
```

Follow the prompts:
- **App name**: `finanalysis` (or your choice, must be globally unique)
- **Region**: Choose closest to your users (e.g., `iad` for US East)
- **Postgres/Redis**: Say "no" for now
- **Deploy now**: Say "no" - we'll configure first

## Step 4: Set Environment Variables

Set your GROQ API key:

```bash
fly secrets set GROQ_API_KEY=your_groq_api_key_here
```

## Step 5: Configure Resources (Important!)

Fly.io free tier has limited resources. Set appropriate limits:

```bash
# Set memory limit (2GB recommended minimum)
fly scale memory 2048

# Set VM size (shared-cpu-2x recommended)
fly scale vm shared-cpu-2x
```

Or add to `fly.toml`:
```toml
[compute]
  memory_mb = 2048
  cpu_kind = "shared"
  cpus = 2
```

## Step 6: Deploy

```bash
fly deploy
```

This will:
1. Build the Docker image
2. Push to Fly.io
3. Deploy the application

## Step 7: Check Status

```bash
# View app status
fly status

# View logs
fly logs

# Open app in browser
fly open
```

## Troubleshooting

### Build Fails

Check build logs:
```bash
fly logs
```

Common issues:
- **Memory issues**: Increase memory limit
- **Port issues**: Ensure using `$PORT` environment variable
- **Dependencies**: Check requirements.txt is complete

### App Crashes

View logs:
```bash
fly logs
```

Check if:
- GROQ_API_KEY is set correctly
- ChromaDB is building (first startup takes time)
- Port is correctly configured

### Update App

After making changes:
```bash
git add .
git commit -m "Your changes"
git push
fly deploy
```

### View App Info

```bash
# App details
fly info

# Environment variables
fly secrets list

# SSH into app
fly ssh console
```

## Useful Commands

```bash
# Restart app
fly apps restart finanalysis

# Scale app
fly scale count 1

# View metrics
fly metrics

# Check regions
fly regions list
```

## Cost Considerations

Fly.io free tier includes:
- 3 shared-cpu-1x VMs with 256MB RAM
- 160GB outbound data transfer

For this app, you'll likely need:
- At least 2GB RAM (shared-cpu-2x)
- This may cost ~$5-10/month

Check pricing: https://fly.io/docs/about/pricing/

## Next Steps

1. Deploy successfully
2. Test the application
3. Share URL with beta testers
4. Monitor usage and costs
5. Set up custom domain (optional)

