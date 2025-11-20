# GitHub Repository Setup

Follow these steps to create and push your repository to GitHub.

## Step 1: Create Repository on GitHub

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `FinAnalysis` (or your preferred name)
   - **Description**: "Financial Analysis RAG Application - AI-powered stock analysis with RAG from financial textbooks"
   - **Visibility**: Choose **Private** (for beta) or **Public**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/FinAnalysis.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/FinAnalysis.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

1. Go to your repository on GitHub
2. Verify all files are there
3. Check that `.env` and `chroma_db/` are NOT in the repository (they're in .gitignore)

## Important Notes

### Files NOT in Repository (by design):
- `.env` - Contains your GROQ_API_KEY (keep this secret!)
- `chroma_db/` - Large database files (will be built on deployment)
- `venv/` - Virtual environment
- `temp_charts/` - Temporary chart files

### Files IN Repository:
- All source code
- Requirements.txt
- Dockerfile and deployment configs
- README and documentation
- PDF files in `data/FinAnalysisTexts/` (if you want to include them)

### Optional: Add PDFs to Git

If you want to include the PDF files in the repository:

1. Remove `chroma_db/` from .gitignore (it's already there)
2. Optionally, you can add PDFs:
   ```bash
   git add data/FinAnalysisTexts/*.pdf
   git commit -m "Add financial analysis PDFs"
   git push
   ```

**Note**: PDFs are large files. Consider using Git LFS (Large File Storage) if they're very large:
```bash
git lfs install
git lfs track "*.pdf"
git add .gitattributes
git add data/FinAnalysisTexts/*.pdf
git commit -m "Add PDFs with Git LFS"
```

## Next Steps

After pushing to GitHub:

1. **Set up deployment** (see DEPLOYMENT.md):
   - Connect GitHub repo to Railway/Render/Fly.io
   - Add `GROQ_API_KEY` as environment variable
   - Deploy!

2. **Add collaborators** (for beta testing):
   - Go to repository Settings → Collaborators
   - Add beta testers

3. **Set up GitHub Actions** (optional):
   - For CI/CD
   - Automated testing
   - Deployment automation

## Troubleshooting

### If you get authentication errors:
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### If push is rejected:
```bash
git pull origin main --rebase
git push -u origin main
```

### To check remote:
```bash
git remote -v
```

