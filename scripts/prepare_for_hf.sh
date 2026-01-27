#!/bin/bash
# scripts/prepare_for_hf.sh

DEPLOY_DIR="deploy_package"

echo "📦 Preparing deployment package in '$DEPLOY_DIR'..."

# 1. Clean and Create Directory
echo "🧹 Cleaning previous build (preserving .git)..."
mkdir -p "$DEPLOY_DIR"
if [ -d "$DEPLOY_DIR/.git" ]; then
    find "$DEPLOY_DIR" -mindepth 1 -maxdepth 1 ! -name ".git" -exec rm -rf {} +
else
    rm -rf "$DEPLOY_DIR"
    mkdir -p "$DEPLOY_DIR"
fi

# Ensure local demo assets are clean before copying
find demo -name "*.wav" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 2. Copy Root Files
echo "📄 Copying Python modules..."
cp tagger.py phonemizer.py utils.py pyproject.toml .dockerignore "$DEPLOY_DIR/"

# Fix pyproject.toml for deployment (remove missing api/main)
python3 -c '
content = open("'$DEPLOY_DIR/pyproject.toml'").read()
content = content.replace(", \"api\"", "").replace("\"api\", ", "")
content = content.replace(", \"main\"", "").replace("\"main\", ", "")
open("'$DEPLOY_DIR/pyproject.toml'", "w").write(content)
'

# 3. Copy Demo App
echo "🎨 Copying Demo application..."
cp -r demo "$DEPLOY_DIR/"

# 4. Copy Dockerfile
# Move it to root of deploy dir for HF Spaces
echo "🐳 Copying Dockerfile..."
cp docker/Dockerfile.demo "$DEPLOY_DIR/Dockerfile"

# 5. Copy Essential Data
echo "📚 Copying Lexicon..."
mkdir -p "$DEPLOY_DIR/data"
cp data/dholuo_lexicon.json "$DEPLOY_DIR/data/"

# 6. Models (SKIPPED - Loaded from Hub at runtime)
echo "🤖 Model files will be downloaded at runtime from the Hub to save Space storage."

# 7. Create README for HF Spaces
echo "📝 Generating README.md..."
cat > "$DEPLOY_DIR/README.md" <<EOF
---
title: Dholuo TTS Demo
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Dholuo TTS Demo 🌍
Context-aware Text-to-Speech synthesis for Dholuo.
EOF

echo "✅ Deployment package ready at '$DEPLOY_DIR'!"
echo "   Size: $(du -sh "$DEPLOY_DIR" | cut -f1)"
echo "   Next: Follow the steps in deployment_guide.md to push to Hugging Face."
