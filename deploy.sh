#!/bin/bash

# Deployment script for Streamlit Cloud
# This script sets up the environment and installs Playwright browsers

set -e  # Exit on any error

echo "🚀 Starting deployment setup..."

# Install system dependencies (these should be in packages.txt)
echo "📦 Installing system dependencies..."

# Set up environment variables for Playwright
export PLAYWRIGHT_BROWSERS_PATH=0
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0

# Create necessary directories
mkdir -p ~/.cache/ms-playwright

echo "🔧 Setting up Playwright..."

# Install Playwright browsers
python -m playwright install chromium

# Verify installation
if [ -d "$HOME/.cache/ms-playwright" ]; then
    echo "✅ Playwright browsers installed successfully!"
    ls -la ~/.cache/ms-playwright/
else
    echo "❌ Playwright installation failed!"
    exit 1
fi

echo "🎉 Deployment setup completed successfully!" 