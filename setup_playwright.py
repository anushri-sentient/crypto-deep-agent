#!/usr/bin/env python3
"""
Setup script for Playwright on Streamlit Cloud deployment.
This script installs the required browser binaries for Playwright.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_playwright_browsers():
    """Install Playwright browsers."""
    try:
        print("Installing Playwright browsers...")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Playwright browsers installed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Playwright browsers: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def setup_environment():
    """Set up environment variables for Playwright."""
    # Set environment variables for headless operation
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "0"
    
    # Create necessary directories
    cache_dir = Path.home() / ".cache" / "ms-playwright"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Environment set up. Cache directory: {cache_dir}")

def main():
    """Main setup function."""
    print("🚀 Setting up Playwright for Streamlit Cloud deployment...")
    
    # Set up environment
    setup_environment()
    
    # Install browsers
    if install_playwright_browsers():
        print("🎉 Playwright setup completed successfully!")
        return 0
    else:
        print("💥 Playwright setup failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 