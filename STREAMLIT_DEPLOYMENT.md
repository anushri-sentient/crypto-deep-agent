# Streamlit Cloud Deployment Guide

This guide explains how to deploy your crypto analysis app to Streamlit Cloud with Playwright support.

## Files Created for Deployment

### 1. `packages.txt`
Contains system dependencies required for Playwright:
- `ffmpeg` - For video processing
- Browser dependencies (libnss3, libatk-bridge2.0-0, etc.)
- System utilities (wget, unzip)

### 2. `.streamlit/config.toml`
Streamlit configuration for optimal deployment:
- Headless server mode
- Disabled CORS and XSRF protection
- Custom theme colors

### 3. `setup_playwright.py`
Python script to install Playwright browsers during deployment.

### 4. `deploy.sh`
Shell script for deployment setup (executable).

## Deployment Steps

### Option 1: Automatic Deployment (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit Cloud deployment files"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set the main file path to your Streamlit app (e.g., `exponential_streamlit.py`)
   - Deploy

### Option 2: Manual Setup

If you need to run the setup manually:

1. **Install system dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install -y $(cat packages.txt)
   ```

2. **Run the deployment script**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Or run the Python setup**
   ```bash
   python setup_playwright.py
   ```

## Troubleshooting

### Common Issues

1. **Playwright browsers not found**
   - Solution: The `setup_playwright()` function in `exponential_scraper.py` will automatically install browsers
   - Check logs for installation messages

2. **Permission errors**
   - Ensure `deploy.sh` is executable: `chmod +x deploy.sh`
   - Check that system dependencies are installed

3. **Memory issues**
   - Streamlit Cloud has memory limits
   - The app uses headless mode to minimize memory usage
   - Consider implementing caching for expensive operations

### Environment Variables

The following environment variables are set automatically:
- `PLAYWRIGHT_BROWSERS_PATH=0` - Use system cache
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0` - Allow browser downloads

### Monitoring Deployment

1. **Check Streamlit Cloud logs**
   - Look for "✅ Playwright browsers installed successfully"
   - Monitor for any error messages

2. **Test the scraper**
   - The app will automatically test Playwright setup
   - Check the "Setup Status" section in your app

## File Structure

```
crypto-deep-agent/
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── packages.txt              # System dependencies
├── requirements.txt          # Python dependencies
├── setup_playwright.py      # Playwright setup script
├── deploy.sh                # Deployment script
├── exponential_scraper.py   # Updated scraper with error handling
└── STREAMLIT_DEPLOYMENT.md  # This file
```

## Performance Tips

1. **Caching**: Use `@st.cache_data` for expensive operations
2. **Headless mode**: Already configured for minimal resource usage
3. **Error handling**: The scraper includes comprehensive error handling
4. **Timeout settings**: Increased timeouts for slow connections

## Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are committed to GitHub
3. Ensure the main file path is correct in Streamlit Cloud settings
4. Test locally first: `streamlit run exponential_streamlit.py` 