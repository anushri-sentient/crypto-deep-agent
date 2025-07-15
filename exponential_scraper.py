from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
import subprocess
import sys
import os

def setup_playwright():
    """Ensure Playwright browsers are installed."""
    try:
        # Check if browsers are already installed
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        print("✅ Playwright browsers are already installed")
        return True
    except Exception as e:
        print(f"⚠️ Playwright browsers not found: {e}")
        print("🔧 Installing Playwright browsers...")
        try:
            subprocess.run([
                sys.executable, "-m", "playwright", "install", "chromium"
            ], check=True, capture_output=True)
            print("✅ Playwright browsers installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Playwright browsers: {e}")
            return False

def scrape_exponential_pools_by_risk():
    # Ensure Playwright is set up
    if not setup_playwright():
        print("❌ Cannot proceed without Playwright browsers")
        return pd.DataFrame()
    
    url = "https://exponential.fi/pools/search?assets=USDC&assets=USDT&assets=DAI&assets=TUSD&assets=LUSD&assets=GUSD&assets=sUSD&assets=MIM&assets=alUSD&risk=A&risk=B&blockchains=Ethereum&blockchains=Optimism&blockchains=Base&blockchains=Polygon&blockchains=Solana&blockchains=Arbitrum"
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            # Wait for pool rows to load
            page.wait_for_selector("tr", timeout=20000)
            html = page.content()
            browser.close()
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return pd.DataFrame()
    
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        print("No pool table found.")
        return pd.DataFrame()
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    pools = []
    for row in table.find_all("tr")[1:]:
        tds = row.find_all("td")
        cells = []
        for i, td in enumerate(tds):
            if headers[i].lower() == "pool":
                text = td.get_text(" ", strip=True)
                # Remove all 'verified' (case-insensitive) from anywhere in the name
                clean_name = re.sub(r"verified", "", text, flags=re.IGNORECASE).strip()
                cells.append(clean_name)
            else:
                cells.append(td.get_text(strip=True))
        if len(cells) == len(headers):
            pools.append(dict(zip(headers, cells)))
    df = pd.DataFrame(pools)
    # Clean and sort by APY
    if 'Yield' in df.columns:
        def parse_apy(val):
            match = re.search(r"([\d.]+)%", val)
            return float(match.group(1)) if match else 0.0
        df['APY'] = df['Yield'].apply(parse_apy)
        df = df.sort_values(by='APY', ascending=False)
    # Add debug column for cleaned pool name
    if 'Pool' in df.columns:
        df['Cleaned Pool'] = df['Pool']
        # Extract protocol/project as first word (after cleaning)
        df['Extracted Protocol'] = df['Pool'].apply(lambda x: x.split()[0] if isinstance(x, str) and x else None)
    return df

# Fetch DeFi Llama pools

def fetch_defillama_pools():
    url = "https://yields.llama.fi/pools"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["data"]

def normalize_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

def match_to_defillama(df):
    llama_pools = fetch_defillama_pools()
    llama_lookup = {}
    for pool in llama_pools:
        key = normalize_name(pool.get("project", ""))
        llama_lookup.setdefault(key, []).append(pool)

    def find_llama_row(row):
        protocol = normalize_name(row.get("Extracted Protocol", ""))
        for llama_project, pools in llama_lookup.items():
            if protocol == llama_project or protocol in llama_project:
                return pools[0]
        return None

    df["defillama"] = df.apply(find_llama_row, axis=1)
    # Optionally extract some DeFi Llama fields for display
    df["DeFiLlama Project"] = df["defillama"].apply(lambda x: x["project"] if x else None)
    df["DeFiLlama TVL"] = df["defillama"].apply(lambda x: x["tvlUsd"] if x else None)
    df["DeFiLlama APY"] = df["defillama"].apply(lambda x: x["apy"] if x else None)
    df["DeFiLlama URL"] = df["defillama"].apply(lambda x: f"https://defillama.com/protocol/{x['project']}" if x and x.get('project') else None)
    return df

# Convenience function for Streamlit

def scrape_and_match_exponential_to_defillama():
    df = scrape_exponential_pools_by_risk()
    if df is not None and not df.empty:
        df = match_to_defillama(df)
    return df

if __name__ == "__main__":
    df = scrape_and_match_exponential_to_defillama()
    if not df.empty:
        print("Top 10 pools by APY (Risk A & B, Major Stables, Main Chains):")
        print(df.head(10)[['Pool', 'Chain', 'TVL', 'Risk', 'Yield']])
    else:
        print("No pools found.") 