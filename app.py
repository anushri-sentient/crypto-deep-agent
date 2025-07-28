import streamlit as st
import requests
import pandas as pd
import logging
import asyncio
from dotenv import load_dotenv
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Constants
DEFILLAMA_POOLS_API = "https://yields.llama.fi/pools"
EXA_API_KEY = os.getenv("EXA_API_KEY")

ALLOWED_PROJECTS = [
    "pendle", "compound-v3", "compound-v2", "beefy", "aave-v3", "aave-v2",
    "uniswap-v3", "uniswap-v2", "euler-v2", "curve-dex", "aerodrome-slipstream", "aerodrome-v1"
]

SIMPLE_STAKING_PROJECTS = [
    "lido", "binance-staked-eth", "rocket-pool", "stakewise-v2",
    "meth-protocol", "liquid-collective", "binance-staked-sol",
    "marinade-liquid-staking", "benqi-staked-avax","jito-liquid-staking"
]

RISK_COLORS = {
    "Low": "#d4edda",
    "Medium": "#fff3cd",
    "High": "#f8d7da"
}
RISK_TEXT_COLORS = {
    "Low": "#155724",
    "Medium": "#856404",
    "High": "#721c24"
}

# --- Helper functions ---

def classify_pool_type(pool):
    name = pool.get("pool", "").lower()
    project = pool.get("project", "").lower()
    symbol = pool.get("symbol", "").lower()

    if project in SIMPLE_STAKING_PROJECTS:
        return "Simple Staking"
    if any(token in symbol for token in ["steth", "cbeth", "wbeth", "stsol", "stavax"]) and ("lp" in name or "pool" in name):
        return "Staking on LP"
    if project in ["aave", "compound", "venus", "morpho"]:
        return "Lending"
    if ("usdc" in symbol and "eth" in symbol) or "lp" in name or "curve" in project or "uniswap" in project:
        return "LP Farming (DEX)"
    if project in ["yearn", "beefy", "autofarm", "reaper"]:
        return "Vault"
    if pool.get("apy", 0) > 30 or pool.get("tvlUsd", 0) < 500_000:
        return "High-Risk Farm"
    return "Other"
def setup_playwright():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        print("✅ Playwright browsers are already installed")
        return True

    except Exception as e:
        print(f"⚠️ Playwright browsers not found or failed to launch: {e}")
        print("🔧 Installing Playwright browsers...")
        try:
            subprocess.run([
                sys.executable, "-m", "playwright", "install", "chromium"
            ], check=True)
            print("✅ Playwright browsers installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Playwright browsers: {e}")
            return False

def classify_risk(pool):
    il_risk = pool.get("ilRisk")
    if il_risk is not None and str(il_risk).lower() == "no":
        return "Low"

    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    strategy = classify_pool_type(pool)

    if apy > 20 or tvl < 100_000:
        return "High"
    if strategy in ["Simple Staking", "Lending"]:
        if tvl > 5_000_000 and apy < 7:
            return "Low"
        return "Medium"
    if strategy in ["LP Farming (DEX)", "Vault", "Staking on LP"]:
        if 5 <= apy <= 15 and tvl > 1_000_000:
            return "Medium"
        elif apy < 5 and tvl > 1_000_000 and str(il_risk).lower() == "no":
            return "Low"
        return "High"
    if tvl > 10_000_000 and apy < 5:
        return "Low"
    if 0.5 <= apy <= 10:
        return "Medium"
    return "High"

def is_valid_lp_pool(pool):
    symbol = pool.get("symbol", "").lower()
    name = pool.get("pool", "").lower()
    project = pool.get("project", "").lower()
    apy = pool.get("apy", 0)
    apy_7d = pool.get("apyBase7d") or apy
    apy_30d = pool.get("apyMean30d") or apy
    tvl = pool.get("tvlUsd", 0)

    if project not in ALLOWED_PROJECTS and project not in SIMPLE_STAKING_PROJECTS:
        return False
    if tvl < 500_000:
        return False
    if apy > 100 or apy < 0.5:
        return False
    if apy_7d is not None and apy_30d is not None and apy_30d > 0:
        if (apy_7d / apy_30d > 4):
            return False
    return True

def score_pool(pool):
    tvl = pool.get("tvlUsd", 0)
    apy = pool.get("apy", 0)
    return tvl * 0.7 + apy * 10000

@st.cache_data(ttl=300)
def fetch_yield_opportunities(token):
    try:
        res = requests.get(DEFILLAMA_POOLS_API)
        res.raise_for_status()
        data = res.json()
        pools = data.get("data", [])
        filtered = []
        token = token.lower()
        for p in pools:
            if not isinstance(p, dict):
                continue
            symbol = p.get("symbol", "").lower()
            project = p.get("project", "").lower()
            if token not in symbol and token not in project:
                continue
            if is_valid_lp_pool(p):
                filtered.append(p)
        return filtered
    except Exception as e:
        st.error(f"Error fetching pools: {e}")
        return []

@st.cache_data(ttl=600)
def fetch_pool_chart_data(pool_id):
    try:
        url = f"https://yields.llama.fi/chart/{pool_id}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        chart = data.get("data", [])
        logging.debug(f"Fetched {len(chart)} points for pool {pool_id}")

        if not chart:
            return None

        df = pd.DataFrame(chart)
        if df.empty or 'apy' not in df.columns:
            logging.debug(f"No data or 'apy' column missing for pool {pool_id}")
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df['apy']

    except Exception as e:
        logging.error(f"Error fetching or parsing data for pool {pool_id}: {e}")
        return None

def format_pool_card_with_details(pool, strategy, risk):
    apy = pool.get("apy", 0)
    apy_base = pool.get("apyBase", None)
    apy_reward = pool.get("apyReward", None)
    apy_7d = pool.get("apyBase7d") or apy
    apy_30d = pool.get("apyMean30d") or apy
    tvl = pool.get("tvlUsd", 0)
    symbol = pool.get("symbol", "N/A").upper()
    project = pool.get("project", "N/A").title()
    chain = pool.get("chain", "N/A").upper()
    pool_id = pool.get("pool")
    il_risk = pool.get("ilRisk", "N/A")
    prediction = pool.get("predictions", {})
    pred_class = prediction.get("predictedClass", "N/A")
    pred_prob = prediction.get("predictedProbability", "N/A")
    binned_conf = prediction.get("binnedConfidence", "N/A")
    underlying_tokens = pool.get("underlyingTokens", [])
    url = f"https://defillama.com/yields/pool/{pool_id}"

    bg_color = RISK_COLORS.get(risk, "#f0f0f0")
    text_color = RISK_TEXT_COLORS.get(risk, "#000")

    underlying_tokens_str = ", ".join([ut[:6] + "..." + ut[-4:] for ut in underlying_tokens]) if underlying_tokens else "N/A"

    return f"""
    <div style="background-color: {bg_color}; color: {text_color}; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="margin-top: 0;">{project} - <span style="font-weight: normal;">{symbol}</span></h3>
        <p><strong>📍 Chain:</strong> {chain}</p>
        <p><strong>💰 TVL:</strong> ${tvl:,.0f}</p>
        <p><strong>📊 APY (Base):</strong> {apy_base if apy_base is not None else 'N/A'}%</p>
        <p><strong>📈 APY (Reward):</strong> {apy_reward if apy_reward is not None else 'N/A'}%</p>
        <p><strong>🔄 APY (Current):</strong> {apy:.2f}%</p>
        <p><strong>🔄 APY (7d mean):</strong> {apy_7d:.2f}%</p>
        <p><strong>🔄 APY (30d mean):</strong> {apy_30d:.2f}%</p>
        <p><strong>⚠️ IL Risk:</strong> {il_risk}</p>
        <p><strong>🧩 Strategy:</strong> {strategy}</p>
        <p><strong>🚦 Risk Level:</strong> {risk}</p>
        <p><strong>🧪 Prediction:</strong> Class: {pred_class} | Probability: {pred_prob} | Confidence: {binned_conf}</p>
        <p><strong>🔗 <a href="{url}" target="_blank">View on DefiLlama</a></strong></p>
        <p><strong>🔑 Underlying Tokens:</strong> {underlying_tokens_str}</p>
    </div>
    """

import time

@st.cache_data(ttl=900)
def get_news_from_exa_api(token):
    if not EXA_API_KEY:
        logging.warning("EXA_API_KEY not set")
        return "EXA API key missing."

    try:
        logging.info(f"📡 Querying EXA API for news: {token}")
        start_time = time.time()

        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "query": token,
                "text": True
            }
        )
        response.raise_for_status()
        elapsed = time.time() - start_time
        logging.info(f"✅ Received EXA news for '{token}' in {elapsed:.2f} seconds")

        

        news_json = response.json()
        results = news_json.get("results", [])
        if not results:
            return f"No recent news found for '{token}'."

        return "\n".join(
            f"- {item.get('title', 'No Title')} ({item.get('url', 'No URL')})"
            for item in results
        )

    except Exception as e:
        logging.error(f"❌ Error fetching news for {token}: {e}")
        return "Error fetching news."



def get_pool_url_from_id(pool_id):
    return f"https://defillama.com/yields/pool/{pool_id}"

@st.cache_data(ttl=3600)
def gpt4o_summarize_all(crawled_content: str, pool: dict, news: str) -> str:
    from openai import OpenAI
    client = OpenAI()

    if not crawled_content and not news:
        return "No content or news to summarize."

    pool_info = f"""
Project: {pool.get('project', 'N/A').title()}
Symbol: {pool.get('symbol', 'N/A').upper()}
Chain: {pool.get('chain', 'N/A').upper()}
TVL: ${pool.get('tvlUsd', 0):,.0f}
APY (Current): {pool.get('apy', 0):.2f}%
APY (Base): {pool.get('apyBase', 'N/A')}%
APY (Reward): {pool.get('apyReward', 'N/A')}%
IL Risk: {pool.get('ilRisk', 'N/A')}
Strategy: {classify_pool_type(pool)}
"""

    prompt = (
        f"You are a DeFi expert. Summarize the following DeFi yield pool.\n"
        f"Use both the structured pool data, the crawled page content, and recent news.\n"
        f"Focus on project purpose, features, risks/benefits, yield strategies, and any predictions.\n\n"
        f"Pool Data:\n{pool_info}\n\n"
        f"Crawled Content:\n{crawled_content}\n\n"
        f"Recent News:\n{news}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a DeFi expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"GPT-4o summarization error: {e}")
        return "Error summarizing content."

async def crawl_pool_url_async(pool_url: str) -> str:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(pool_url)
            r.raise_for_status()
            return r.text
    except Exception as e:
        logging.error(f"Error crawling pool url {pool_url}: {e}")
        return ""

async def _crawl_news_and_summarize(pool_url: str, pool: dict) -> str:
    try:
        crawled_html = await crawl_pool_url_async(pool_url)

        symbol = pool.get("symbol", "").upper()
        project = pool.get("project", "").title()
        chain = pool.get("chain", "").upper()

        # Fetch news for symbol, project and chain
        news_symbol = get_news_from_exa_api(symbol)
        news_project = get_news_from_exa_api(project)
        news_chain = get_news_from_exa_api(chain)

        combined_news = (
            f"News on {symbol}:\n{news_symbol}\n\n"
            f"News on {project}:\n{news_project}\n\n"
            f"News on {chain}:\n{news_chain}"
        )

        summary = gpt4o_summarize_all(crawled_html, pool, combined_news)
        return summary
    except Exception as e:
        logging.error(f"Error during crawling/news fetching/summarization: {e}")
        return "Error during crawling or summarization."

def get_crawled_news_summary_for_pool(pool_url: str, pool: dict) -> str:
    return asyncio.run(_crawl_news_and_summarize(pool_url, pool))


# --- Main app ---

def main():
    st.title("🚀 DeFi Yield Opportunities Explorer")

    token = st.text_input("Enter a token symbol (e.g., ETH, USDC, MATIC):").strip()
    if not token:
        st.info("Please enter a token symbol above to fetch yield opportunities.")
        return

    logging.info(f"🔍 Searching yield opportunities for token: {token}")
    with st.spinner(f"Fetching yield opportunities for '{token.upper()}'..."):
        pools = fetch_yield_opportunities(token)

    if not pools:
        st.warning(f"No suitable yield pools found for token '{token}'.")
        return

    # Sort pools by score descending
    pools.sort(key=score_pool, reverse=True)

    # Group pools by risk
    risk_groups = {"Low": [], "Medium": [], "High": []}
    for pool in pools:
        risk = classify_risk(pool)
        if risk in risk_groups:
            risk_groups[risk].append(pool)

    # Initialize show_more state for each risk
    for risk in risk_groups:
        key = f"show_more_{risk.lower()}"
        if key not in st.session_state:
            st.session_state[key] = False

    st.markdown(f"### Top yield pools for '{token.upper()}' (max 3 per risk level)")

    for risk in ["Low", "Medium", "High"]:
        all_pools = risk_groups[risk]
        if not all_pools:
            continue

        st.subheader(f"{risk} Risk Pools")
        key = f"show_more_{risk.lower()}"
        display_pools = all_pools if st.session_state[key] else all_pools[:3]

        for pool in display_pools:
            strategy = classify_pool_type(pool)
            risk_label = classify_risk(pool)
            st.markdown(format_pool_card_with_details(pool, strategy, risk_label), unsafe_allow_html=True)

            pool_id = pool.get("pool")
            summary_key = f"ai_summary_{pool_id}"

            if summary_key not in st.session_state:
                st.session_state[summary_key] = ""

            if st.button(f"Get AI Summary for {pool_id}", key=f"btn_{pool_id}"):
                with st.spinner("Fetching AI summary... This may take up to 30s."):
                    pool_url = get_pool_url_from_id(pool_id)
                    summary = get_crawled_news_summary_for_pool(pool_url, pool)
                    st.session_state[summary_key] = summary

            if st.session_state[summary_key]:
                st.markdown(f"**AI Summary:**\n\n{st.session_state[summary_key]}")

            st.markdown("---")

        if len(all_pools) > 3:
            toggle_label = "Show Less" if st.session_state[key] else "Show More"
            if st.button(f"{toggle_label} {risk} Risk Pools", key=f"toggle_{risk.lower()}"):
                st.session_state[key] = not st.session_state[key]


if __name__ == "__main__":
    setup_playwright()
    main()
