import streamlit as st
import requests
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import os
import openai
import re
from report import generate_pool_analysis  # Updated import
import subprocess
import sys
from optimizer import create_portfolio_optimizer

# Configuration
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Constants
DEFILLAMA_POOLS_API = "https://yields.llama.fi/pools"
COINGECKO_API = "https://api.coingecko.com/api/v3"
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ALLOWED_PROJECTS = [
    "pendle", "compound-v3", "compound-v2", "beefy", "aave-v3", "aave-v2",
    "uniswap-v3", "uniswap-v2", "euler-v2", "curve-dex", "aerodrome-slipstream",
    "aerodrome-v1", "morpho"
]

SIMPLE_STAKING_PROJECTS = [
    "lido", "binance-staked-eth", "rocket-pool", "stakewise-v2",
    "meth-protocol", "liquid-collective", "binance-staked-sol",
    "marinade-liquid-staking", "benqi-staked-avax", "jito-liquid-staking"
]

# Styling
def load_custom_css():
    """Load custom CSS styles for the application"""
    st.markdown("""
    <style>
    .main { padding-top: 1rem; }

    .token-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .pool-card {
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        background: white;
    }

    .pool-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    .risk-low { border-left: 5px solid #10b981; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); }
    .risk-medium { border-left: 5px solid #f59e0b; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); }
    .risk-high { border-left: 5px solid #ef4444; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); }

    .portfolio-summary {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
    }

    .ai-summary {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .comprehensive-report {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }

    .breakdown-report {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }

    .strategy-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .report-type-selector {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
    }

    .badge-staking { background: #dbeafe; color: #1e40af; }
    .badge-lending { background: #dcfce7; color: #166534; }
    .badge-farming { background: #fef3c7; color: #92400e; }
    .badge-vault { background: #e7e5e4; color: #44403c; }
    </style>
    """, unsafe_allow_html=True)

# Classification Functions (keeping existing functions)
def classify_pool_type(pool):
    """Classify the pool type based on project and symbol"""
    name = pool.get("pool", "").lower()
    project = pool.get("project", "").lower()
    symbol = pool.get("symbol", "").lower()

    if project in SIMPLE_STAKING_PROJECTS:
        return "Simple Staking"
    if any(token in symbol for token in ["steth", "cbeth", "wbeth", "stsol", "stavax"]) and ("lp" in name or "pool" in name):
        return "Staking on LP"
    if "aave" in project or "compound" in project or project in ["venus", "morpho"]:
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
    """Classify risk level of a pool"""
    il_risk = pool.get("ilRisk")
    if il_risk is not None and str(il_risk).lower() == "no":
        return "Low"

    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    strategy = classify_pool_type(pool)

    if apy > 20 or tvl < 100_000:
        return "High"

    if strategy in ["Simple Staking", "Lending"]:
        return "Low" if tvl > 5_000_000 and apy < 7 else "Medium"

    if strategy in ["LP Farming (DEX)", "Vault", "Staking on LP"]:
        if apy > 25:
            return "High"
        elif 5 <= apy <= 18 and tvl > 1_000_000:
            return "Medium"
        elif apy < 5 and tvl > 500_000 and str(il_risk).lower() == "no":
            return "Low"
        elif tvl > 10_000_000 and apy <= 20:
            return "Medium"
        return "High"

    if tvl > 10_000_000 and apy < 5:
        return "Low"
    return "Medium" if 0.5 <= apy <= 10 else "High"

def is_valid_pool(pool):
    """Check if a pool meets validity criteria"""
    project = pool.get("project", "").lower()
    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    apy_7d = pool.get("apyBase7d") or apy
    apy_30d = pool.get("apyMean30d") or apy

    if project not in ALLOWED_PROJECTS and project not in SIMPLE_STAKING_PROJECTS:
        return False
    if tvl < 500_000 or apy > 100 or apy < 0.5:
        return False
    if apy_30d and apy_30d > 0 and (apy_7d / apy_30d > 4):
        return False
    return True

def score_pool(pool):
    """Score a pool for ranking"""
    tvl = pool.get("tvlUsd", 0)
    apy = pool.get("apy", 0)
    risk_level = classify_risk(pool)

    score = 0
    if risk_level == "Low":
        score = tvl * 0.0001 + apy * 10
    elif risk_level == "Medium":
        score = tvl * 0.00005 + apy * 20
    else:
        score = tvl * 0.00001 + apy * 30
    return score

# Data Fetching Functions (keeping existing functions)
@st.cache_data(ttl=300)
def fetch_coingecko_token_data(token_id):
    """Fetch token data from CoinGecko API"""
    try:
        search_url = f"{COINGECKO_API}/search?query={token_id}"
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_data = search_response.json()

        coins = search_data.get("coins", [])
        if not coins:
            return None

        coin_id = None
        for coin in coins:
            if coin["symbol"].lower() == token_id.lower():
                coin_id = coin["id"]
                break

        if not coin_id:
            if coins:
                coin_id = coins[0]["id"]
            else:
                return None

        token_url = f"{COINGECKO_API}/coins/{coin_id}"
        token_response = requests.get(token_url)
        token_response.raise_for_status()
        token_data = token_response.json()

        history_url = f"{COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days=7"
        history_response = requests.get(history_url)
        history_response.raise_for_status()
        history_data = history_response.json()

        return {
            "coin_id": coin_id,
            "token_data": token_data,
            "price_history": history_data
        }

    except Exception as e:
        logging.error(f"Error fetching CoinGecko data for {token_id}: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_yield_opportunities(token):
    """Fetch yield opportunities for a token"""
    try:
        response = requests.get(DEFILLAMA_POOLS_API)
        response.raise_for_status()
        data = response.json()
        pools = data.get("data", [])

        filtered_pools = []
        token = token.lower()

        for pool in pools:
            if not isinstance(pool, dict):
                continue

            symbol = pool.get("symbol", "").lower()
            project = pool.get("project", "").lower()

            if (token in symbol or token in project) and is_valid_pool(pool):
                filtered_pools.append(pool)

        return filtered_pools

    except Exception as e:
        st.error(f"Error fetching pools: {e}")
        return []

@st.cache_data(ttl=900)
def get_news_summary(token, pools=None):
    """Get news summary using EXA API"""
    if not EXA_API_KEY:
        return "EXA API key missing for news analysis.", []

    search_terms = [token]

    if pools:
        unique_chains = list(set([pool.get("chain", "") for pool in pools[:5] if pool.get("chain")]))
        unique_projects = list(set([pool.get("project", "") for pool in pools[:5] if pool.get("project")]))

        for chain in unique_chains[:2]:
            if chain.lower() not in ["ethereum", "arbitrum", "optimism", "polygon", "binance", "solana"]:
                search_terms.append(chain)
        for project in unique_projects[:2]:
            if project.lower() not in ["uniswap", "aave", "curve", "compound"]:
                search_terms.append(project)

    try:
        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "query": " ".join(search_terms),
                "text": True,
                "numResults": 5,
                "type": "neural",
                "publishedAfter": "2024-01-01"
            }
        )
        response.raise_for_status()

        news_json = response.json()
        results = news_json.get("results", [])

        if not results:
            return f"No recent DeFi news found for {token}.", []

        news_summary_display = f"**📰 Recent DeFi News for {token.upper()}**"

        context_info = []
        if unique_chains:
            context_info.append(f"Active on: {', '.join(unique_chains[:3])}")
        if unique_projects:
            context_info.append(f"Key protocols: {', '.join(unique_projects[:3])}")

        if context_info:
            news_summary_display += f"\n*Context: {' | '.join(context_info)}*"

        news_summary_display += "\n\n"

        raw_news_data = []
        for i, item in enumerate(results[:3], 1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            text = item.get("text", "")

            if text:
                text_lower = text.lower()
                relevant_section = ""
                token_lower = token.lower()
                sentences = text.split('. ')
                for sentence in sentences:
                    if token_lower in sentence.lower():
                        relevant_section = sentence.strip()
                        break
                if not relevant_section and len(text) > 200:
                    relevant_section = text[:200]
                elif not relevant_section:
                    relevant_section = text

                snippet = relevant_section + "..." if len(relevant_section) < len(text) else relevant_section
            else:
                snippet = "No preview available."

            news_summary_display += f"{i}. **{title}**\n   {snippet}\n   [Read more]({url})\n\n"
            raw_news_data.append(item)

        return news_summary_display, raw_news_data

    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return "Unable to fetch recent news at this time.", []


# UI Components (Keeping existing functions and adding Nansen display)
def create_token_info_card(token_data):
    """Create token info card with CoinGecko data"""
    if not token_data:
        return

    token_info = token_data["token_data"]
    market_data = token_info.get("market_data", {})

    current_price = market_data.get("current_price", {}).get("usd", 0)
    price_change_24h = market_data.get("price_change_percentage_24h", 0)
    market_cap = market_data.get("market_cap", {}).get("usd", 0)
    volume_24h = market_data.get("total_volume", {}).get("usd", 0)

    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 2rem; border-radius: 20px; margin: 1rem 0;">
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            image_url = token_info.get('image', {}).get('small', '')
            if image_url:
                st.image(image_url, width=60)

        with col2:
            st.markdown(f"## {token_info.get('name', 'N/A')}")
            st.markdown(f"**{token_info.get('symbol', 'N/A').upper()}**")

        price_change_emoji = "⬆️" if price_change_24h >= 0 else "⬇️"
        change_color = "#4ade80" if price_change_24h >= 0 else "#f87171"
        change_symbol = "+" if price_change_24h >= 0 else ""

        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">${current_price:,.4f}</div>
            <div style="font-size: 1.2rem; color: {change_color};">{price_change_emoji} {change_symbol}{price_change_24h:.2f}% (24h)</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Market Cap**")
            st.markdown(f"${market_cap:,.0f}")
        with col2:
            st.markdown("**Volume (24h)**")
            st.markdown(f"${volume_24h:,.0f}")

        st.markdown("</div>", unsafe_allow_html=True)

def create_pool_card_with_dual_reports(pool, strategy, risk, token_data_for_ai, raw_news_data_for_ai):
    """Enhanced pool card with both report types"""
    apy = pool.get("apy", 0)
    apy_base = pool.get("apyBase", None)
    apy_reward = pool.get("apyReward", None)
    tvl = pool.get("tvlUsd", 0)
    symbol = pool.get("symbol", "N/A").upper()
    project = pool.get("project", "N/A").title()
    chain = pool.get("chain", "N/A").upper()
    pool_id = pool.get("pool", "")
    il_risk = pool.get("ilRisk", "N/A")
    url = f"https://defillama.com/yields/pool/{pool_id}"

    # Session state keys for both report types
    breakdown_key = f"breakdown_report_{pool_id}"
    comprehensive_key = f"comprehensive_report_{pool_id}"
    
    if breakdown_key not in st.session_state:
        st.session_state[breakdown_key] = False
    if comprehensive_key not in st.session_state:
        st.session_state[comprehensive_key] = False

    with st.container():
        border_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
        border_color = border_colors.get(risk, "#6b7280")

        st.markdown(f"""
        <div style="border-left: 5px solid {border_color}; background: white; border-radius: 15px;
                    padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {project}")
            st.markdown(f"**{symbol}** • {chain}")

            strategy_colors = {
                "Simple Staking": "#dbeafe",
                "Lending": "#dcfce7",
                "LP Farming (DEX)": "#fef3c7",
                "Vault": "#e7e5e4",
                "Staking on LP": "#ccb8ee",
                "High-Risk Farm": "#fecaca",
                "Other": "#e2e8f0"
            }
            badge_color = strategy_colors.get(strategy, "#f3f4f6")
            st.markdown(f"""
            <span style="background: {badge_color}; padding: 0.3rem 0.8rem; border-radius: 15px;
                         font-size: 0.8rem; font-weight: 600;">{strategy}</span>
            """, unsafe_allow_html=True)

        with col2:
            apy_color = "#10b981" if apy < 10 else "#f59e0b" if apy < 20 else "#ef4444"
            st.markdown(f"""
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: bold; color: {apy_color};">{apy:.2f}%</div>
                <div style="color: #6b7280;">APY</div>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TVL", f"${tvl/1000000:.1f}M")
        with col2:
            st.metric("Chain", chain)
        with col3:
            st.metric("Risk", risk)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Base APY:** {apy_base if apy_base is not None else 'N/A'}%")
            st.write(f"**IL Risk:** {il_risk}")
        with col2:
            st.write(f"**Reward APY:** {apy_reward if apy_reward is not None else 'N/A'}%")
            st.write(f"**Pool ID:** {pool_id[:15]}...")

        st.markdown(f"[🔗 View on DefiLlama]({url})")

        # Report selection buttons
        st.markdown("### 📊 Analysis Reports")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Pool Breakdown Analysis", key=f"breakdown_btn_{pool_id}", help="Quick tactical analysis with metrics focus"):
                st.session_state[breakdown_key] = not st.session_state[breakdown_key]
                st.session_state[comprehensive_key] = False  # Close other report
        
        with col2:
            if st.button("🔍 Comprehensive Investment Report", key=f"comprehensive_btn_{pool_id}", help="Deep research with web crawling"):
                st.session_state[comprehensive_key] = not st.session_state[comprehensive_key]
                st.session_state[breakdown_key] = False  # Close other report

        # Display breakdown report
        if st.session_state[breakdown_key]:
            with st.spinner("🔄 Generating Pool Breakdown Analysis..."):
                # Prepare token info string for the breakdown report
                token_info_str = ""
                if token_data_for_ai and token_data_for_ai.get("token_data"):
                    td = token_data_for_ai["token_data"]
                    md = td.get("market_data", {})
                    token_info_str = f"""
                    Token: {td.get('name', 'N/A')} ({td.get('symbol', 'N/A').upper()})
                    Price: ${md.get('current_price', {}).get('usd', 0):,.4f}
                    24h Change: {md.get('price_change_percentage_24h', 0):.2f}%
                    Market Cap: ${md.get('market_cap', {}).get('usd', 0):,.0f}
                    """
                
                breakdown_report = generate_pool_analysis(
                    pool_data=pool,
                    token_info=token_info_str,
                    news_data=raw_news_data_for_ai,
                    report_type="breakdown"
                )
            
            st.markdown(f'<div class="breakdown-report">{breakdown_report}</div>', unsafe_allow_html=True)

        # Display comprehensive report
        if st.session_state[comprehensive_key]:
            with st.spinner("🌐 Generating Comprehensive Investment Report (includes web crawling)..."):
                comprehensive_report = generate_pool_analysis(
                    pool_data=pool,
                    token_info="",  # Comprehensive report gets info from web crawling
                    news_data=raw_news_data_for_ai,
                    report_type="comprehensive"
                )
            
            st.markdown(f'<div class="comprehensive-report">{comprehensive_report}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

def create_price_chart(price_history):
    """Create price chart using Plotly"""
    if not price_history:
        return None

    prices = price_history.get("prices", [])
    if not prices:
        return None

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        mode="lines",
        name="Price",
        line=dict(color="#3b82f6", width=3),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.1)"
    ))

    fig.update_layout(
        title="7-Day Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )

    return fig

# Enhanced Portfolio Functions
@st.cache_data(ttl=3600, show_spinner=True)
def generate_llm_portfolio_recommendation(token_symbol, total_value, eligible_pools, risk_preference):
    """Generate portfolio recommendation using LLM"""
    if not OPENAI_API_KEY:
        return None, "OpenAI API key is not configured for LLM portfolio."

    openai.api_key = OPENAI_API_KEY

    pools_for_llm = []
    for p in eligible_pools[:25]:
        pools_for_llm.append({
            "pool_id": p.get("pool"),
            "project": p.get("project"),
            "symbol": p.get("symbol"),
            "chain": p.get("chain"),
            "apy": p.get("apy"),
            "tvlUsd": p.get("tvlUsd"),
            "risk": classify_risk(p),
            "strategy": classify_pool_type(p)
        })

    pools_str = "\n".join([
        f"- ID: {p['pool_id']} | Project: {p['project']} | Symbol: {p['symbol']} | Chain: {p['chain']} | APY: {p['apy']:.2f}% | TVL: ${p['tvlUsd'] / 1e6:.1f}M | Risk: {p['risk']} | Strategy: {p['strategy']}"
        for p in pools_for_llm
    ])

    prompt = f"""
You are an expert DeFi portfolio manager. Construct a profitable, chain-diverse, and risk-appropriate yield farming portfolio.

**User Investment Details:**
- Total amount: ${total_value:,.0f} in {token_symbol.upper()}
- Risk Preference: {risk_preference}

**Available Pools:**
{pools_str}

**Instructions:**
1. Determine optimal diversification strategy based on risk preference
2. Select pools from the provided list (use exact Pool IDs)
3. Prefer exposure to at least 3 different blockchains
4. Provide percentage allocations that sum to 100%

**Portfolio Explanation:**
[Explain your strategic approach and rationale]

**Portfolio Allocation:**
| Pool ID | Project | Symbol | Chain | APY (%) | Risk Level | Allocation (%) |
|---|---|---|---|---|---|---|
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional DeFi portfolio manager."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        llm_output = response.choices[0].message.content

        if "Portfolio Allocation:" not in llm_output:
            return None, "LLM response malformed."

        explanation_part, allocation_table_part = llm_output.split("Portfolio Allocation:", 1)
        explanation = explanation_part.strip()

        table_lines = allocation_table_part.strip().split('\n')
        data_rows = [line for line in table_lines if line.strip().startswith('|') and '---' not in line and 'Pool ID' not in line]

        portfolio_allocations = []
        pool_id_map = {p.get("pool", ""): p for p in eligible_pools}

        for row in data_rows:
            if "TOTAL" in row.upper():
                continue
            cols = [col.strip() for col in row.strip('|').split('|')]
            if len(cols) == 7:
                try:
                    pool_id = cols[0]
                    allocation_percent_str = cols[6].replace('%', '').strip()
                    allocation_percent = float(allocation_percent_str)

                    original_pool_data = pool_id_map.get(pool_id)

                    if original_pool_data:
                        portfolio_allocations.append({
                            "pool": original_pool_data,
                            "allocation_percent": allocation_percent,
                            "risk": classify_risk(original_pool_data),
                            "apy": original_pool_data.get("apy", 0)
                        })
                    else:
                        logging.warning(f"Could not map pool ID '{pool_id}' from LLM output.")
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing table row: {row} - {e}")
                    continue

        # Normalize allocations
        current_total_allocation = sum([item["allocation_percent"] for item in portfolio_allocations])
        if current_total_allocation > 0 and abs(current_total_allocation - 100) > 0.1:
            for item in portfolio_allocations:
                item["allocation_percent"] = (item["allocation_percent"] / current_total_allocation) * 100

        return portfolio_allocations, explanation

    except openai.AuthenticationError:
        return None, "OpenAI API key is invalid or not set."
    except Exception as e:
        logging.error(f"Failed to generate LLM portfolio: {e}")
        return None, f"An unexpected error occurred: {e}"

# Main Application
def main():
    st.set_page_config(
        page_title="DeFi Yield Explorer - Enhanced Reports",
        page_icon="🚀",
        layout="wide"
    )

    load_custom_css()

    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🚀 DeFi Yield Explorer
        </h1>
        <p style="font-size: 1.2rem; color: #6b7280; margin: 0.5rem 0;">
            Discover, analyze, and optimize your DeFi yield farming strategies with dual AI report types
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced intro section explaining report types
    with st.expander("📋 Report Types Explained", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📈 Pool Breakdown Analysis
            **Fast & Tactical** - Perfect for active traders
            
            **Features:**
            - ⚡ Quick generation (no web crawling)
            - 📊 Deep metrics analysis (APY breakdown, IL calculations)
            - 🎯 Position sizing recommendations
            - 📉 Technical indicators & efficiency ratios
            - 🔄 Rebalancing strategies
            - 💰 Capital allocation guidance
            
            **Best for:** Day-to-day portfolio management, tactical decisions
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Comprehensive Investment Report
            **Deep & Thorough** - Perfect for major investments
            
            **Features:**
            - 🌐 Web crawling of official docs & audits
            - 📋 Multi-source information synthesis
            - 🛡️ Detailed security analysis
            - 📖 Protocol deep-dive research
            - ⚖️ Risk assessment from multiple angles
            - 🎯 Investment thesis validation
            
            **Best for:** Major investment decisions, due diligence
            """)

    col1, col2 = st.columns([2, 1])
    with col1:
        token = st.text_input("🔍 Enter token symbol (e.g., ETH, USDC, MATIC):", placeholder="ETH").strip()
    with col2:
        token_amount = st.number_input("💰 Token amount you have:", min_value=0.0, value=1.0, step=0.1)

    if not token:
        st.info("👆 Enter a token symbol above to start exploring yield opportunities")
        return

    with st.spinner("🔄 Fetching token data and yield opportunities..."):
        token_data = fetch_coingecko_token_data(token)
        pools = fetch_yield_opportunities(token)

    if not pools:
        st.warning(f"❌ No suitable yield pools found for token '{token.upper()}'")
        # Still show token info and news/Nansen if pools are empty but token data exists
        if token_data:
            create_token_info_card(token_data)
            price_chart = create_price_chart(token_data.get("price_history"))
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            news_summary_display, raw_news_for_ai = get_news_summary(token, pools) # Pass empty pools if none

            # Directly show Tab 3 content if no pools found for quick lookup
            st.markdown("---")
            st.markdown("### 📰 Market Intelligence (No Yield Pools Found)")
            st.markdown(news_summary_display)

            market_data = token_data["token_data"].get("market_data", {})
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Market Metrics (from CoinGecko)")
                st.write(f"**All-time High:** ${market_data.get('ath', {}).get('usd', 0):,.4f}")
                st.write(f"**All-time Low:** ${market_data.get('atl', {}).get('usd', 0):,.4f}")
                st.write(f"**Market Cap Rank:** #{market_data.get('market_cap_rank', 'N/A')}")
            with col2:
                st.markdown("#### ⏱️ Performance (from CoinGecko)")
                st.write(f"**7d Change:** {market_data.get('price_change_percentage_7d', 0):.2f}%")
                st.write(f"**30d Change:** {market_data.get('price_change_percentage_30d', 0):.2f}%")
                st.write(f"**1y Change:** {market_data.get('price_change_percentage_1y', 0):.2f}%")
            

    if token_data:
        create_token_info_card(token_data)

        price_chart = create_price_chart(token_data.get("price_history"))
        if price_chart:
            st.plotly_chart(price_chart, use_container_width=True)

        token_price = token_data["token_data"].get("market_data", {}).get("current_price", {}).get("usd", 0)
    else:
        st.warning("⚠️ Could not fetch token data from CoinGecko")
        token_price = 0

    pools.sort(key=score_pool, reverse=True)
    news_summary_display, raw_news_for_ai = get_news_summary(token, pools)

    tab1, tab2, tab3 = st.tabs(["🎯 Pool Explorer", "📊 Portfolio Constructor", "📰 Market Intel"])

    with tab1:
        risk_groups = {"Low": [], "Medium": [], "High": []}
        for pool in pools:
            risk = classify_risk(pool)
            if risk in risk_groups:
                risk_groups[risk].append(pool)

        for risk in risk_groups:
            key = f"show_more_{risk.lower()}"
            if key not in st.session_state:
                st.session_state[key] = False

        st.markdown(f"### 🏆 Top Yield Opportunities for {token.upper()}")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_apy = sum([p["apy"] for p in pools]) / len(pools) if pools else 0
            st.metric("Average APY", f"{avg_apy:.2f}%")
        with col2:
            total_tvl = sum([p["tvlUsd"] for p in pools])
            st.metric("Total TVL", f"${total_tvl/1000000:.1f}M")
        with col3:
            st.metric("Available Pools", len(pools))
        with col4:
            st.metric("Low Risk Pools", len(risk_groups["Low"]))

        # Report type selector at the top
        st.markdown("### 📊 Choose Your Analysis Style")
        
        analysis_style = st.radio(
            "Select default report type for pools:",
            ["📈 Pool Breakdown (Fast)", "🔍 Comprehensive (Deep)"],
            horizontal=True,
            help="You can always switch between both types for each individual pool"
        )

        risk_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

        for risk in ["Low", "Medium", "High"]:
            risk_pools = risk_groups[risk]
            if not risk_pools:
                continue

            st.markdown(f"## {risk_emojis[risk]} {risk} Risk Pools ({len(risk_pools)} available)")

            key = f"show_more_{risk.lower()}"
            display_pools = risk_pools if st.session_state[key] else risk_pools[:3]

            for pool in display_pools:
                strategy = classify_pool_type(pool)
                risk_level = classify_risk(pool)
                create_pool_card_with_dual_reports(pool, strategy, risk_level, token_data, raw_news_for_ai)
                st.markdown("---")

            if len(risk_pools) > 3:
                toggle_label = "Show Less" if st.session_state[key] else f"Show {len(risk_pools) - 3} More"
                if st.button(f"{toggle_label} {risk} Risk Pools", key=f"toggle_{risk.lower()}"):
                    st.session_state[key] = not st.session_state[key]
                    st.rerun()

    with tab2:
        st.markdown("### 📊 AI-Powered Portfolio Constructor")
        
        if token_price > 0:
            col1, col2 = st.columns(2)
            with col1:
                investment_amount = st.number_input(
                    f"💰 Investment Amount (USD)", 
                    min_value=100.0, 
                    value=float(token_amount * token_price) if token_price > 0 else 1000.0,
                    step=100.0
                )
            
            with col2:
                risk_preference = st.selectbox(
                    "🎯 Risk Preference",
                    ["Conservative", "Balanced", "Aggressive"],
                    index=1
                )

            if st.button("🤖 Generate AI Portfolio", type="primary"):
                with st.spinner("🔄 AI is analyzing pools and generating optimal portfolio..."):
                    portfolio_allocations, explanation = generate_llm_portfolio_recommendation(
                        token, investment_amount, pools[:25], risk_preference
                    )
                
                if portfolio_allocations:
                    st.markdown("### 🎯 Your Optimized Portfolio")
                    st.markdown(f"**Strategy Explanation:**\n{explanation}")
                    
                    # Portfolio summary
                    total_expected_apy = sum([item["apy"] * (item["allocation_percent"]/100) for item in portfolio_allocations])
                    chain_diversity = len(set([item["pool"]["chain"] for item in portfolio_allocations]))
                    risk_distribution = {}
                    for item in portfolio_allocations:
                        risk_level = item["risk"]
                        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + item["allocation_percent"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Expected APY", f"{total_expected_apy:.2f}%")
                    with col2:
                        st.metric("Chain Diversity", f"{chain_diversity} chains")
                    with col3:
                        st.metric("Total Investment", f"${investment_amount:,.0f}")
                    with col4:
                        low_risk_pct = risk_distribution.get("Low", 0)
                        st.metric("Low Risk %", f"{low_risk_pct:.1f}%")
                    
                    # Detailed portfolio breakdown
                    st.markdown("### 📋 Portfolio Breakdown")
                    
                    portfolio_data = []
                    for item in portfolio_allocations:
                        pool = item["pool"]
                        allocation_amount = investment_amount * (item["allocation_percent"] / 100)
                        
                        portfolio_data.append({
                            "Project": pool["project"],
                            "Symbol": pool["symbol"],
                            "Chain": pool["chain"],
                            "APY": f"{pool['apy']:.2f}%",
                            "Risk": item["risk"],
                            "Allocation": f"{item['allocation_percent']:.1f}%",
                            "Amount": f"${allocation_amount:,.0f}",
                            "Expected Annual": f"${allocation_amount * pool['apy'] / 100:,.0f}"
                        })
                    
                    df = pd.DataFrame(portfolio_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    if st.button("📥 Export Portfolio Data"):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"{token.upper()}_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    # Individual pool reports for portfolio
                    st.markdown("### 📊 Detailed Analysis for Portfolio Pools")
                    
                    for item in portfolio_allocations:
                        pool = item["pool"]
                        strategy = classify_pool_type(pool)
                        risk_level = item["risk"]
                        
                        with st.expander(f"📈 {pool['project']} - {pool['symbol']} ({item['allocation_percent']:.1f}% allocation)"):
                            # Quick breakdown report for each portfolio pool
                            token_info_str = ""
                            if token_data and token_data.get("token_data"):
                                td = token_data["token_data"]
                                md = td.get("market_data", {})
                                token_info_str = f"""
                                Token: {td.get('name', 'N/A')} ({td.get('symbol', 'N/A').upper()})
                                Price: ${md.get('current_price', {}).get('usd', 0):,.4f}
                                24h Change: {md.get('price_change_percentage_24h', 0):.2f}%
                                """
                            
                            with st.spinner("Generating portfolio pool analysis..."):
                                breakdown_report = generate_pool_analysis(
                                    pool_data=pool,
                                    token_info=token_info_str,
                                    news_data=raw_news_for_ai,
                                    report_type="breakdown"
                                )
                            
                            st.markdown(breakdown_report)
                
                else:
                    st.error("❌ Failed to generate portfolio. Please try again.")
        else:
            st.warning("⚠️ Token price data needed for portfolio construction")

    with tab3:
        st.markdown("### 📰 Market Intelligence")
        st.markdown(news_summary_display)

        if token_data:
            market_data = token_data["token_data"].get("market_data", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Market Metrics (from CoinGecko)")
                st.write(f"**All-time High:** ${market_data.get('ath', {}).get('usd', 0):,.4f}")
                st.write(f"**All-time Low:** ${market_data.get('atl', {}).get('usd', 0):,.4f}")
                st.write(f"**Market Cap Rank:** #{market_data.get('market_cap_rank', 'N/A')}")

            with col2:
                st.markdown("#### ⏱️ Performance (from CoinGecko)")
                st.write(f"**7d Change:** {market_data.get('price_change_percentage_7d', 0):.2f}%")
                st.write(f"**30d Change:** {market_data.get('price_change_percentage_30d', 0):.2f}%")
                st.write(f"**1y Change:** {market_data.get('price_change_percentage_1y', 0):.2f}%")

    

        # Additional market insights (existing content)
        st.markdown("### 🌐 DeFi Market Context (from DefiLlama Pools)")
        
        if pools:
            # Chain distribution analysis
            chain_distribution = {}
            for pool in pools[:20]:  # Top 20 pools
                chain = pool.get("chain", "Unknown")
                if chain not in chain_distribution:
                    chain_distribution[chain] = {"count": 0, "total_tvl": 0, "avg_apy": []}
                chain_distribution[chain]["count"] += 1
                chain_distribution[chain]["total_tvl"] += pool.get("tvlUsd", 0)
                chain_distribution[chain]["avg_apy"].append(pool.get("apy", 0))
            
            # Calculate averages
            for chain in chain_distribution:
                apys = chain_distribution[chain]["avg_apy"]
                chain_distribution[chain]["avg_apy"] = sum(apys) / len(apys) if apys else 0
            
            st.markdown("#### 🌐 Chain Distribution (Top Opportunities)")
            
            chain_data = []
            for chain, data in chain_distribution.items():
                chain_data.append({
                    "Chain": chain,
                    "Pool Count": data["count"],
                    "Total TVL": f"${data['total_tvl']/1000000:.1f}M",
                    "Avg APY": f"{data['avg_apy']:.2f}%"
                })
            
            df_chains = pd.DataFrame(chain_data)
            st.dataframe(df_chains, use_container_width=True)

if __name__ == "__main__":
    setup_playwright()
    main()
