#!/usr/bin/env python3
"""
Crypto DeepSearch Agent DataLake - Streamlit UI
A comprehensive web interface for interacting with the datalake
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
import openai
import json
import subprocess
from playwright.sync_api import sync_playwright

# NEW: Import modules for report generation and utilities
import report_generator as rg
import utils as ut # Alias for convenience

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../agent'))

# Configure page
st.set_page_config(
    page_title="Crypto DeepSearch Agent DataLake",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
    .modal-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.5); z-index: 9999; display: flex; align-items: center; justify-content: center;
    }
    .modal-content {
        background: white; padding: 2rem; border-radius: 10px; max-width: 700px; width: 90vw; max-height: 90vh; overflow-y: auto;
        box-shadow: 0 4px 32px rgba(0 4px 32px rgba(0,0,0,0.2);
        position: relative;
    }
    .modal-close {
        position: absolute; top: 10px; right: 20px; font-size: 2rem; cursor: pointer; color: #888;
    }
    /* Improved Table CSS for Responsiveness */
    .responsive-table table {
        width: 100%;
        table-layout: auto; /* Use auto for better column sizing */
        border-collapse: collapse;
    }
    .responsive-table th, .responsive-table td {
        word-wrap: break-word;
        overflow-wrap: break_word;
        font-size: 0.9rem;
        padding: 8px;
        text-align: left;
    }
    .responsive-table td a {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)
def classify_pool(pool):
    """Classifies a pool into a specific strategy type based on its properties."""
    
    dex_projects = {"uniswap-v2", "uniswap-v3", "sushiswap", "balancer", "curve", "pancakeswap"}
    money_markets = {"compound", "aave", "venus", "cream", "justlend", "morpho", "ironbank"}
    looping_projects = {"yearn", "rari", "alpha", "gearbox", "notional", "instadapp", "lodestar"}

    project = str(pool.get("project", "")).lower()
    tokens = pool.get("underlyingTokens") or []
    il_risk = str(pool.get("ilRisk", "no")).lower()
    pool_meta = str(pool.get("poolMeta", "")).lower()
    exposure = str(pool.get("exposure", "")).lower()
    symbol = str(pool.get("symbol", ""))
    stablecoin = pool.get("stablecoin", False)
    api_category = str(pool.get("category", "")).lower()

    # 1. Stablecoin
    if stablecoin:
        return "Stablecoin"

    # 2. Money Market
    if any(mm in project for mm in money_markets):
        return "Money Market"

    # 3. Pendle Tokens
    if "pendle" in project:
        if "pt" in pool_meta:
            return "Pendle PT Token"
        if "yt" in pool_meta:
            return "YT Token"
        return "Pendle Pool"

    # 4. Looping / Leveraged Strategies
    if any(lp in project for lp in looping_projects) or exposure in {"looping", "leveraged"}:
        return "Looping / Leveraged"

    # 5. DEX Pool
    if any(dex in project for dex in dex_projects):
        if len(tokens) >= 2 and il_risk in {"yes", "no"}:
            return "DEX Pool"

    # 6. Single Asset Staking / Lending
    if exposure == "single" or api_category in {"lending", "staking", "cdp", "lsd"}:
        if len(tokens) == 1 or (symbol and '-' not in symbol):
            return "Single Asset Staking/Lending"

    # 7. Fallback to non-generic API category
    if api_category and api_category not in {"yield", "other"}:
        return api_category.capitalize()

    # 8. Final catch-all
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


def normalize_query(query):
    """Normalize and expand query with synonyms and common aliases"""
    # Token aliases
    token_aliases = {
        'eth': 'ethereum', 'btc': 'bitcoin', 'matic': 'polygon',
        'avax': 'avalanche', 'sol': 'solana', 'ada': 'cardano',
        'dot': 'polkadot', 'link': 'chainlink', 'uni': 'uniswap',
        'usdc': 'usdc', 'usdt': 'usdt', 'dai': 'dai', 'frax': 'frax' # Ensure common stablecoins are present
    }

    # Chain aliases
    chain_aliases = {
        'eth': 'ethereum', 'arb': 'arbitrum', 'op': 'optimism',
        'poly': 'polygon', 'bsc': 'binance smart chain', 'base': 'base'
    }

    # Strategy aliases
    strategy_aliases = {
        'staking': 'single asset', 'lending': 'single asset',
        'farming': 'liquidity providing', 'lp': 'liquidity providing',
        'yield farming': 'liquidity providing', 'liquid staking': 'single asset',
        'lsd': 'single asset', 'money market': 'single asset',
        'leveraged': 'looping', 'looping': 'looping',
        'dex': 'dex pool'
    }

    normalized = query.lower()

    # Apply aliases
    for alias, full_name in {**token_aliases, **chain_aliases, **strategy_aliases}.items():
        # Use regex to replace whole words only to avoid partial matches (e.g., 'eth' in 'ethereum')
        normalized = normalized.replace(alias, full_name)

    return normalized

def fallback_parse_query(query):
    """Fallback keyword-based query parsing when OpenAI API is unavailable"""
    import re
    import json
    query_lower = query.lower()

    filters = {
        "chain": None,
        "token": None,
        "apy_min": None,
        "apy_max": None,
        "risk_level": None,
        "stablecoin": None,
        "exposure": None,
        "project": None,
        "trending": None
    }

    # Chain detection
    chains = ["ethereum", "arbitrum", "optimism", "polygon", "base", "solana", "avalanche"]
    for chain in chains:
        if chain in query_lower or (len(chain) > 3 and chain[:3] in query_lower):
            filters["chain"] = chain.capitalize()
            break

    # Token detection
    tokens = ["eth", "btc", "usdc", "usdt", "dai", "weth", "wbtc", "matic", "avax", "sol", "uni", "link"]
    for token in tokens:
        if token in query_lower:
            filters["token"] = token.upper()
            break

    # APY detection
    apy_match = re.search(r'(\d+)%?\s*apy', query_lower)
    if apy_match:
        filters["apy_min"] = int(apy_match.group(1))
    elif "high" in query_lower and ("apy" in query_lower or "yield" in query_lower):
        filters["apy_min"] = 15

    # Risk level detection
    if any(word in query_lower for word in ["safe", "low risk", "stable", "no impermanent loss", "no il"]):
        filters["risk_level"] = "no"
    elif "high risk" in query_lower:
        filters["risk_level"] = "high"
    elif "medium risk" in query_lower:
        filters["risk_level"] = "medium"

    # Stablecoin detection
    if "stablecoin" in query_lower or "stable coin" in query_lower:
        filters["stablecoin"] = True

    # Exposure detection
    if any(word in query_lower for word in ["staking", "lending", "single asset", "liquid staking", "money market"]):
        filters["exposure"] = "single"
    elif any(word in query_lower for word in ["lp", "liquidity", "farming", "multi", "dex pool"]):
        filters["exposure"] = "multi"

    # Project detection
    projects = ["aave", "compound", "uniswap", "yearn", "beefy", "curve", "balancer", "lido", "convex", "frax", "pendle"]
    for project in projects:
        if project in query_lower:
            filters["project"] = project
            break

    # Trending detection
    if any(word in query_lower for word in ["trending", "hot", "popular", "new"]):
        filters["trending"] = True

    return json.dumps(filters)

def get_filters_from_query(query):
    """Enhanced query parsing with fallback mechanisms"""
    # Normalize query first
    normalized_query = normalize_query(query)

    PROMPT_TEMPLATE = '''
You are an expert assistant that converts user queries about DeFi yields into structured JSON filters.

User query: "{query}"
Normalized query: "{normalized_query}"

Instructions: Extract the relevant filter parameters from the query and return a JSON object with the following keys:
- chain (e.g., "Ethereum", "Arbitrum", "Polygon", "Optimism", "Base", "Solana", or null if not specified)
- token (e.g., "USDT", "USDC", "ETH", "BTC", or null if not specified. Use uppercase for token symbols.)
- apy_min (minimum APY %, integer or null if not specified)
- apy_max (maximum APY %, integer or null if not specified)
- risk_level (e.g., "no" for low risk/no impermanent loss, "high", "medium", or null if not specified)
- stablecoin (true if only stablecoin pools, false if not, or null if not specified)
- exposure: Use "single" for single asset staking/lending (no impermanent loss), "multi" for LPs/farms (has impermanent loss). If "liquid staking" or "money market" is mentioned, it implies "single". If "dex pool" is mentioned, it implies "multi".
- project (specific protocol like "aave", "compound", "uniswap", "lido", or null if not specified. Use lowercase.)
- trending (true if user wants trending/hot pools, or null)

Keyword mappings:
- "high yield", "high apy" → apy_min: 15
- "stable", "safe", "low risk", "no impermanent loss", "no il" → risk_level: "no"
- "stablecoin", "stable coin" → stablecoin: true
- "trending", "hot", "popular", "new" → trending: true
- "staking", "lending", "single asset", "liquid staking", "money market" → exposure: "single"
- "LP", "liquidity", "farming", "multi asset", "dex pool" → exposure: "multi"

If a field is not mentioned, return null for that field.

Examples:
Input: Show ETH staking options
Output: {{{{"chain": "Ethereum", "token": "ETH", "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": "single", "project": null, "trending": null}}}}
Input: Show me LP pools for USDC
Output: {{{{"chain": null, "token": "USDC", "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": "multi", "project": null, "trending": null}}}}
Input: Find high APY opportunities on Arbitrum
Output: {{{{"chain": "Arbitrum", "token": null, "apy_min": 15, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": null, "project": null, "trending": null}}}}
Input: Show me pools with no impermanent loss
Output: {{{{"chain": null, "token": null, "apy_min": null, "apy_max": null, "risk_level": "no", "stablecoin": null, "exposure": null, "project": null, "trending": null}}}}
Input: Show me trending Aave pools
Output: {{{{"chain": null, "token": null, "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": null, "project": "aave", "trending": true}}}}
Input: find pools on polygon
Output: {{{{"chain": "Polygon", "token": null, "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": null, "project": null, "trending": null}}}}
Input: Show me liquid staking derivatives
Output: {{{{"chain": null, "token": null, "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": "single", "project": null, "trending": null}}}}
Input: Show me money market pools
Output: {{{{"chain": null, "token": null, "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": "single", "project": null, "trending": null}}}}
Input: Show me dex pools
Output: {{{{"chain": null, "token": null, "apy_min": null, "apy_max": null, "risk_level": null, "stablecoin": null, "exposure": "multi", "project": null, "trending": null}}}}


Return only the JSON object.
'''
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY not set. Using fallback keyword-based parsing.")
        return fallback_parse_query(query)

    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = PROMPT_TEMPLATE.format(query=query, normalized_query=normalized_query)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in DeFi yield analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0
        )
    except Exception as e:
        st.warning(f"OpenAI API error: {e}. Using fallback parsing.")
        return fallback_parse_query(query)

    import re
    text = response.choices[0].message.content
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        filters_json = match.group(1)
        import json
        try:
            filters = json.loads(filters_json)
            # Post-process: map 'exposure': 'staking' to 'single' (redundant if prompt is good, but safe)
            if filters.get('exposure') == 'staking':
                filters['exposure'] = 'single'
            return json.dumps(filters)
        except Exception:
            return filters_json
    else:
        st.error("OpenAI did not return a valid JSON object.")
        return fallback_parse_query(query)

def apply_filters(data, filters):
    """Enhanced filter application with fuzzy matching and new filter types"""
    def is_stablecoin_token(symbol):
        # List of common stablecoins
        stablecoins = {'USDC', 'USDT', 'DAI', 'TUSD', 'BUSD', 'USDP', 'GUSD', 'LUSD', 'FRAX', 'EURS', 'USDD', 'USDE', 'USDN', 'MIM', 'sUSD', 'alUSD', 'crvUSD'}
        if not symbol:
            return False
        tokens = symbol.upper().replace('-', ' ').split() # Handle "USDC-USDT" -> "USDC", "USDT"
        return any(t in stablecoins for t in tokens)

    def matches(pool):
        # Chain filtering with fuzzy matching
        if filters.get("chain"):
            pool_chain = pool.get("chain", "").lower()
            filter_chain = filters["chain"].lower()
            # Check if filter chain is contained in pool chain or vice versa for broad matching
            if filter_chain not in pool_chain and pool_chain not in filter_chain:
                return False

        # Token filtering with symbol matching
        if filters.get("token"):
            pool_symbol_parts = [p.strip() for p in pool.get("symbol", "").upper().split('-')]
            filter_token = filters["token"].upper()
            if filter_token not in pool_symbol_parts:
                return False

        # APY filtering
        if filters.get("apy_min") is not None and (pool.get("apy") is None or pool["apy"] < filters["apy_min"]):
            return False
        if filters.get("apy_max") is not None and (pool.get("apy") is None or pool["apy"] > filters["apy_max"]):
            return False

        # Risk level filtering
        if filters.get("risk_level") == "no" and ut.is_no_il_risk(pool) is False: # Using ut.is_no_il_risk
            return False
        # General risk level matching (medium, high, etc.)
        if filters.get("risk_level") and filters["risk_level"] != "no":
            pool_risk = pool.get("risk", "").lower()
            filter_risk = filters["risk_level"].lower()
            if pool_risk != filter_risk: # Exact match for other risk levels
                return False

        # Stablecoin logic
        if filters.get("stablecoin") is True:
            pool_symbol = pool.get("symbol", "")
            if not is_stablecoin_token(pool_symbol):
                return False

        # Exposure filtering
        if filters.get("exposure"):
            pool_exposure = pool.get("exposure", "")
            if pool_exposure is None: # If pool has no exposure data, assume it doesn't match
                return False
            
            # Special handling for "single" and "multi"
            if filters["exposure"] == "single" and pool_exposure != "single":
                return False
            elif filters["exposure"] == "multi" and pool_exposure != "multi": # Only include multi-asset if filter is multi
                return False

        # Project/protocol filtering
        if filters.get("project"):
            pool_project = pool.get("project", "").lower()
            filter_project = filters["project"].lower()
            if filter_project not in pool_project: # Fuzzy match for projects
                return False

        # Trending filtering
        if filters.get("trending") is True and not ut.is_trending(pool): # Using ut.is_trending
            return False

        return True
    return [pool for pool in data if matches(pool)]

# Helper to get DeFiLlama yields
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_defillama_yields():
    url = "https://yields.llama.fi/pools"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()["data"]

# <<< START: MODIFIED FUNCTION TO ADD STRATEGY CLASSIFICATION >>>
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_all_pools():
    """Fetches pools from DeFiLlama and classifies them using the classify_pool function."""
    pools = get_defillama_yields()
    for pool in pools:
        # Add the 'strategy' key to each pool dictionary
        pool['strategy'] = classify_pool(pool)
    return pools
# <<< END: MODIFIED FUNCTION >>>


# Build filter options from the full pool list
all_pools = get_all_pools()
ALL_TOKENS = sorted(list(set(c for p in all_pools for c in (p.get('symbol', '') or '').replace('-', ' ').split() if c)))
ALL_CHAINS = sorted(list(set(p.get('chain', 'Unknown') for p in all_pools if p.get('chain'))))
ALL_PROTOCOLS = sorted(list(set(p.get('project', 'Unknown') for p in all_pools if p.get('project'))))
ALL_CATEGORIES = sorted(list(set(p.get('category', 'Unknown') for p in all_pools if p.get('category'))))
ALL_ATTRIBUTES = ['All', 'single_exposure']

# Default safe chains
DEFAULT_CHAINS = ["Ethereum", "Arbitrum", "Optimism", "Base", "Polygon", "Solana"]

def prompt_filter_ui(preselect=None):
    include_all_chains = st.checkbox(
        "Include all chains (may be riskier)", value=False,
        help="By default, only well-known chains are included. Check to include all chains."
    )
    if include_all_chains:
        chains = ALL_CHAINS
    else:
        chains = [c for c in DEFAULT_CHAINS if c in ALL_CHAINS]
    st.write(f"Chains used for filtering: {chains}")
    tokens = st.multiselect("Token(s)", options=ALL_TOKENS, default=preselect.get('tokens', []) if preselect else [])
    yield_type = st.radio("Type of yield", ["Single Token Staking", "LP Farming", "High-Risk High-Yield"], index=["Single Token Staking", "LP Farming", "High-Risk High-Yield"].index(preselect.get('yield_type', 'Single Token Staking')) if preselect else 0)
    platform = st.selectbox("Platform", options=[''] + ALL_PROTOCOLS, format_func=lambda x: x or 'All', index=([''] + ALL_PROTOCOLS).index(preselect.get('platform', '') if preselect and preselect.get('platform', '') in ([''] + ALL_PROTOCOLS) else '') if preselect else 0)
    categories = st.multiselect("Category", options=ALL_CATEGORIES, default=preselect.get('categories', []) if preselect else [])
    attribute = st.selectbox("Attribute", options=ALL_ATTRIBUTES, index=ALL_ATTRIBUTES.index(preselect.get('attribute', 'All')) if preselect else 0)
    min_apy = st.slider("Minimum APY (%)", 0.0, 100.0, preselect.get('min_apy', 0.0) if preselect else 0.0)
    risk = st.selectbox("Risk appetite", options=['', 'Low', 'Medium', 'High'], format_func=lambda x: x or 'All', index=(['', 'Low', 'Medium', 'High'].index(preselect.get('risk', '')) if preselect else 0))
    exposure = st.selectbox(
        "Exposure Type",
        options=["All", "Single (no IL risk)", "Multi (LP, has IL risk)"],
        help="Single = single asset staking/lending. Multi = LPs/farms (impermanent loss risk)."
    )
    return chains, tokens, yield_type, platform, categories, attribute, min_apy, risk, exposure

def apply_prompt_filters(pools, chains, tokens, yield_type, platform, categories, attribute, min_apy, risk, exposure):
    filtered = []
    for p in pools:
        if chains and p.get('chain') not in chains:
            continue
        # Check if any token in the pool matches any selected token
        if tokens:
            pool_tokens = [t.strip().upper() for t in (p.get('symbol', '') or '').replace('-', ' ').split()]
            if not any(t.upper() in pool_tokens for t in tokens):
                continue

        # Strategy-based filtering
        pool_strategy = p.get('strategy', 'Other')
        if yield_type == 'Single Token Staking':
            # Check if it's explicitly single asset or a money market
            if pool_strategy not in ["Single Asset Staking/Lending", "Money Market"]:
                continue
        elif yield_type == 'LP Farming':
            # Check if it's explicitly a DEX pool or has multi exposure
            if pool_strategy != "DEX Pool" and p.get('exposure') != 'multi':
                continue
        elif yield_type == 'High-Risk High-Yield':
            # This category is more about allowing higher risk, not strictly filtering by it
            # If a specific risk is selected, it will be handled by the general risk filter below
            pass
        
        if platform and p.get('project', '').lower() != platform.lower(): # Case-insensitive project match
            continue
        if categories and p.get('category') not in categories:
            continue
        if attribute == 'single_exposure' and not ut.is_no_il_risk(p): # Using ut.is_no_il_risk
            continue
        if min_apy is not None and (p.get('apy') or 0) < min_apy:
            continue
        if risk:
            # Map simplified risk to API's 'risk' field or calculate from 'ilRisk'
            if risk == 'Low':
                if p.get('ilRisk', '').lower() == 'yes': # If there's IL, it's not low risk
                    continue
                # Also consider risk score for 'Low'
                if ut.get_risk_score(p) > 50: # Threshold for what is "low" risk score
                    continue
            elif risk == 'Medium':
                if p.get('risk', '').lower() != 'medium': # Check API's risk field
                    continue
            elif risk == 'High':
                if p.get('risk', '').lower() != 'high': # Check API's risk field
                    continue
        
        # Exposure filter
        if exposure == "Single (no IL risk)" and p.get('exposure') != 'single':
            continue
        if exposure == "Multi (LP, has IL risk)" and p.get('exposure') != 'multi':
            continue
        filtered.append(p)
    # Always sort by APY descending
    filtered = sorted(filtered, key=lambda x: x.get('apy', 0), reverse=True)
    return filtered

def is_dead(pool):
    apy = pool.get('apy', 0) or 0
    apy_base = pool.get('apyBase', 0) or 0
    return (abs(apy) < 0.001 and abs(apy_base) < 0.001)

def make_clickable(name, url):
    if url:
        return f'<a href="{url}" target="_blank">{name}</a>'
    return name

def colored_badge(text, color):
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:8px;font-size:90%;white-space:nowrap;">{text}</span>'

def get_combined_badges_for_table(pool):
    """Returns a list of badge texts for display in data_editor (not HTML)."""
    badges = []
    if ut.is_trending(pool): # Using ut.is_trending
        badges.append('🔥 Trending')
    
    stability = ut.get_stability_badge_text(pool) # Using ut.get_stability_badge_text
    if stability:
        badges.append(stability)
    
    return ", ".join(badges) # Join with comma for display in a single table cell

# <<< START: MODIFIED FUNCTION TO DISPLAY STRATEGY IN TABLE >>>
def display_selectable_pool_results(pools_to_display, key_suffix=""):
    """
    Displays pools in a selectable st.data_editor table and manages report generation.
    """
    if not pools_to_display:
        st.warning('No pools match your criteria.')
        return

    # Sort by APY for initial display
    pools_to_display = sorted(pools_to_display, key=lambda x: x.get('apy', 0), reverse=True)

    # Highest APY and Lowest Risk suggestions
    if pools_to_display:
        highest_apy_pool = max(pools_to_display, key=lambda x: x.get('apy', 0))
        lowest_risk_pool = min(pools_to_display, key=lambda x: ut.get_risk_score(x))

        st.markdown('<div style="background-color:#eaf6ff;padding:12px 16px;border-radius:10px;margin-bottom:8px;">'
                    + '🏆 <b>Highest APY Suggestion:</b> '
                    + make_clickable(ut.get_display_name(highest_apy_pool), ut.get_pool_url(highest_apy_pool))
                    + f' — <b>{highest_apy_pool.get("apy", 0):.2f}% APY</b> '
                    + colored_badge('Highest APY', '#1f77b4')
                    + '</div>', unsafe_allow_html=True)

        st.markdown('<div style="background-color:#eaffea;padding:12px 16px;border-radius:10px;margin-bottom:16px;">'
                    + '🛡️ <b>Lowest Risk Suggestion:</b> '
                    + make_clickable(ut.get_display_name(lowest_risk_pool), ut.get_pool_url(lowest_risk_pool))
                    + f' — Risk Score: <b>{ut.get_risk_score(lowest_risk_pool)}</b> (lower = safer) '
                    + colored_badge('Lowest Risk', '#2ca02c')
                    + '</div>', unsafe_allow_html=True)

    # Prepare data for st.data_editor
    df_data = []
    pool_id_map = {}
    for i, p in enumerate(pools_to_display):
        unique_pool_id = f"{p.get('chain', 'N/A')}-{p.get('project', 'N/A')}-{p.get('pool', i)}"
        pool_id_map[unique_pool_id] = p
        df_data.append({
            '_pool_id': unique_pool_id,
            'Select': False,
            'Project': p.get('project'),
            'Strategy': p.get('strategy', 'N/A'), # NEW: Add strategy column data
            'Symbol': ut.get_display_name(p),
            'Pool Id': p.get('pool'),
            'Chain': p.get('chain'),
            'APY %': f"{p.get('apy', 0):.2f}",
            'TVL $': f"${p.get('tvlUsd', 0):,.0f}",
            'Risk Score': f"{ut.get_risk_score(p):.2f}",
            'Badges': get_combined_badges_for_table(p)
        })
    df = pd.DataFrame(df_data)

    st.markdown("### Filtered Pools (Select for Report)")
    st.write("Select one or more pools using the checkboxes to generate a comprehensive investment report.")

    edited_df = st.data_editor(
        df,
        key=f"data_editor_{key_suffix}",
        column_config={
            "_pool_id": None, # Hide the internal ID column
            "Select": st.column_config.CheckboxColumn("Select for Report", default=False, width="small"),
            "Project": st.column_config.Column("Project", width="small"),
            "Strategy": st.column_config.Column("Strategy", width="small"), # NEW: Configure strategy column
            "Symbol": st.column_config.Column("Symbol", width="medium"),
            "Chain": st.column_config.Column("Chain", width="small"),
            "APY %": st.column_config.Column("APY %", width="small"),
            "TVL $": st.column_config.Column("TVL $", width="small"),
            "Risk Score": st.column_config.Column("Risk Score", width="small"),
            "Badges": st.column_config.Column("Badges", width="medium"),
            "Pool Id": st.column_config.Column("Pool Id", width="small")
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_order=["Select", "Project", "Strategy", "Symbol", "Chain", "APY %", "TVL $", "Risk Score", "Badges", "Pool Id"] # NEW: Define column order
    )

    selected_pool_ids = edited_df[edited_df['Select']]['_pool_id'].tolist()
    
    st.session_state.selected_pools_for_report = [pool_id_map[pool_id] for pool_id in selected_pool_ids if pool_id in pool_id_map]

    if st.session_state.get('selected_pools_for_report'):
        st.markdown(f"**{len(st.session_state.selected_pools_for_report)} pool(s) selected for report.**")
        
        if st.button("Clear Selection", key=f"clear_selection_btn_{key_suffix}"):
            st.session_state.selected_pools_for_report = []
            if 'generated_report' in st.session_state:
                del st.session_state.generated_report
            st.rerun()

        if st.button("Generate Comprehensive Investment Report", key=f"generate_report_btn_{key_suffix}"):
            st.session_state.generated_report = None 

            if not st.session_state.selected_pools_for_report:
                st.error("No pools selected for report generation. Please select at least one pool.")
            else:
                with st.spinner("Collecting detailed context and generating AI report... This may take a while. (Approx. 30-60 seconds per pool)"):
                    full_contexts = []
                    for pool in st.session_state.selected_pools_for_report:
                        context = rg.collect_full_pool_context(pool)
                        full_contexts.append(context)
                    
                    report,context = rg.generate_comprehensive_report_with_ai(full_contexts)
                    st.session_state.generated_report = report
                    st.session_state.used_context = context
                
    else:
        st.write("No pools selected for report.")
        if 'generated_report' in st.session_state:
            del st.session_state.generated_report

    if st.session_state.get('generated_report'):
        st.markdown("---")
        st.subheader("Comprehensive Investment Report by AI Agent")
        st.warning("Disclaimer: This report is generated by an AI model based on available data. It should not be considered financial advice. Always do your own research (DYOR) before making investment decisions.")
        st.markdown(st.session_state.generated_report, unsafe_allow_html=True)
        st.download_button(
            label="Download Report as Markdown",
            data=st.session_state.generated_report,
            file_name="defi_investment_report.md",
            mime="text/markdown",
            key=f"download_report_btn_{key_suffix}"
        )
    if st.session_state.get('used_context'):
        with st.expander("🔍 Show raw context data used for report generation"):
            st.json(st.session_state.used_context)

# <<< END: MODIFIED FUNCTION >>>


def main():

    st.markdown('<h1 class="main-header">🔎 Crypto DeepSearch Agent DataLake</h1>', unsafe_allow_html=True)

    gemini_nlp_section()

def gemini_nlp_section():
    st.title("🔎 DeFi Yield Explorer (Natural Language)")
    st.markdown("""
Welcome to the DeFi Yield Explorer! Enter your DeFi yield question in plain English below, or try one of the example queries. Results are always sorted by APY.

**How is Risk Score calculated?**
- **Risk Score = 0.5 × (outlier) + 0.3 × sigma + 0.2 × (predictedClass == 'Down')**
- Lower = safer. Outlier, high sigma, or predictedClass 'Down' increases risk.

**Badges in table:**
- 🔥 Trending: APY is up, predictedClass is Stable/Up, and APY > 30d mean
- Stable APY: APY is stable over 30 days
- Volatile APY: APY is volatile
- Reward-Heavy: Pool relies on reward tokens
    """)
    st.markdown("**Example queries (click to use):**")
    example_queries = [
        "Show high APY pools",
        "Show me single asset staking pools for USDC",
        "Find multi-token LP pools for ETH",
        "Show only stablecoin pools on Arbitrum",
        "Show me trending pools",
        "Show pools with lowest risk",
        "Show blue chip pools with TVL > $50M",
        "Show me pools with no impermanent loss",
        "Show me Uniswap pools on Ethereum",
        "find pools on polygon",
        "Show me liquid staking derivatives",
        "High yield opportunities under 5% risk",
        "Show me money market pools", # NEW EXAMPLE
        "Show me dex pools" # NEW EXAMPLE
    ]
    # Create clickable example queries in columns
    cols = st.columns(3)
    for i, q in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(q, key=f"example_{i}", help="Click to use this query"):
                st.session_state.selected_query = q
                st.rerun()
    # Handle selected query from buttons
    query_placeholder = st.empty()
    if 'selected_query' in st.session_state:
        query = query_placeholder.text_input("Enter your DeFi query:", value=st.session_state.selected_query, key="defi_query_input")
        # Clear the selected query after using it
        del st.session_state.selected_query
    else:
        query = query_placeholder.text_input("Enter your DeFi query:", value=st.session_state.get("defi_query_input", ""), key="defi_query_input")

    # Add a clear button for the query
    if query:
        if st.button("Clear Query", key="clear_query"):
            st.session_state["defi_query_input"] = ""
            query = ""
            if 'generated_report' in st.session_state:
                del st.session_state.generated_report
            if 'selected_pools_for_report' in st.session_state:
                del st.session_state.selected_pools_for_report
            st.rerun()

    all_pools = get_all_pools()
    
    only_blue_chip = st.checkbox(
        "Show only blue chip protocols (Aave, Compound, Uniswap, Yearn, Beefy, Curve, Balancer, Lido, Convex)", value=True, key="blue_chip_checkbox"
    )
    min_tvl = ut.DEFAULT_TVL if st.checkbox("Require TVL > $10M (recommended)", value=True, key="min_tvl_checkbox") else 0 # Using ut.DEFAULT_TVL

    def is_blue_chip(project):
        return any(proto in project.lower() for proto in ut.DEFAULT_PROTOCOLS) # Using ut.DEFAULT_PROTOCOLS

    def strict_filter(p):
        if only_blue_chip and not is_blue_chip(p['project']):
            return False
        if p['chain'] not in DEFAULT_CHAINS: # Use DEFAULT_CHAINS here for strict filter
            return False
        if p.get('tvlUsd', 0) < min_tvl:
            return False
        return True

    if not query:
        st.info("Showing default results: blue chip protocols, TVL > $10M, on major chains, sorted by APY. Enter a query above or adjust filters.")
        default_filtered = [p for p in all_pools if strict_filter(p) and not is_dead(p)]
        display_selectable_pool_results(default_filtered, key_suffix="default_display")
    else:
        with st.spinner("Parsing your query with AI..."):
            filters_json = get_filters_from_query(query)
        try:
            filters = json.loads(filters_json)
        except Exception as e:
            st.error(f"Could not parse filters from AI: {e}\nRaw output: {filters_json}")
            return

        with st.expander("Show Parsed Filters (from AI)", expanded=False):
            st.json(filters)

        # Apply general filters first, then the user's explicit query filters
        base_filtered_pools = [p for p in all_pools if (not only_blue_chip or is_blue_chip(p['project'])) and p.get('tvlUsd', 0) >= min_tvl]
        
        # Apply chain filter from query first, or default chains if not specified
        if not filters.get("chain") or filters.get("chain") in ["null", None, "None"]:
            # If AI didn't specify chain, filter by DEFAULT_CHAINS for display
            base_filtered_pools = [p for p in base_filtered_pools if p['chain'] in DEFAULT_CHAINS]
        else:
            # If AI specified a chain, strictly filter by it
            base_filtered_pools = [p for p in base_filtered_pools if p['chain'].lower() == filters["chain"].lower()]

        # Now apply the remaining filters parsed from the query
        filtered_pools = apply_filters(base_filtered_pools, filters)
        
        # Final filter out dead pools
        filtered_pools = [p for p in filtered_pools if not is_dead(p)]
        
        display_selectable_pool_results(filtered_pools, key_suffix="query_display")


if __name__ == "__main__":
    setup_playwright()
    main()