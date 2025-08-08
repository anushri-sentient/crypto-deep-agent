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
import subprocess
import sys
import re
from report import generate_pool_analysis
from dex_utils import (
    get_dexscreener_pair_data,
    get_pool_analytics,
    format_number,
    format_percentage,
    find_pair_for_pool,
    build_dexscreener_url_from_pair,
)

# Configuration
logging.basicConfig(level=logging.INFO)
load_dotenv()

# API Constants
DEFILLAMA_POOLS_API = "https://yields.llama.fi/pools"
DEFILLAMA_PROTOCOLS_API = "https://api.llama.fi/protocols"
DEFILLAMA_CHART_API = "https://yields.llama.fi/chart"
COINGECKO_API = "https://api.coingecko.com/api/v3"
DEXSCREENER_API = "https://api.dexscreener.com/latest"
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cache for top 50 coins
TOP_50_COINS_CACHE = None
TOP_50_COINS_CACHE_TIME = None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_dexscreener_data(pair_address=None, token_symbol=None):
    """Fetch data from DexScreener API"""
    try:
        if pair_address:
            url = f"{DEXSCREENER_API}/dex/pairs/{pair_address}"
        elif token_symbol:
            url = f"{DEXSCREENER_API}/dex/search?q={token_symbol}"
        else:
            return None
            
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        return data
    except Exception as e:
        logging.error(f"Error fetching DexScreener data: {e}")
        return None

@st.cache_data(ttl=300)
def get_dex_pair_info(pool, debug=False):
    """Get DEX pair information for a pool using enhanced chain mapping"""
    try:
        # Use the enhanced find_pair_for_pool function from dex_utils
        # This properly handles chain mapping and pair resolution
        from dex_utils import find_pair_for_pool
        
        pair_data = find_pair_for_pool(pool, debug=debug)
        return pair_data
    except Exception as e:
        logging.error(f"Error getting DEX pair info: {e}")
        return None

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


@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_pool_chart(pool_id):
    """Fetch historical APY and TVL chart data for a pool"""
    try:
        url = f"{DEFILLAMA_CHART_API}/{pool_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        logging.error(f"Error fetching chart data for pool {pool_id}: {e}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_top_50_coins():
    """Fetch top 50 cryptocurrencies by market cap"""
    try:
        url = f"{COINGECKO_API}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=1&sparkline=false"
        response = requests.get(url)
        response.raise_for_status()
        coins = response.json()
        
        # Create a set of coin symbols for fast lookup
        top_50_symbols = set()
        for coin in coins:
            symbol = coin.get('symbol', '').lower()
            if symbol:
                top_50_symbols.add(symbol)
        
        return top_50_symbols
    except Exception as e:
        logging.error(f"Error fetching top 50 coins: {e}")
        return set()

def is_coin_in_top_50(symbol):
    """Check if a coin symbol is in the top 50 by market cap"""
    global TOP_50_COINS_CACHE, TOP_50_COINS_CACHE_TIME
    
    current_time = datetime.now()
    
    # Refresh cache every hour
    if (TOP_50_COINS_CACHE is None or 
        TOP_50_COINS_CACHE_TIME is None or 
        (current_time - TOP_50_COINS_CACHE_TIME).seconds > 3600):
        
        TOP_50_COINS_CACHE = fetch_top_50_coins()
        TOP_50_COINS_CACHE_TIME = current_time
    
    # Special handling for wrapped tokens
    wrapped_token_mapping = {
        'weth': 'eth',
        'wbtc': 'btc',
        'wmatic': 'matic',
        'wavax': 'avax',
        'wbnb': 'bnb',
        'wftm': 'ftm',
        'wone': 'one',
        'wcelo': 'celo',
        'wmovr': 'movr',
        'wglmr': 'glmr',
        'wrose': 'rose',
        'wcfx': 'cfx',
        'wbtc': 'btc',
        'weth': 'eth',
    }
    
    symbol_lower = symbol.lower()
    
    # Check if it's a wrapped token
    if symbol_lower in wrapped_token_mapping:
        return wrapped_token_mapping[symbol_lower] in TOP_50_COINS_CACHE
    
    return symbol_lower in TOP_50_COINS_CACHE

def get_top_50_coins_display():
    """Get a display string of top 50 coins for UI"""
    global TOP_50_COINS_CACHE
    
    if TOP_50_COINS_CACHE is None:
        TOP_50_COINS_CACHE = fetch_top_50_coins()
    
    # Get first 20 coins for display
    top_coins_list = sorted(list(TOP_50_COINS_CACHE))[:20]
    return ", ".join([coin.upper() for coin in top_coins_list]) + "..."

def debug_high_risk_filtering(pools):
    """Debug function to show high-risk filtering process"""
    if not st.session_state.get("show_debug_logs", False):
        return
    
    st.markdown("### 🔍 High-Risk Filtering Debug")
    
    # Show the actual top 50 coins being used
    top_50_coins = fetch_top_50_coins()
    st.markdown(f"**Top 50 Coins Used for Filtering:** {', '.join(sorted(list(top_50_coins))[:30])}...")
    
    total_high_risk = 0
    filtered_high_risk = 0
    
    for pool in pools:
        if classify_risk(pool) == "High":
            total_high_risk += 1
            symbol = pool.get("symbol", "").lower()
            symbol_parts = re.split(r'[-/\s]+', symbol)
            
            # Check if both tokens are in top 50
            if symbol_parts and len(symbol_parts) >= 2:
                token1 = symbol_parts[0].strip()
                token2 = symbol_parts[1].strip()
                
                if (token1 and token2 and 
                    is_coin_in_top_50(token1) and 
                    is_coin_in_top_50(token2)):
                    
                    filtered_high_risk += 1
                    st.success(f"✅ {pool.get('symbol', 'N/A')} - Both tokens {token1.upper()} and {token2.upper()} are in top 50 (APY: {pool.get('apy', 0):.1f}%)")
                else:
                    missing_tokens = []
                    if not is_coin_in_top_50(token1):
                        missing_tokens.append(token1.upper())
                    if not is_coin_in_top_50(token2):
                        missing_tokens.append(token2.upper())
                    st.error(f"❌ {pool.get('symbol', 'N/A')} - Missing from top 50: {', '.join(missing_tokens)}")
    
    st.info(f"📊 Filtering Results: {filtered_high_risk}/{total_high_risk} high-risk pools contain top 50 coins")

def test_top_50_filtering():
    """Test function to verify top 50 filtering logic"""
    if not st.session_state.get("show_debug_logs", False):
        return
    
    st.markdown("### 🧪 Top 50 Filtering Test")
    
    top_50_coins = fetch_top_50_coins()
    
    # Test cases
    test_pools = [
        "BTC-USDC",  # Should be included (both BTC and USDC are top 50)
        "ETH-USDT",  # Should be included (both ETH and USDT are top 50)
        "WETH-USDC", # Should be included (both WETH and USDC are top 50)
        "CRCL-WETH", # Should be excluded (CRCL is not top 50)
        "SBET-WETH", # Should be excluded (SBET is not top 50)
        "TRUMP-WETH", # Should be excluded (TRUMP is not top 50)
        "SOL-USDC",  # Should be included (both SOL and USDC are top 50)
        "ADA-USDT",  # Should be included (both ADA and USDT are top 50)
    ]
    
    st.markdown("**Test Cases:**")
    for test_pool in test_pools:
        symbol_parts = re.split(r'[-/\s]+', test_pool.lower())
        
        if symbol_parts and len(symbol_parts) >= 2:
            token1 = symbol_parts[0].strip()
            token2 = symbol_parts[1].strip()
            
            if (token1 and token2 and 
                is_coin_in_top_50(token1) and 
                is_coin_in_top_50(token2)):
                st.success(f"✅ {test_pool} - Both tokens {token1.upper()} and {token2.upper()} are in top 50")
            else:
                missing_tokens = []
                if not is_coin_in_top_50(token1):
                    missing_tokens.append(token1.upper())
                if not is_coin_in_top_50(token2):
                    missing_tokens.append(token2.upper())
                st.error(f"❌ {test_pool} - Missing from top 50: {', '.join(missing_tokens)}")
    
    # Show some actual top 50 coins
    st.markdown(f"**Sample Top 50 Coins:** {', '.join(sorted(list(top_50_coins))[:30])}...")

def filter_high_risk_pools(pools):
    """Filter high-risk pools to only include top 50 coins, sorted by APY"""
    high_risk_pools = []
    top_50_coins = fetch_top_50_coins()
    
    for pool in pools:
        if classify_risk(pool) == "High":
            symbol = pool.get("symbol", "").lower()
            
            # More sophisticated token extraction
            # Handle common patterns like "TOKEN1-TOKEN2", "TOKEN1/TOKEN2", "TOKEN1 TOKEN2"
            symbol_parts = re.split(r'[-/\s]+', symbol)
            
            # Only include if BOTH tokens are in top 50
            if symbol_parts and len(symbol_parts) >= 2:
                token1 = symbol_parts[0].strip()
                token2 = symbol_parts[1].strip()
                
                if (token1 and token2 and 
                    is_coin_in_top_50(token1) and 
                    is_coin_in_top_50(token2)):
                    
                    high_risk_pools.append(pool)
                    
                    # Debug logging
                    if st.session_state.get("show_debug_logs", False):
                        st.success(f"✅ {pool.get('symbol', 'N/A')} - Both tokens {token1.upper()} and {token2.upper()} are in top 50 (APY: {pool.get('apy', 0):.1f}%)")
                else:
                    # Debug logging for excluded pools
                    if st.session_state.get("show_debug_logs", False):
                        missing_tokens = []
                        if not is_coin_in_top_50(token1):
                            missing_tokens.append(token1.upper())
                        if not is_coin_in_top_50(token2):
                            missing_tokens.append(token2.upper())
                        st.error(f"❌ {pool.get('symbol', 'N/A')} - Missing from top 50: {', '.join(missing_tokens)}")
    
    # Sort by APY (highest first)
    high_risk_pools.sort(key=lambda x: x.get("apy", 0), reverse=True)
    
    return high_risk_pools

ALLOWED_PROJECTS = [
    "pendle", "compound-v3", "compound-v2", "beefy", "aave-v3", "aave-v2",
    "uniswap-v3", "uniswap-v2", "euler-v2", "curve-dex", "aerodrome-slipstream",
    "aerodrome-v1", "morpho", "kamino", "raydium", "drift", "orca", 
    "ratex", "exponent", "loopscale", "meteora", "jupiter"
]

SIMPLE_STAKING_PROJECTS = [
    "lido", "binance-staked-eth", "rocket-pool", "stakewise-v2",
    "meth-protocol", "liquid-collective", "binance-staked-sol",
    "marinade-liquid-staking", "benqi-staked-avax", "jito-liquid-staking"
]

SIMPLE_STAKING_LP_TOKENS = [
    "steth", "wsteth", "cbeth", "reth", "sfrxeth", "ankreth", "oseth", "lseth", "sweth", "ethx", "bedrock", "teth",
    "stsol", "msol", "jsol", "bsol", "stavax", "savax", "ankrbnb", "stkbnb"
]

# Chain Score Multipliers
CHAIN_SCORE_MULTIPLIERS = {
    "ethereum": 1.25,
    "arbitrum": 1.20,
    "optimism": 1.20,
    "polygon": 1.18,
    "binance": 1.18,
    "avalanche": 1.15,
    "solana": 1.15,
    "base": 1.15,
    "zksync era": 1.12,
    "linea": 1.10,
    "scroll": 1.10,
    "blast": 1.10,
    "fantom": 1.08,
    "gnosis": 1.07,
    "celo": 1.06,
    "kava": 1.05,
    "metis": 1.05,
    "polygon zkevm": 1.08,
    "arbitrum nova": 1.05,
}

def load_custom_css():
    """Load modern minimal CSS styles"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Variables */
    :root {
        --primary-color: #6366f1;
        --primary-light: #818cf8;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --bg-card: #ffffff;
        --border-color: #e5e7eb;
        --border-radius: 12px;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg-secondary);
        color: var(--text-primary);
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        text-align: center;
        color: white;
        box-shadow: var(--shadow-lg);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
    }

    /* Search Container */
    .search-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }

    /* Pool Cards */
    .pool-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .pool-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-color);
    }

    .pool-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        opacity: 0;
        transition: opacity 0.2s ease;
    }

    .pool-card:hover::before {
        opacity: 1;
    }

    /* Card Header */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.3;
    }

    .card-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }

    /* APY Badge */
    .apy-badge {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        min-width: 70px;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }

    .apy-badge.low-apy {
        background: linear-gradient(135deg, var(--success-color) 0%, #34d399 100%);
    }

    .apy-badge.medium-apy {
        background: linear-gradient(135deg, var(--warning-color) 0%, #fbbf24 100%);
        color: white;
    }

    .apy-badge.high-apy {
        background: linear-gradient(135deg, var(--danger-color) 0%, #f87171 100%);
        color: white;
    }

    /* Large APY Badge for more prominence */
    .apy-badge.large-apy {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.75rem 1rem;
        min-width: 90px;
        box-shadow: var(--shadow-md);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    /* Metrics Grid */
    .card-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin: 1rem 0;
    }

    .metric-item {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.75rem;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }

    .metric-item:hover {
        background: #f3f4f6;
        transform: translateY(-1px);
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Badges */
    .badges-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }

    .badge {
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        border: 1px solid var(--border-color);
        background: var(--bg-secondary);
        color: var(--text-secondary);
    }

    .badge.strategy {
        background: rgba(99, 102, 241, 0.1);
        color: var(--primary-color);
        border-color: rgba(99, 102, 241, 0.2);
    }

    .badge.risk-low {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border-color: rgba(16, 185, 129, 0.2);
    }

    .badge.risk-high {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger-color);
        border-color: rgba(239, 68, 68, 0.2);
    }

    /* Action Buttons */
    .card-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
    }

    .action-btn {
        flex: 1;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        text-align: center;
        display: inline-block;
    }

    .action-btn:hover {
        background: var(--primary-color);
        color: white;
        transform: translateY(-1px);
        text-decoration: none;
        box-shadow: var(--shadow-md);
    }

    /* Section Headers */
    .section-header {
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    /* Summary Cards */
    .summary-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .summary-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }

    .summary-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .summary-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }

    .summary-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Token Info Card */
    .token-info-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-lg);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .card-metrics {
            grid-template-columns: 1fr;
        }
        
        .summary-cards {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fadeInUp 0.4s ease-out;
    }

    /* Source Badge */
    .source-badge {
        display: inline-block;
        font-size: 0.7rem;
        color: var(--text-muted);
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        margin-left: 0.5rem;
        text-decoration: none;
        transition: all 0.2s ease;
    }

    .source-badge:hover {
        background: var(--primary-color);
        color: white;
        text-decoration: none;
    }

    /* Streamlit Overrides */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: var(--bg-secondary) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.25rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        margin: 0 !important;
        padding: 0.5rem 1rem !important;
        background: transparent !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: white !important;
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: var(--border-radius) !important;
        overflow: hidden !important;
    }

    /* Metric styling */
    .stMetric {
        background: var(--bg-card) !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Classification Functions (keeping existing logic)
def classify_pool_type(pool):
    """Classify the pool type based on project and symbol."""
    name = pool.get("pool", "").lower()
    project = pool.get("project", "").lower()
    symbol = pool.get("symbol", "").lower()

    if project in SIMPLE_STAKING_PROJECTS:
        return "Staking"
    
    if any(token in symbol for token in SIMPLE_STAKING_LP_TOKENS) and ("lp" in name or "pool" in name):
        return "Staking on LP"
    
    if "aave" in project or "compound" in project or project in ["venus", "morpho"]:
        return "Money Market"
    
    if "uniswap" in project or "curve" in project or "lp" in name or ("usdc" in symbol and "eth" in symbol):
        return "LP Farming"
    
    if project in ["yearn", "beefy", "autofarm", "reaper"]:
        return "Vault"
    
    if pool.get("apy", 0) > 30 and pool.get("tvlUsd", 0) < 5_000_000:
        return "Leveraged"
    
    return "Other"

def classify_risk(pool):
    """Classify risk level based on IL Risk."""
    il_risk = pool.get("ilRisk", "yes").lower()
    return "Low" if il_risk == "no" else "High"

def is_valid_pool(pool, ignore_project_filter=False):
    """Check if a pool meets validity criteria."""
    project = pool.get("project", "").lower()
    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    apy_7d = pool.get("apyBase7d")
    apy_30d = pool.get("apyMean30d")

    if not ignore_project_filter and project not in ALLOWED_PROJECTS and project not in SIMPLE_STAKING_PROJECTS:
        return False
    
    if tvl < 100_000:
        return False
    
    if apy < 0.1:
        return False
    
    if apy_7d is not None and apy_30d is not None and apy_30d > 0.001:
        ratio = apy_7d / apy_30d
        if ratio > 5 or ratio < 0.2:
            return False
            
    return True

def score_pool(pool):
    """Score a pool for ranking based on risk level and metrics."""
    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    chain = pool.get("chain", "").lower()
    risk = classify_risk(pool)
    
    if risk == "High":
        tvl_score = (tvl / 1_000_000) * 0.3
        apy_score = apy * 0.7
        score = apy_score + tvl_score
    else:
        score = apy
    
    chain_multiplier = CHAIN_SCORE_MULTIPLIERS.get(chain, 1.0)
    score *= chain_multiplier

    return score

# Data Fetching Functions (keeping existing logic)
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
    """Fetch yield opportunities for a specific token, applying project filters."""
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

            if (token in symbol or token in project) and is_valid_pool(pool, ignore_project_filter=False):
                filtered_pools.append(pool)

        return filtered_pools

    except Exception as e:
        st.error(f"Error fetching pools for token: {e}")
        return []

# UI Components
def show_source_badge(name: str, href: str = None):
    """Display source badge"""
    if st.session_state.get("show_source_labels", True):
        if href:
            st.markdown(f'<a href="{href}" target="_blank" class="source-badge">📊 {name}</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="source-badge">📊 {name}</span>', unsafe_allow_html=True)

def display_dex_data(dex_data, pool):
    """Display DexScreener data in a clean format"""
    st.markdown("### 📊 DEX Market Data")
    
    # Basic pair info
    pair_name = dex_data.get("pairAddress", "N/A")
    dex_id = dex_data.get("dexId", "N/A")
    chain_id = dex_data.get("chainId", "N/A")
    
    # Get pool chain for comparison
    pool_chain = pool.get("chain", "N/A").title()
    
    # Price data
    price_usd = dex_data.get("priceUsd", "N/A")
    price_change_24h = dex_data.get("priceChange", {}).get("h24", 0)
    
    # Liquidity data
    liquidity = dex_data.get("liquidity", {})
    liquidity_usd = liquidity.get("usd", 0)
    
    # Volume data
    volume = dex_data.get("volume", {})
    volume_24h = volume.get("h24", 0)
    
    # Token info
    base_token = dex_data.get("baseToken", {})
    quote_token = dex_data.get("quoteToken", {})
    
    # Display in a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Market Info**")
        st.markdown(f"- **DEX**: {dex_id}")
        st.markdown(f"- **Pool Chain**: {pool_chain}")
        st.markdown(f"- **DEX Chain**: {chain_id}")
        
        # Show chain match status
        if chain_id.lower() == pool_chain.lower():
            st.success("✅ Chain Match: Pool and DEX data are from the same chain")
        else:
            st.warning(f"⚠️ Chain Mismatch: Pool is on {pool_chain}, DEX data is from {chain_id}")
        
        st.markdown(f"- **Pair**: {base_token.get('symbol', 'N/A')}/{quote_token.get('symbol', 'N/A')}")
        
        if price_usd != "N/A":
            st.metric("Price USD", f"{float(price_usd):.6f}", f"{price_change_24h:+.2f}%")
    
    with col2:
        st.markdown("**Liquidity & Volume**")
        st.metric("Liquidity", f"{format_number(liquidity_usd)}")
        st.metric("Volume (24h)", f"{format_number(volume_24h)}")
    
    # Additional metrics
    if dex_data.get("fdv"):
        st.metric("Fully Diluted Valuation", f"{format_number(dex_data['fdv'])}")
    
    # Links
    st.markdown("**Links**")
    col1, col2 = st.columns(2)
    with col1:
        if dex_data.get("url"):
            st.markdown(f"[View on DexScreener]({dex_data['url']})")
    with col2:
        if dex_data.get("pairAddress"):
            st.markdown(f"[Contract: {dex_data['pairAddress'][:10]}...](https://etherscan.io/address/{dex_data['pairAddress']})")

def create_pool_card(pool, index=0):
    """Create modern pool card using Streamlit components with inline DEX data"""
    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    symbol = pool.get("symbol", "N/A").upper()
    project = pool.get("project", "N/A").title()
    chain = pool.get("chain", "N/A").title()
    pool_id = pool.get("pool", "")
    il_risk = pool.get("ilRisk", "N/A")
    
    strategy = classify_pool_type(pool)
    risk = classify_risk(pool)
    
    # APY badge styling
    if apy < 8:
        apy_class = "low-apy"
    elif apy < 15:
        apy_class = "medium-apy"
    else:
        apy_class = "high-apy"
    
    # Card styling based on strategy and risk
    strategy_class = strategy.lower().replace(" ", "-")
    risk_class = "low-risk" if risk == "Low" else "high-risk"
    
    # Combine strategy and risk for unique card styling
    card_class = f"{strategy_class} {risk_class}"
    
    defillama_url = f"https://defillama.com/yields/pool/{pool_id}"
    
    # Create unique session state keys
    breakdown_key = f"breakdown_{pool_id}_{index}"
    comprehensive_key = f"comprehensive_{pool_id}_{index}"
    
    if breakdown_key not in st.session_state:
        st.session_state[breakdown_key] = False
    if comprehensive_key not in st.session_state:
        st.session_state[comprehensive_key] = False

    # Fetch DEX data inline (cached for performance)
    # Check if global debug mode is enabled
    debug_mode = st.session_state.get("global_dex_debug", False)
    dex_data = get_dex_pair_info(pool, debug=debug_mode)
    

    
    # Create the card using Streamlit components
    with st.container():
        # Card container with custom styling
        st.markdown(f"""
        <div class="pool-card {card_class} fade-in" style="animation-delay: {index * 0.1}s;">
        """, unsafe_allow_html=True)
        
        # Card header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{project}**")
            st.caption(f"{symbol} • {chain}")
        with col2:
            st.markdown(f'<div class="apy-badge {apy_class} large-apy">{apy:.1f}%</div>', unsafe_allow_html=True)
        
        # Badges row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<span class="badge strategy">{strategy}</span>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="badge risk-{risk.lower()}">{risk} Risk</span>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TVL**")
            st.markdown(f"**{format_number(tvl)}**")
        with col2:
            st.markdown("**IL Risk**")
            st.markdown(f"**{il_risk}**")
        
        # DEX Data Section (inline)
        st.markdown("---")
        st.markdown("**📊 DEX Market Data**")
        
        if dex_data:
            # DEX data in compact format
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Price and chain info
                price_usd = dex_data.get("priceUsd", "N/A")
                chain_id = dex_data.get("chainId", "N/A")
                pool_chain = pool.get("chain", "N/A").title()
                
                if price_usd != "N/A":
                    st.metric("Price", f"{float(price_usd):.6f}")
                else:
                    st.metric("Price", "N/A")
                
                # Chain info (without match text)
                st.caption(f"Chain: {chain_id}")
            
            with col2:
                # Volume data
                volume_24h = dex_data.get("volume", {}).get("h24", 0)
                st.metric("Volume 24h", f"{format_number(volume_24h)}")
                
                # DEX info
                dex_id = dex_data.get("dexId", "N/A")
                st.caption(f"DEX: {dex_id}")
            
            with col3:
                # Liquidity data
                liquidity_usd = dex_data.get("liquidity", {}).get("usd", 0)
                st.metric("Liquidity", f"{format_number(liquidity_usd)}")
                
                # Pair info
                base_token = dex_data.get("baseToken", {}).get("symbol", "N/A")
                quote_token = dex_data.get("quoteToken", {}).get("symbol", "N/A")
                st.caption(f"Pair: {base_token}/{quote_token}")
        else:
            # Fallback when no DEX data available
            col1, col2 = st.columns(2)
            with col1:
                st.info("No DEX data available")
            with col2:
                st.markdown(f'<a href="https://dexscreener.com/search?q={symbol}" target="_blank" class="action-btn">🔍 Search on DexScreener</a>', unsafe_allow_html=True)
        
        # Chart Section
        st.markdown("---")
        st.markdown("**📈 APY History**")
        
        # Fetch chart data
        chart_data = fetch_pool_chart(pool_id)
        
        if chart_data and len(chart_data) > 0:
            # Convert to DataFrame for plotting
            df_chart = pd.DataFrame(chart_data)
            # Handle both Unix timestamps and ISO format
            try:
                df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'], unit='s')
            except:
                # If unit='s' fails, try parsing as ISO format
                df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'])
            df_chart = df_chart.sort_values('timestamp')
            
            # Create the chart
            fig = go.Figure()
            
            # Add APY line
            fig.add_trace(go.Scatter(
                x=df_chart['timestamp'],
                y=df_chart['apy'],
                mode='lines',
                name='APY %',
                line=dict(color='#00ff88', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>APY:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Add TVL line on secondary y-axis
            if 'tvlUsd' in df_chart.columns:
                fig.add_trace(go.Scatter(
                    x=df_chart['timestamp'],
                    y=df_chart['tvlUsd'],
                    mode='lines',
                    name='TVL USD',
                    line=dict(color='#ff6b6b', width=2),
                    yaxis='y2',
                    hovertemplate='<b>Date:</b> %{x}<br><b>TVL:</b> $%{y:,.0f}<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=f"{project} APY & TVL History",
                xaxis_title="Date",
                yaxis=dict(
                    title="APY %",
                    side="left",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis2=dict(
                    title="TVL USD",
                    side="right",
                    overlaying="y",
                    showgrid=False
                ),
                height=300,
                margin=dict(l=50, r=50, t=50, b=50),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Show current APY prominently
            current_apy = df_chart['apy'].iloc[-1]
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
                color: white;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                text-align: center;
                font-size: 1.1rem;
                font-weight: 600;
                margin: 1rem 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            ">
                📈 Current APY: {current_apy:.2f}%
            </div>
            """, unsafe_allow_html=True)
            
            # Show some stats
            col1, col2 = st.columns(2)
            with col1:
                max_apy = df_chart['apy'].max()
                st.metric("Max APY", f"{max_apy:.2f}%")
            with col2:
                if 'tvlUsd' in df_chart.columns:
                    current_tvl = df_chart['tvlUsd'].iloc[-1]
                    st.metric("Current TVL", f"${format_number(current_tvl)}")
        else:
            st.info("📊 No historical chart data available for this pool.")
        
        # Actions (simplified - removed DEX button since data is inline)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📈 Quick Analysis", key=f"breakdown_btn_{pool_id}_{index}", help="Fast tactical analysis"):
                st.session_state[breakdown_key] = not st.session_state[breakdown_key]
                st.session_state[comprehensive_key] = False
        with col2:
            if st.button(f"🔍 Deep Analysis", key=f"comprehensive_btn_{pool_id}_{index}", help="Comprehensive research"):
                st.session_state[comprehensive_key] = not st.session_state[comprehensive_key]
                st.session_state[breakdown_key] = False
        
        # External links
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<a href="{defillama_url}" target="_blank" class="action-btn">🔗 View Pool</a>', unsafe_allow_html=True)
        with col2:
            if dex_data and dex_data.get("url"):
                st.markdown(f'<a href="{dex_data["url"]}" target="_blank" class="action-btn">📊 DexScreener</a>', unsafe_allow_html=True)
            else:
                st.markdown(f'<a href="https://dexscreener.com/search?q={symbol}" target="_blank" class="action-btn">📊 DexScreener</a>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display reports if requested
    if st.session_state[breakdown_key]:
        with st.spinner("🔄 Generating quick analysis..."):
            try:
                breakdown_report = generate_pool_analysis(
                    pool_data=pool,
                    token_info="",
                    news_data=[],
                    report_type="breakdown"
                )
                st.info(breakdown_report)
            except Exception as e:
                st.error(f"Error generating analysis: {e}")

    if st.session_state[comprehensive_key]:
        with st.spinner("🌐 Generating comprehensive analysis..."):
            try:
                comprehensive_report = generate_pool_analysis(
                    pool_data=pool,
                    token_info="",
                    news_data=[],
                    report_type="comprehensive"
                )
                st.success(comprehensive_report)
            except Exception as e:
                st.error(f"Error generating analysis: {e}")



def create_token_info_card(token_data):
    """Create modern token info card using Streamlit components"""
    if not token_data:
        return

    token_info = token_data["token_data"]
    market_data = token_info.get("market_data", {})

    current_price = market_data.get("current_price", {}).get("usd", 0)
    price_change_24h = market_data.get("price_change_percentage_24h", 0)
    market_cap = market_data.get("market_cap", {}).get("usd", 0)
    volume_24h = market_data.get("total_volume", {}).get("usd", 0)

    # Token header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(token_info.get('image', {}).get('small', ''), width=64)
    with col2:
        st.markdown(f"## {token_info.get('name', 'N/A')}")
        st.markdown(f"**{token_info.get('symbol', 'N/A').upper()}**")

    # Price and metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"{current_price:,.4f}", f"{price_change_24h:+.2f}%")
    with col2:
        st.metric("Market Cap", format_number(market_cap))
    with col3:
        st.metric("Volume (24h)", format_number(volume_24h))

def create_summary_cards(pools):
    """Create summary statistics cards using Streamlit components"""
    if not pools:
        return
    
    total_pools = len(pools)
    avg_apy = sum(p.get("apy", 0) for p in pools) / total_pools if total_pools > 0 else 0
    total_tvl = sum(p.get("tvlUsd", 0) for p in pools)
    low_risk_count = len([p for p in pools if classify_risk(p) == "Low"])
    chains = len(set(p.get("chain", "") for p in pools))
    
    # Create summary cards using Streamlit components for minimal design
    cols = st.columns(5)
    
    with cols[0]:
        st.metric("🏆 Total Pools", total_pools)
    
    with cols[1]:
        st.metric("📊 Avg APY", f"{avg_apy:.1f}%")
    
    with cols[2]:
        st.metric("💰 Total TVL", format_number(total_tvl))
    
    with cols[3]:
        st.metric("🛡️ Low Risk", low_risk_count)
    
    with cols[4]:
        st.metric("🌐 Chains", chains)

def display_pools_grid(pools, section_name=""):
    """Display pools in a modern grid layout using Streamlit components"""
    if not pools:
        st.info("No pools found. Try adjusting your search criteria or explore other tokens.")
        return

    # Display pools in a grid using columns
    for i in range(0, len(pools), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(pools):
                with cols[j]:
                    create_pool_card(pools[i + j], i + j)

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
            if chain.lower() not in ["ethereum", "arbitrum", "optimism", "polygon", "binance", "solana", "avax", "bnb", "base", "zksync era", "linea", "scroll", "blast"]:
                search_terms.append(chain)
        for project in unique_projects[:2]:
            if project.lower() not in ["uniswap", "aave", "curve", "compound", "lido"]:
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

        news_summary_display = f"**📰 Recent DeFi News for {token.upper()}**\n\n"
        raw_news_data = []
        
        for i, item in enumerate(results[:3], 1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            text = item.get("text", "")

            if text:
                sentences = text.split('. ')
                token_lower = token.lower()
                relevant_section = ""
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

def main():
    """Main application function"""
    st.set_page_config(
        page_title="DeFi Yield Explorer - Modern UI",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_custom_css()

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚀 DeFi Yield Explorer</div>
        <div class="hero-subtitle">Discover high-yield opportunities with AI-powered analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        if "show_source_labels" not in st.session_state:
            st.session_state["show_source_labels"] = True
        if "show_debug_logs" not in st.session_state:
            st.session_state["show_debug_logs"] = False
        if "global_dex_debug" not in st.session_state:
            st.session_state["global_dex_debug"] = False
        
        st.session_state["show_source_labels"] = st.checkbox("Show source badges", value=st.session_state["show_source_labels"])
        st.session_state["show_debug_logs"] = st.checkbox("Show debug info", value=st.session_state["show_debug_logs"])
        st.session_state["global_dex_debug"] = st.checkbox("Debug DEX chain mapping", value=st.session_state["global_dex_debug"], help="Enable detailed logging for DEX chain mapping issues")

    # Search Section
    st.markdown("""
    <div class="search-container">
        <h3 style="margin: 0 0 1rem 0; color: var(--text-primary);">🔍 Find Yield Opportunities</h3>
        <p style="margin: 0 0 1.5rem 0; color: var(--text-secondary);">Enter a token symbol to discover the best DeFi yield farming opportunities</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        token = st.text_input("", placeholder="Enter token symbol (e.g., ETH, USDC, WBTC)", label_visibility="collapsed")
    with col2:
        search_clicked = st.button("🔍 Search Pools", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize variables
    token_data = None
    pools = []
    news_summary_display = ""
    raw_news_data = []

    # Fetch data when token is provided
    if token:
        with st.spinner(f"🔄 Finding opportunities for {token.upper()}..."):
            token_data = fetch_coingecko_token_data(token)
            pools = fetch_yield_opportunities(token)
            news_summary_display, raw_news_data = get_news_summary(token, pools)

        # Sort pools by score
        pools.sort(key=score_pool, reverse=True)

        # Display token info if available
        if token_data:
            create_token_info_card(token_data)
            show_source_badge("CoinGecko", "https://www.coingecko.com/en/api")

    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Pool Explorer", "📰 Market Intel", "🌱 Trending"])

    with tab1:
        if not token:
            st.info("🎯 Ready to Find High Yields? Enter a token symbol above to discover the best DeFi opportunities")
        elif not pools:
            st.warning(f"❌ No suitable yield pools found for {token.upper()}. Try a different token.")
        else:
            # Summary Cards
            create_summary_cards(pools)
            
            # Categorize pools by risk
            risk_groups = {"Low": [], "High": []}
            for pool in pools:
                risk = classify_risk(pool)
                if risk == "Low":
                    risk_groups["Low"].append(pool)
            
            # Apply special filtering for high-risk pools (top 100 coins only)
            high_risk_filtered = filter_high_risk_pools(pools)
            risk_groups["High"] = high_risk_filtered
            
            # Debug filtering process if enabled
            debug_high_risk_filtering(pools)
            test_top_50_filtering()

            # Display pools by risk category
            for risk_level in ["Low", "High"]:
                risk_pools = risk_groups[risk_level]
                if not risk_pools:
                    continue

                emoji = "🟢" if risk_level == "Low" else "🔴"
                
                if risk_level == "High":
                    st.markdown(f"## {emoji} {risk_level} Risk Pools - Both Tokens Top 50 ({len(risk_pools)} pools)")
                    st.info(f"🔍 High-risk pools are filtered to only include pools where BOTH tokens are in the top 50 by market cap, sorted by highest APY. Top coins: {get_top_50_coins_display()}")
                else:
                    st.markdown(f"## {emoji} {risk_level} Risk Pools ({len(risk_pools)} pools)")

                # Show/hide toggle
                show_key = f"show_all_{risk_level.lower()}"
                if show_key not in st.session_state:
                    st.session_state[show_key] = False

                display_count = len(risk_pools) if st.session_state[show_key] else min(6, len(risk_pools))
                display_pools = risk_pools[:display_count]
                
                display_pools_grid(display_pools, f"{risk_level}_risk")
                
                # Load more button
                if len(risk_pools) > 6:
                    remaining = len(risk_pools) - display_count
                    if remaining > 0:
                        if st.button(f"📥 Show {remaining} More {risk_level} Risk Pools", key=f"load_more_{risk_level}"):
                            st.session_state[show_key] = True
                            st.rerun()
                    elif st.session_state[show_key]:
                        if st.button(f"📤 Show Less", key=f"show_less_{risk_level}"):
                            st.session_state[show_key] = False
                            st.rerun()

    with tab2:
        st.markdown("## 📰 Market Intelligence")
        
        if not token:
            st.info("📰 Market News & Analysis - Enter a token to see latest market intelligence")
        else:
            if news_summary_display:
                st.markdown(news_summary_display)
                show_source_badge("EXA.ai", "https://exa.ai/")
            
            # Market context if pools exist
            if pools:
                st.markdown("### 🌐 Market Context")
                chain_distribution = {}
                for pool in pools[:20]:
                    chain = pool.get("chain", "Unknown")
                    if chain not in chain_distribution:
                        chain_distribution[chain] = {"count": 0, "total_tvl": 0, "avg_apy": []}
                    chain_distribution[chain]["count"] += 1
                    chain_distribution[chain]["total_tvl"] += pool.get("tvlUsd", 0)
                    chain_distribution[chain]["avg_apy"].append(pool.get("apy", 0))
                
                for chain in chain_distribution:
                    apys = chain_distribution[chain]["avg_apy"]
                    chain_distribution[chain]["avg_apy"] = sum(apys) / len(apys) if apys else 0
                
                chain_data = []
                for chain, data in chain_distribution.items():
                    chain_data.append({
                        "Chain": chain,
                        "Pools": data["count"],
                        "Total TVL": f"{data['total_tvl']/1000000:.1f}M",
                        "Avg APY": f"{data['avg_apy']:.1f}%"
                    })
                
                df_chains = pd.DataFrame(chain_data)
                st.dataframe(df_chains, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("## 🌱 Trending Opportunities")
        
        st.info("🚀 Coming Soon - Trending pools and ecosystem insights will be available here")

if __name__ == "__main__":
    setup_playwright()

    main()