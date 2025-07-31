# utils.py

import os
import logging
import requests
import streamlit as st  # For potential use with @st.cache_data

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Constants ---
DEFAULT_TVL = 10_000_000  # $10 Million
DEFAULT_PROTOCOLS = [
    "aave", "compound", "uniswap", "yearn", "beefy",
    "curve", "balancer", "lido", "convex", "frax"
]
STABLECOINS = {
    'USDC', 'USDT', 'DAI', 'FRAX', 'LUSD',
    'TUSD', 'BUSD', 'USDP', 'GUSD', 'MIM', 'sUSD'
}

DEFAULT_CHAINS = ["Ethereum", "Arbitrum", "Optimism", "Base", "Polygon", "Solana"]

# --- API Setup ---
LLAMAFI_API_KEY = os.getenv("LLAMAFI_API_KEY")
if not LLAMAFI_API_KEY:
    logger.warning("LLAMAFI_API_KEY not found in environment variables.")
LLAMAFI_HEADERS = {"Authorization": f"Bearer {LLAMAFI_API_KEY}"} if LLAMAFI_API_KEY else {}

# --- API Fetchers ---

def get_historical_apy_tvl(pool_id):
    try:
        url = f"https://pro-api.llama.fi/chart/{pool_id}"
        response = requests.get(url, headers=LLAMAFI_HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        logger.error(f"Error fetching historical APY/TVL for pool {pool_id}: {e}")
        return []

def get_historical_borrow_apy(pool_id):
    try:
        url = f"https://pro-api.llama.fi/yields/chartLendBorrow/{pool_id}"
        response = requests.get(url, headers=LLAMAFI_HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        logger.error(f"Error fetching historical borrow data for pool {pool_id}: {e}")
        return []

# @st.cache_data(ttl=300)
def get_pools_old():
    try:
        url = "https://pro-api.llama.fi/yields/poolsOld"
        response = requests.get(url, headers=LLAMAFI_HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        logger.error(f"Error fetching /poolsOld data: {e}")
        return []

# @st.cache_data(ttl=300)
def get_borrow_pools():
    try:
        url = "https://pro-api.llama.fi/yields/poolsBorrow"
        response = requests.get(url, headers=LLAMAFI_HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        logger.error(f"Error fetching borrow pool data: {e}")
        return []

# @st.cache_data(ttl=300)
def get_perps_metrics():
    try:
        url = "https://pro-api.llama.fi/yields/perps"
        response = requests.get(url, headers=LLAMAFI_HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        logger.error(f"Error fetching perps data: {e}")
        return []

# @st.cache_data(ttl=300)
def get_lsd_rates():
    try:
        url = "https://pro-api.llama.fi/yields/lsdRates"
        response = requests.get(url, headers=LLAMAFI_HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching LSD rates: {e}")
        return {}


# --- Pool Utilities ---

def get_display_name(pool):
    project = pool.get('project', 'Unknown Project')
    symbol = pool.get('symbol', 'Unknown Pool')

    if project.lower() in [p.lower() for p in DEFAULT_PROTOCOLS]:
        return symbol
    return f"{symbol}"

def get_risk_score(pool):
    score = 0
    sigma = pool.get('sigma', 0)
    predictions = pool.get('predictions', {})

    score += sigma * 0.3

    if pool.get('outlier', False):
        score += 50

    if predictions.get('predictedClass') == 'Down':
        score += 20

    if pool.get('ilRisk', '').lower() == 'yes':
        score += 10

    if pool.get('rewardTokens'):
        score += 5

    apy = pool.get('apy', 0)
    if apy > 100:
        score += (apy / 100) * 0.5
    elif apy > 50:
        score += 2

    return round(score, 2)

def is_trending(pool):
    apy_1d = pool.get('apyPct1D', 0) or 0
    apy_7d = pool.get('apyPct7D', 0) or 0
    apy_30d_mean = pool.get('apyMean30d', 0) or 0
    current_apy = pool.get('apy', 0)
    predicted_class = pool.get('predictions', {}).get('predictedClass', '')

    return (
        apy_1d > 0 and
        apy_7d > 0 and
        predicted_class in ['Stable/Up', 'Up'] and
        current_apy > apy_30d_mean * 1.1
    )

def get_stability_badge_text(pool):
    apy_pct_1d = pool.get('apyPct1D')
    apy_pct_7d = pool.get('apyPct7D')
    current_apy = pool.get('apy', 0)

    if apy_pct_1d is None or apy_pct_7d is None or current_apy == 0:
        return None

    change_1d = abs(apy_pct_1d / 100 * current_apy)
    change_7d = abs(apy_pct_7d / 100 * current_apy)

    if change_1d < current_apy * 0.05 and change_7d < current_apy * 0.10:
        return 'Stable APY'
    elif pool.get('rewardTokens'):
        return 'Reward-Heavy'
    else:
        return 'Volatile APY'

def is_no_il_risk(pool):
    il_risk = pool.get('ilRisk', '').lower()
    exposure = pool.get('exposure', '').lower()
    symbol = pool.get('symbol', '').upper()

    if il_risk == 'no' or exposure == 'single':
        return True

    if '-' in symbol:
        tokens = symbol.split('-')
        if all(token in STABLECOINS for token in tokens):
            return True

    return False

def get_pool_url(pool):
    chain = pool.get('chain')
    project = pool.get('project')
    pool_id = pool.get('pool')

    if chain and project and pool_id:
        return f"https://defillama.com/yields/pool/{pool_id}"
    return None
