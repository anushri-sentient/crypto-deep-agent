# utils.py

import streamlit as st # Still needed for st.cache_data decorator if applied here

# Constants that define "blue chip" protocols and default TVL
DEFAULT_PROTOCOLS = ["aave", "compound", "uniswap", "yearn", "beefy", "curve", "balancer", "lido", "convex"]
DEFAULT_TVL = 10_000_000  # $10M
# utils.py

DEFAULT_TVL = 10_000_000 # $10 Million
DEFAULT_PROTOCOLS = ["aave", "compound", "uniswap", "yearn", "beefy", "curve", "balancer", "lido", "convex", "frax"]

def get_display_name(pool):
    """Constructs a user-friendly display name for a pool."""
    project = pool.get('project', 'Unknown Project')
    symbol = pool.get('symbol', 'Unknown Pool')
    pool_meta = pool.get('poolMeta')

    if project and symbol:
        # Example: Aave USDC, Compound ETH
        if project.lower() in [p.lower() for p in DEFAULT_PROTOCOLS]:
            return f"{symbol}"
        # General fallback for other projects
        return f"{symbol}"
    elif symbol:
        return symbol
    elif project:
        return project
    return "N/A"

def get_risk_score(pool):
    """Calculates a simple risk score for a pool."""
    # Example risk score calculation based on provided data schema
    # Lower score is safer
    score = 0
    predictions = pool.get('predictions', {})
    
    # High sigma (volatility) increases risk
    sigma = pool.get('sigma', 0)
    score += sigma * 0.3

    # Outliers (unusual behavior) increase risk
    if pool.get('outlier', False):
        score += 50 # Add a significant penalty for outliers

    # Predicted class 'Down' (negative price prediction) increases risk
    if predictions.get('predictedClass') == 'Down':
        score += 20 # Add a penalty for predicted downturn

    # Impermanent Loss risk
    if pool.get('ilRisk', '').lower() == 'yes':
        score += 10 # Add penalty for IL risk

    # Reward tokens imply emissions risk, could be temporary
    if pool.get('rewardTokens'):
        score += 5

    # If APY is very high, it might imply higher risk (can be adjusted)
    apy = pool.get('apy', 0)
    if apy > 100:
        score += (apy / 100) * 0.5 # Scale penalty by APY
    elif apy > 50:
        score += 2

    # Normalize a bit or cap
    return round(score, 2)

def is_trending(pool):
    """Determines if a pool is trending up."""
    apy_1d = pool.get('apyPct1D', 0) if pool.get('apyPct1D', 0) is not None else 0
    apy_7d = pool.get('apyPct7D', 0) if pool.get('apyPct7D', 0) is not None else 0
    apy_30d_mean = pool.get('apyMean30d', 0) if pool.get('apyMean30d', 0) is not None else 0
    current_apy = pool.get('apy', 0)
    predicted_class = pool.get('predictions', {}).get('predictedClass', '')

    # Considered trending if APY has increased recently, predicted class is not 'Down',
    # and current APY is significantly higher than 30-day mean
    return (apy_1d > 0 and apy_7d > 0 and 
            predicted_class in ['Stable/Up', 'Up'] and
            current_apy > apy_30d_mean * 1.1 # 10% higher than 30d average
           )

def get_stability_badge_text(pool):
    """Returns a badge text indicating APY stability."""
    apy_pct_1d = pool.get('apyPct1D')
    apy_pct_7d = pool.get('apyPct7D')
    
    if apy_pct_1d is None or apy_pct_7d is None:
        return None # Not enough data

    # Calculate absolute changes over 1D and 7D relative to current APY
    current_apy = pool.get('apy', 0)
    if current_apy == 0:
        return None # Cannot assess stability if APY is zero

    # Convert percentage change to actual change
    change_1d = abs(apy_pct_1d / 100 * current_apy)
    change_7d = abs(apy_pct_7d / 100 * current_apy)

    # Threshold for stability (e.g., less than 5% fluctuation relative to current APY)
    stability_threshold_1d = current_apy * 0.05
    stability_threshold_7d = current_apy * 0.10

    if change_1d < stability_threshold_1d and change_7d < stability_threshold_7d:
        return 'Stable APY'
    elif pool.get('rewardTokens'):
        return 'Reward-Heavy'
    else:
        return 'Volatile APY'

def is_no_il_risk(pool):
    """Checks if a pool has no impermanent loss risk based on data."""
    # Pools are typically single-asset or stablecoin LPs
    il_risk = pool.get('ilRisk', '').lower()
    exposure = pool.get('exposure', '').lower()
    
    if il_risk == 'no': # Explicitly stated no IL risk
        return True
    
    if exposure == 'single': # Single asset pools have no IL
        return True

    # Check for common stablecoin symbols in pool
    symbol = pool.get('symbol', '').upper()
    stablecoins = {'USDC', 'USDT', 'DAI', 'FRAX', 'LUSD', 'TUSD', 'BUSD', 'USDP', 'GUSD', 'MIM', 'sUSD'}
    
    # If it's a multi-token pool, check if all are stablecoins
    if '-' in symbol:
        tokens_in_pool = symbol.split('-')
        if all(t in stablecoins for t in tokens_in_pool):
            return True # Stablecoin LP, generally considered low/no IL

    return False

def get_pool_url(pool):
    """Constructs a DeFiLlama pool URL for direct linking."""
    chain = pool.get('chain')
    project = pool.get('project')
    pool_id = pool.get('pool')
    
    if chain and project and pool_id:
        return f"https://defillama.com/yields/pool/{pool_id}"
    return None