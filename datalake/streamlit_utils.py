"""
Streamlit UI utility functions for DeFi DataLake app.
"""

def is_trending(pool):
    """Return True if the pool is considered trending (less strict logic)."""
    apy_pct_7d = pool.get('apyPct7D', 0) or 0
    predicted_class = pool.get('predictions', {}).get('predictedClass')
    apy = pool.get('apy', 0) or 0
    apy_mean = pool.get('apyMean30d', 0) or 0
    if (apy_pct_7d > 0.1 and apy > apy_mean):
        return True
    if predicted_class in ["Stable", "Up"] and apy > apy_mean:
        return True
    return False

def is_dead(pool):
    """Return True if the pool is considered dead (no APY and no APY base)."""
    apy = pool.get('apy', 0) or 0
    apy_base = pool.get('apyBase', 0) or 0
    return (apy == 0 and apy_base == 0)

def get_display_name(pool):
    """Get a display name for the pool."""
    for key in ['poolMeta', 'symbol', 'project']:
        val = pool.get(key)
        if val and val != 'None':
            return val
    return 'Unknown'

def get_display_category(pool):
    """Get a display category for the pool."""
    val = pool.get('category')
    if val and val != 'None':
        return val
    exposure = pool.get('exposure')
    if exposure == 'single':
        return 'Single Asset'
    if exposure == 'multi':
        return 'LP/Multi-Asset'
    return 'Unknown'

def get_display_risk(pool):
    """Get a display risk label for the pool."""
    val = pool.get('risk')
    if val and val != 'None':
        return val
    il_risk = pool.get('ilRisk')
    if il_risk == 'no':
        return 'No IL Risk'
    if il_risk == 'yes':
        return 'Has IL Risk'
    return 'Unknown'

def make_clickable(name, url):
    """Return HTML for a clickable link if url is present, else just the name."""
    if url:
        return f'<a href="{url}" target="_blank">{name}</a>'
    return name

def colored_badge(text, color):
    """Return HTML for a colored badge."""
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:8px;font-size:90%">{text}</span>'

def get_stability_badge(pool):
    """Return a stability badge label for the pool."""
    apy = pool.get('apy', 0) or 0
    apy_mean = pool.get('apyMean30d', 0) or 0
    sigma = pool.get('sigma', 0) or 0
    reward_tokens = pool.get('rewardTokens')
    if abs(apy - apy_mean) < 0.5 and sigma < 1 and (not reward_tokens or len(reward_tokens) == 0):
        return "Stable APY"
    elif sigma >= 1 or abs(apy - apy_mean) > 1:
        return "Volatile APY"
    elif reward_tokens and len(reward_tokens) > 0:
        return "Reward-Heavy"
    return ""

def get_risk_score(pool):
    """Calculate a risk score for the pool (lower is safer)."""
    outlier = 1 if pool.get('outlier') else 0
    sigma = pool.get('sigma', 0) or 0
    predicted_class = pool.get('predictions', {}).get('predictedClass')
    down = 1 if predicted_class == "Down" else 0
    return round(0.5 * outlier + 0.3 * sigma + 0.2 * down, 2)

def is_no_il_risk(pool):
    """Return True if the pool has no impermanent loss risk."""
    il_risk = pool.get('ilRisk')
    if il_risk is not None:
        return str(il_risk).lower() == 'no'
    return False

def get_pool_url(pool):
    """Get a URL for the pool if available."""
    if pool.get('url'):
        return pool['url']
    if pool.get('pool'):
        return f"https://defillama.com/yields/pool/{pool['pool']}"
    return None 