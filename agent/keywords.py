# Comprehensive list of default financial & crypto symbols
DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
    'SPY', 'QQQ', 'BTC', 'ETH', 'DJI', 'NASDAQ'
]

# Comprehensive financial and crypto keyword set
DEFAULT_KEYWORDS = [
    'stock market', 'bull market', 'bear market', 'market crash',
    'earnings report', 'fed rate', 'inflation', 'recession',
    'cryptocurrency', 'bitcoin', 'ethereum', 'trading', 'investment',
    'hedge fund', 'hedge funds', 'wall street', 'finance', 'financial',
    'economy', 'economic', 'banking', 'bank', 'banks', 'mortgage',
    'interest rate', 'interest rates', 'federal reserve', 'fed',
    'sec', 'securities', 'bonds', 'treasury', 'dollar', 'currency',
    'oil price', 'gas price', 'commodities', 'futures', 'options',
    'dividend', 'dividends', 'ipo', 'merger', 'acquisition',
    'bankruptcy', 'default', 'bailout', 'stimulus', 'tax', 'taxes'
]

def get_stock_symbols(custom_symbols=None):
    if custom_symbols:
        return DEFAULT_SYMBOLS + [s for s in custom_symbols if s and s not in DEFAULT_SYMBOLS]
    return DEFAULT_SYMBOLS

def get_financial_keywords(custom_keywords=None):
    if custom_keywords:
        return [k for k in custom_keywords if k]  # Only use custom
    return DEFAULT_KEYWORDS
