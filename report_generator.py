import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv
import openai  # Keep import for completeness, though not used in the provided functions
import json
import google.generativeai as genai
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

import subprocess

# Import utility functions
from utils import get_display_name, get_pool_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Corrected parameter name
EXA_API_KEY = os.getenv("EXA_API_KEY")
CRAWLER_API = os.getenv("CRAWLER_API_ENDPOINT", "https://crawl4ai.dev.sentient.xyz/crawl_direct")

# Load CoinGecko API key and prepare headers
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_HEADERS = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}

def get_token_market_data(token_symbol):
    logger.info(f"Fetching market data for token: {token_symbol}")
    token_symbol = token_symbol.lower()

    try:
        coin_list_resp = requests.get(
            "https://pro-api.coingecko.com/api/v3/coins/list",
            headers=COINGECKO_HEADERS
        )
        coin_list_resp.raise_for_status()
        coin_list = coin_list_resp.json()
    except Exception as e:
        logger.error(f"Error fetching coin list from CoinGecko: {e}")
        return {'price': None, 'market_cap': None, '24h_change': None, 'sentiment': 'unknown'}

    coin_id = next((coin['id'] for coin in coin_list if coin['symbol'] == token_symbol), None)
    if not coin_id:
        logger.warning(f"Token symbol '{token_symbol}' not found in CoinGecko")
        return {'price': None, 'market_cap': None, '24h_change': None, 'sentiment': 'unknown'}

    try:
        market_data_resp = requests.get(
            f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}",
            params={"localization": "false", "market_data": "true"},
            headers=COINGECKO_HEADERS
        )
        market_data_resp.raise_for_status()
        data = market_data_resp.json()

        price = data['market_data']['current_price']['usd']
        market_cap = data['market_data']['market_cap']['usd']
        change_24h = data['market_data']['price_change_percentage_24h']

        sentiment = 'positive' if change_24h > 1 else 'negative' if change_24h < -1 else 'neutral'

        return {
            'price': price,
            'market_cap': market_cap,
            '24h_change': change_24h / 100.0 if change_24h else None,
            'sentiment': sentiment
        }
    except Exception as e:
        logger.error(f"Error fetching market data for {coin_id}: {e}")
        return {'price': None, 'market_cap': None, '24h_change': None, 'sentiment': 'unknown'}

def simulate_news_search_fallback(entity):
    return f"Fallback news for {entity}: No API key configured."

def get_news_from_exa_api(query_term):
    if not EXA_API_KEY:
        logger.warning("EXA_API_KEY not set, using fallback news.")
        return simulate_news_search_fallback(query_term)

    try:
        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "query": query_term,
                "text": True
            }
        )
        response.raise_for_status()
        news_json = response.json()

        results = news_json.get("results", [])
        if not results:
            return f"No recent news found for '{query_term}'."

        return "\n".join(
            f"- {item.get('title', 'No Title')} ({item.get('url', 'No URL')})"
            for item in results
        )
    except Exception as e:
        logger.error(f"Error fetching news from Exa API for '{query_term}': {e}")
        return simulate_news_search_fallback(query_term)


CRAWLER_API = "https://crawl4ai.dev.sentient.xyz/crawl_direct"
CRAWLER_AUTH_TOKEN = os.getenv("CRAWLER_AUTH_TOKEN")
CRAWLER_HEADER = os.getenv("CRAWLER_HEADER")

async def crawl_pool_url_async(pool_url):
    config = CrawlerRunConfig(
        wait_until="networkidle"  # Wait for network to be idle (page loaded)
    )
    api_key = os.getenv("CRAWLER_AUTH_TOKEN")
    async with AsyncWebCrawler(api_key=api_key) as crawler:
        result = await crawler.arun(url=pool_url, config=config)        
        
        return result.cleaned_html

def crawl_pool_url(pool_url):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # Streamlit case
        return asyncio.ensure_future(crawl_pool_url_async(pool_url))
    else:
        return loop.run_until_complete(crawl_pool_url_async(pool_url))


def gemini_summarize_html(crawled_content, pool_name=""):
    print("Crawled content for summarization:", crawled_content)
    if not crawled_content:
        return "No content to summarize."

    prompt = (
        f"You are a DeFi expert. Summarize the following text.\n"
        f"Focus on the project's purpose, features, risks/benefits, and algorithmic predictions etc.\n"
        f"\n{crawled_content}"
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt, generation_config={"temperature": 0.3})
        print("Gemini response:", response.text)
        return response.text
    except Exception as e:
        logger.error(f"Gemini summarization error: {e}")
        return "Error summarizing content."

# @st.cache_data(ttl=3600)
def collect_full_pool_context(pool):
    display_name = get_display_name(pool)
    context = {k: v for k, v in pool.items() if k not in ['priceChart', 'apyChart', 'tvlChart', 'predictedClass']}
    context['display_name'] = display_name

    pool_url = get_pool_url(pool)
    # Check if pool_url is valid before attempting to crawl
    if pool_url and (pool_url.startswith('http://') or pool_url.startswith('https://')):
        crawled_content = crawl_pool_url(pool_url)
        context['crawled_summary'] = gemini_summarize_html(crawled_content, display_name)
    else:
        context['crawled_summary'] = "No valid pool URL available for crawling."

    token_symbols = [t.strip() for t in (pool.get('symbol', '') or '').replace('-', ' ').split()]
    context['token_market_data'] = {t: get_token_market_data(t) for t in token_symbols}

    # Ensure 'project' is a string before adding to entities
    project_entity = pool.get('project')
    if not isinstance(project_entity, str):
        project_entity = None  # Exclude if not a string

    entities = set(filter(None, [project_entity] + token_symbols + [pool.get('chain')]))
    context['news_and_sentiment'] = {e: get_news_from_exa_api(e) for e in entities}

    print("context given to report generator", context)

    return context

def generate_comprehensive_report_with_ai(selected_pools_context):
    if not selected_pools_context:
        return "No pool data available."

    # The comprehensive risk template
    risk_template = """
🔍 Yield Staking Risk Template

I. 📊 Key Dimensions of Risk
Dimension	Description
APY Sustainability	Is the APY likely to hold or decay over time? Are rewards emissions sustainable?
Volatility Risk	Is the token price stable? How sensitive is the APY to price movement?
Protocol Risk	Smart contract vulnerabilities, chain-level issues, or governance attacks.
Impermanent Loss (IL)	Relevant in DEX LPs. Measures loss due to asset divergence.
Reward Risk	Are rewards paid in volatile assets? Can they be dumped easily?
TVL vs APY Signal	If APY drops when TVL rises, yield may be unsustainable.

II. 🪙 Asset/Strategy Types & Their Typical Risks

1. DEX Liquidity Pools (e.g., ETH/USDC on Uniswap)
Risks:
IL (depends on correlation)
Rewards heavy APY (can decay)
Smart contract risk (DEX + strategy)
How to Evaluate:
Check if APY includes token rewards (farmed tokens, inflationary?)
Look at TVL/APY trend on DeFiLlama

2. Money Markets (Aave, Compound)
Risks:
Liquidation risk if borrowing
Collateral depreciation
APY usually stable, but low
How to Evaluate:
Use Aave Risk Dashboard
Look at asset correlation (deposit vs collateral)
Understand LTV and liquidation thresholds

3. PT/Fixed Yield Tokens (e.g., Pendle PTs)
Risks:
Illiquidity or pricing inefficiencies
Smart contract risk (Pendle or protocol source)
How to Evaluate:
Check maturity dates
See how they trade vs par value
Yield rate is fixed, but price floats – monitor slippage

4. YT/Variable Yield Tokens (Pendle YTs)
Risks:
Yield collapse if underlying APY dries up
High volatility in pricing
How to Evaluate:
Use Pendle Analytics
Track historical yield trends of underlying assets
Watch for reward cliffs or halvenings

5. Looping (e.g., on Aave or Alpaca)
Risks:
High liquidation risk due to leverage
APY can flip negative due to borrow rate increasing
Oracle manipulation (chain or asset-specific)
How to Evaluate:
Use real-time health factor trackers
Test leverage limits in simulators
Follow Chainlink oracle status if relevant

6. Stablecoins
Risks:
Asset risk (depeg)
Counterparty or protocol backing (USDT, DAI, FRAX)
How to Evaluate:
Use DeFi dashboards for peg monitoring
Assess collateral type (algorithmic, overcollateralized, etc.)

III. 🛠️ How to Analyze Yield Opportunities

1. Check TVL vs APY
If TVL ↑ and APY ↓ → Emission likely fixed or unsustainable
Use: DeFiLlama Yields, look at charts

2. Examine Reward Composition
Are rewards in native protocol tokens? → Dump risk
Stablecoin or ETH rewards are more sustainable

3. Chain & Protocol Audit
Has the smart contract been audited? By whom?
Look at past exploits or TVL drop-offs

4. Token & Chain Stability
Is the chain congested? Prone to downtime?
Does the asset have sufficient liquidity across major DEXs?

"""

    parts = [
        "You are a highly experienced DeFi investment analyst specializing in risk assessment.",
        "Your task is to provide a comprehensive report for the following DeFi pools.",
        "Utilize the provided 'Yield Staking Risk Template' to structure your analysis and identify key opportunities and risks for each pool.",
        "For each pool, first provide a general summary, then identify its likely 'Asset/Strategy Type' and provide a detailed risk assessment based on the 'Key Dimensions of Risk'.",
        "Conclude each pool's analysis with a summary table, similar to the 'Sample Risk Summary Table' provided in the template, assessing the pool against relevant risk dimensions (e.g., APY Sust., Volatility, IL Risk, Protocol Risk). Use emojis like ✅ for low risk, ❗ for medium, ❌ for high, and 'N/A' for not applicable. Dont show the Sample Risk table",
        "Finally, provide a concluding section comparing all selected pools.",
        f"\n{risk_template}\n"
    ]

    for idx, ctx in enumerate(selected_pools_context, 1):
        name = ctx.get("display_name", f"Pool #{idx}")
        parts.append(f"\n## {name} Analysis")
        parts.append(f"**General Pool Information:**")
        parts.append(f"- **Project:** {ctx.get('project', 'N/A')}")
        parts.append(f"- **Chain:** {ctx.get('chain', 'N/A')}")
        parts.append(f"- **Symbol:** {ctx.get('symbol', 'N/A')}")
        parts.append(f"- **APY (Current):** {ctx.get('apyBase', 'N/A')} (Base), {ctx.get('apyReward', 'N/A')} (Reward), {ctx.get('apy', 'N/A')} (Total)")
        parts.append(f"- **TVL (USD):** {ctx.get('tvlUsd', 'N/A')}")
        parts.append(f"- **URL:** {get_pool_url(ctx) or 'N/A'}\n")

        parts.append(f"**Website Summary:**\n{ctx.get('crawled_summary', 'N/A')}\n")

        tokens = ctx.get("token_market_data", {})
        if tokens:
            parts.append("**Token Market Data:**")
            for symbol, data in tokens.items():
                price_str = f"${data['price']:.4f}" if data['price'] is not None else "N/A"
                market_cap_str = f"${data['market_cap']:,}" if data['market_cap'] is not None else "N/A"
                change_str = f"{data['24h_change']:.2%}" if data['24h_change'] is not None else "N/A"
                parts.append(
                    f"- **{symbol.upper()}:** Price = {price_str}, Market Cap = {market_cap_str}, "
                    f"24h Change = {change_str}, Sentiment = {data.get('sentiment', 'unknown')}"
                )
            parts.append("")  # Add a blank line for spacing

        news = ctx.get("news_and_sentiment", {})
        if news:
            parts.append("**Recent News & Sentiment:**")
            for entity, content in news.items():
                parts.append(f"- **{entity}:** {content}")
            parts.append("")  # Add a blank line for spacing

        # Explicit instruction for AI to perform risk analysis based on the template
        parts.append("---")
        parts.append(f"**Risk Assessment for {name}:**")
        parts.append("Based on the provided data and the 'Yield Staking Risk Template', here is the risk profile for this pool:")
        parts.append("Please identify the closest 'Asset/Strategy Type' and then provide an assessment against the 'Key Dimensions of Risk', summarizing with a table and notes.")


    # Add a concluding section instruction
    parts.append("\n## Overall Comparative Analysis and Conclusion")
    parts.append("Provide a brief comparative analysis of all the selected pools, highlighting their main strengths, weaknesses, and a final recommendation or insight based on their risk profiles.")


    prompt = "\n\n".join(parts)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt, generation_config={"temperature": 0.7, "max_output_tokens": 4000}) # Increased max tokens for comprehensive report
        return response.text,selected_pools_context
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"Error generating report: {e}"