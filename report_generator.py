import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

CRAWLER_AUTH_TOKEN = os.getenv("CRAWLER_AUTH_TOKEN")

async def crawl_pool_url_async(pool_url):
    config = CrawlerRunConfig(
        wait_until="networkidle"
    )
    api_key = CRAWLER_AUTH_TOKEN
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
        return asyncio.ensure_future(crawl_pool_url_async(pool_url))
    else:
        return loop.run_until_complete(crawl_pool_url_async(pool_url))

def gpt4o_summarize_html(crawled_content, pool_name=""):
    if not crawled_content:
        return "No content to summarize."

    prompt = (
        f"You are a DeFi expert. Summarize the following text.\n"
        f"Focus on the project's purpose, features, risks/benefits, and algorithmic predictions etc.\n\n"
        f"{crawled_content}"
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
        logger.error(f"GPT-4o summarization error: {e}")
        return "Error summarizing content."

def collect_full_pool_context(pool):
    display_name = get_display_name(pool)
    context = {k: v for k, v in pool.items() if k not in ['priceChart', 'apyChart', 'tvlChart', 'predictedClass']}
    context['display_name'] = display_name

    pool_url = get_pool_url(pool)
    if pool_url and (pool_url.startswith('http://') or pool_url.startswith('https://')):
        crawled_content = crawl_pool_url(pool_url)
        context['crawled_summary'] = gpt4o_summarize_html(crawled_content, display_name)
    else:
        context['crawled_summary'] = "No valid pool URL available for crawling."

    token_symbols = [t.strip() for t in (pool.get('symbol', '') or '').replace('-', ' ').split()]
    context['token_market_data'] = {t: get_token_market_data(t) for t in token_symbols}

    project_entity = pool.get('project')
    if not isinstance(project_entity, str):
        project_entity = None

    entities = set(filter(None, [project_entity] + token_symbols + [pool.get('chain')]))
    context['news_and_sentiment'] = {e: get_news_from_exa_api(e) for e in entities}

    return context

def generate_comprehensive_report_with_ai(selected_pools_context):
    if not selected_pools_context:
        return "No pool data available."

    risk_template = """
🔍 Yield Staking Risk Template

I. 📊 Key Dimensions of Risk
... (your risk dimensions here — truncated for brevity)
"""

    parts = [
        "You are a highly experienced DeFi investment analyst specializing in risk assessment.",
        "Your task is to provide a comprehensive report for the following DeFi pools.",
        "Utilize the provided 'Yield Staking Risk Template' to structure your analysis and identify key opportunities and risks for each pool.",
        "For each pool, provide a summary, identify the strategy type, and assess risks.",
        "Conclude with a comparative analysis.",
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
            parts.append("")

        news = ctx.get("news_and_sentiment", {})
        if news:
            parts.append("**Recent News & Sentiment:**")
            for entity, content in news.items():
                parts.append(f"- **{entity}:** {content}")
            parts.append("")

        parts.append("---")
        parts.append(f"**Risk Assessment for {name}:**")
        parts.append("Based on the provided data and the 'Yield Staking Risk Template', assess this pool against relevant risks.")

    parts.append("\n## Overall Comparative Analysis and Conclusion")
    parts.append("Provide a brief comparative analysis of all the selected pools and a final recommendation.")

    prompt = "\n\n".join(parts)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a highly experienced DeFi investment analyst specializing in risk assessment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip(), selected_pools_context
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"Error generating report: {e}", selected_pools_context
