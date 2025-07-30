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
from crawler import summarize_pool_info

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

    .badge-staking { background: #dbeafe; color: #1e40af; }
    .badge-lending { background: #dcfce7; color: #166534; }
    .badge-farming { background: #fef3c7; color: #92400e; }
    .badge-vault { background: #e7e5e4; color: #44403c; }
    </style>
    """, unsafe_allow_html=True)

# Classification Functions
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
    # Check for unsustainable APY spikes
    if apy_30d and apy_30d > 0 and (apy_7d / apy_30d > 4):
        return False
    return True

def score_pool(pool): # Simplified scoring for initial filtering
    """Score a pool for ranking (used for initial selection for LLM)"""
    tvl = pool.get("tvlUsd", 0)
    apy = pool.get("apy", 0)
    risk_level = classify_risk(pool)

    score = 0
    if risk_level == "Low":
        score = tvl * 0.0001 + apy * 10
    elif risk_level == "Medium":
        score = tvl * 0.00005 + apy * 20
    else: # High
        score = tvl * 0.00001 + apy * 30
    return score


# Data Fetching Functions
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
    """
    Get news summary using EXA API with chain and project context.
    Returns both a formatted string for display and raw news data for AI.
    """
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

@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_summary_for_pool(pool, token_info, raw_news_data):
    """
    Generates an AI-driven investment summary for a given DeFi pool.
    """
    if not OPENAI_API_KEY:
        return "OpenAI API key is not configured for AI summaries. Please set OPENAI_API_KEY in your .env file."

    openai.api_key = OPENAI_API_KEY

    pool_details = f"""
    Pool Name: {pool.get('pool', 'N/A')}
    Project: {pool.get('project', 'N/A')}
    Symbol: {pool.get('symbol', 'N/A')}
    Chain: {pool.get('chain', 'N/A')}
    APY: {pool.get('apy', 0):.2f}% (Base APY: {pool.get('apyBase', 'N/A')}%, Reward APY: {pool.get('apyReward', 'N/A')}%)
    TVL (USD): ${pool.get('tvlUsd', 0):,.0f}
    Strategy: {classify_pool_type(pool)}
    Calculated Risk Level: {classify_risk(pool)}
    Impermanent Loss (IL) Risk: {pool.get('ilRisk', 'N/A')}
    """

    token_details = "Token Market Data Unavailable."
    if token_info and token_info.get("token_data"):
        td = token_info["token_data"]
        md = td.get("market_data", {})
        token_details = f"""
        Token Name: {td.get('name', 'N/A')} ({td.get('symbol', 'N/A').upper()})
        Current Price: ${md.get('current_price', {}).get('usd', 0):,.4f}
        24h Price Change: {md.get('price_change_percentage_24h', 0):.2f}%
        Market Cap: ${md.get('market_cap', {}).get('usd', 0):,.0f}
        Market Cap Rank: #{md.get('market_cap_rank', 'N/A')}
        All-Time High (ATH): ${md.get('ath', {}).get('usd', 0):,.4f}
        All-Time Low (ATL): ${md.get('atl', {}).get('usd', 0):,.4f}
        7-Day Price Change: {md.get('price_change_percentage_7d', 0):.2f}%
        30-Day Price Change: {md.get('price_change_percentage_30d', 0):.2f}%
        1-Year Price Change: {md.get('price_change_percentage_1y', 0):.2f}%
        """

    news_texts = ""
    if raw_news_data:
        news_texts = "\n\n".join([
            f"Title: {n.get('title', 'No Title')}\nURL: {n.get('url', '')}\nText: {n.get('text', 'No content available.')}"
            for n in raw_news_data
        ])
    if not news_texts:
        news_texts = "No specific recent news articles found for this token or project."

    summarized_crawled_info = "No specific additional information crawled for this pool from web."
    pool_id_for_crawler = pool.get("pool") # Use the 'pool' object (the DefiLlama pool data)
    if pool_id_for_crawler:
        try:
            crawled_info_result = summarize_pool_info(f"https://defillama.com/yields/pool/{pool_id_for_crawler}")
            print(f"Crawled info for pool {pool_id_for_crawler}: {crawled_info_result}")
            if crawled_info_result: # Check if crawler returned something
                summarized_crawled_info = crawled_info_result
        except Exception as e:
            logging.error(f"Error crawling information for pool {pool_id_for_crawler}: {e}")
            summarized_crawled_info = "Failed to retrieve additional crawled information for this pool due to an error."
    

    prompt = f"""
    You are a highly experienced and cautious DeFi investment analyst.
    Provide a comprehensive investment summary for the following DeFi yield pool.
    Your analysis should be structured, cover key aspects, and conclude with a nuanced recommendation.

    **DeFi Pool Data:**
    {pool_details}

    **Associated Token Market Data:**
    {token_details}

    **Recent News Articles (relevant to the token or project):**
    {news_texts}

    **Summarized Crawled Information:**
    {summarized_crawled_info}

    **Based on the above data, provide an investment insight report following this structure. Give preference to summarized crawled information relevant**

    ### 💡 AI Investment Insight for {pool.get('symbol', 'N/A').upper()} on {pool.get('project', 'N/A').title()} ({pool.get('chain', 'N/A').upper()})

    **1. Overview of the Pool:**
    [Briefly describe the pool, its primary function (lending, LP farming, staking), the project it belongs to, and its TVL and current APY. Mention the calculated risk level and strategy.]

    **2. Investment Opportunity & Benefits:**
    [Explain the main attractive points. How does it generate yield? Highlight the APY in context of its TVL and risk. Mention any benefits from the token's market position or positive news.]

    **3. Key Risks & Considerations:**
    [Crucially, detail the potential downsides.
    - **Protocol Risk:** Smart contract vulnerabilities, oracle failures, governance risks.
    - **Market Risk:** Token price volatility (refer to 24h, 7d, 30d, 1y changes, ATH/ATL), impermanent loss (if applicable, refer to IL Risk), APY variability.
    - **Liquidity Risk:** Consider TVL size; smaller TVL can imply lower liquidity and higher slippage.
    - **Regulatory Risk:** General DeFi regulatory uncertainty.
    - **News Impact:** Any negative news or warnings from the provided articles.
    - **Identified APY Risk:** Base APY vs Reward APY
    - **Risk Analysis:** [Provide a brief analysis of the overall risk profile based on the identified risks.]

    **4. Key Features and Differentiators:**
    [Highlight what sets this pool apart from others. Consider unique mechanisms, incentives, or partnerships that enhance its value proposition.]

    **5. Underlying Protocols & Assets**
    [Discuss the underlying protocols and assets involved. Are they well-established? Do they have a history of security and reliability?]

    **6. Token Market Context:**
    [Analyze the token's recent market performance (price trends, volatility) and its overall market cap and rank. How does its past performance inform future expectations? Is it an established asset or more speculative?]

    **7. Recommendation & Suitability:**
    [Provide a clear, balanced conclusion. Is this generally a "Good Investment," "Consider with Caution," or "High Risk, High Reward"? State what kind of investor (e.g., risk-averse, experienced, yield-hungry) this pool might be suitable for. Emphasize the importance of personal risk tolerance and due diligence. Avoid definitive "buy/sell" advice.]

    **8. Important Considerations Before Investing:**
    [Any final crucial advice, such as monitoring APY, checking audits, understanding the protocol's mechanics, and diversifying.]
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a highly experienced DeFi investment analyst. Provide clear, balanced, and actionable insights based on the provided data. Structure your response strictly as requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        return "OpenAI API key is invalid or not set. Please ensure `OPENAI_API_KEY` is correct in your .env file."
    except openai.APICallError as e:
        logging.error(f"OpenAI API Error: {e.code} - {e.message}")
        return f"Error communicating with OpenAI API: {e.message}. (Code: {e.code})"
    except Exception as e:
        logging.error(f"Failed to generate AI summary: {e}")
        return f"An unexpected error occurred while generating the summary: {e}"


@st.cache_data(ttl=3600, show_spinner=True)
def generate_llm_portfolio_recommendation(token_symbol, total_value, eligible_pools, risk_preference):
    """
    Generates a portfolio recommendation using an LLM based on user preferences.
    The LLM also determines the optimal diversification strategy.
    """
    if not OPENAI_API_KEY:
        return None, "OpenAI API key is not configured for LLM portfolio. Please set OPENAI_API_KEY in your .env file."

    openai.api_key = OPENAI_API_KEY

    # Prepare simplified pool data for the LLM
    pools_for_llm = []
    # Limit to top 25 scored pools to keep context window manageable and focus LLM on best options
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

    # FIXED: Provide the full pool ID to the LLM
    pools_str = "\n".join([
        f"- ID: {p['pool_id']} | Project: {p['project']} | Symbol: {p['symbol']} | Chain: {p['chain']} | APY: {p['apy']:.2f}% | TVL: ${p['tvlUsd'] / 1e6:.1f}M | Risk: {p['risk']} | Strategy: {p['strategy']}"
        for p in pools_for_llm
    ])
    print(f"Pools for LLM:\n{pools_str}")
    prompt = f"""
You are an expert DeFi portfolio manager and strategist. Your goal is to construct a profitable, chain-diverse, and risk-appropriate yield farming portfolio tailored to the user's preferences.

**User Investment Details:**
- Total amount to invest: ${total_value:,.0f} in {token_symbol.upper()}
- User's Risk Preference: {risk_preference} (Conservative, Balanced, or Aggressive)

**Available Yield Pools (from which to choose):**
{pools_str}

**Instructions:**
1. **Determine the optimal diversification strategy** based on the user's risk preference, the characteristics of the available pools (e.g., APY, TVL, Risk Level), and blockchain diversity.
    - Diversify across **multiple blockchains**, especially mid-tier chains (e.g., Avalanche, Optimism, Base, Fantom), which offer strong yields and acceptable security profiles.
    - Avoid overconcentration on a single chain, project, or high-risk pool unless warranted by the strategy (e.g., Aggressive risk preference).

2. **Select relevant pools** from the "Available Yield Pools" list above based on your chosen strategy.
    - **Only select pools that are present in the provided list. Use the full Pool ID exactly as shown.**
    - Prefer portfolios with exposure to **at least 3 different blockchains** (favoring a mix of top-tier like Ethereum/Arbitrum and mid-tier chains like Base or Optimism) unless user preference or pool quality restricts this.

3. **Determine percentage allocations** for each selected pool.
    - The total allocation MUST sum to **100%**.
    - Distribute allocations logically based on APY, TVL, risk level, and **chain diversity**.
        - Conservative: Favor high TVL, low-risk pools on top-tier chains.
        - Balanced: Mix of high-APY and lower-risk pools, with healthy exposure to mid-tier chains.
        - Aggressive: Favor high-APY pools across multiple chains, including emerging or mid-tier networks, but still diversify to mitigate smart contract or chain-specific risk.

4. **Provide a "Portfolio Explanation" first, followed by a "Portfolio Allocation" markdown table.**

**Portfolio Explanation:**
[Explain your strategic approach to constructing this portfolio. Clearly state the chosen diversification strategy (e.g., ‘Multi-Chain Balanced’ or ‘Aggressive Mid-Tier Yield Strategy’). Discuss how the selected pools and their allocations align with the user’s risk preference, including rationale for chain exposure, APY prioritization, and risk mitigation. Highlight key advantages and potential trade-offs.]

**Portfolio Allocation:**
| Pool ID | Project | Symbol | Chain | APY (%) | Risk Level | Allocation (%) |
|---|---|---|---|---|---|---|
"""


    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional DeFi portfolio manager. Provide clear, balanced, and actionable portfolio recommendations. Structure your response strictly as requested, providing an explanation first and then a markdown table for allocation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        llm_output = response.choices[0].message.content
        logging.info(f"LLM Raw Output:\n{llm_output}")

        # Parse the LLM's output
        if "Portfolio Allocation:" not in llm_output:
            return None, "LLM response did not contain the expected 'Portfolio Allocation:' section. Please try again."

        explanation_part, allocation_table_part = llm_output.split("Portfolio Allocation:", 1)
        explanation = explanation_part.strip()

        # Extract markdown table
        table_lines = allocation_table_part.strip().split('\n')
        if len(table_lines) < 2: # Check for at least header and one row
            return None, "LLM response's allocation table is malformed or empty. Please try again."

        # FIXED: Filter for valid table rows to ignore trailing text from the LLM
        data_rows = [line for line in table_lines if line.strip().startswith('|') and '---' not in line and 'Pool ID' not in line]


        portfolio_allocations = []
        # FIXED: Create a mapping from the FULL pool ID back to the original pool object
        pool_id_map = {p.get("pool", ""): p for p in eligible_pools}

        for row in data_rows:
            if "TOTAL" in row.upper(): # Skip the total row if LLM adds it
                continue
            cols = [col.strip() for col in row.strip('|').split('|')]
            if len(cols) == 7: # Expect 7 columns as defined in prompt
                try:
                    pool_id = cols[0] # This is now the full ID
                    allocation_percent_str = cols[6].replace('%', '').strip()
                    allocation_percent = float(allocation_percent_str)

                    # FIXED: Look up using the full ID
                    original_pool_data = pool_id_map.get(pool_id)

                    if original_pool_data:
                        portfolio_allocations.append({
                            "pool": original_pool_data,
                            "allocation_percent": allocation_percent,
                            "risk": classify_risk(original_pool_data),
                            "apy": original_pool_data.get("apy", 0)
                        })
                    else:
                        logging.warning(f"Could not map pool ID '{pool_id}' from LLM output back to original pool list. The LLM may have hallucinated a pool.")
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing table row data: {row} - {e}")
                    continue
            else:
                logging.warning(f"Skipping malformed table row: {row}. Expected 7 columns, got {len(cols)}")

        # Normalize allocations if they don't sum to exactly 100% (due to LLM rounding)
        current_total_allocation = sum([item["allocation_percent"] for item in portfolio_allocations])
        if current_total_allocation > 0 and abs(current_total_allocation - 100) > 0.1:
            logging.warning(f"Allocations sum to {current_total_allocation:.2f}%, normalizing to 100%.")
            for item in portfolio_allocations:
                item["allocation_percent"] = (item["allocation_percent"] / current_total_allocation) * 100

        return portfolio_allocations, explanation

    except openai.AuthenticationError:
        return None, "OpenAI API key is invalid or not set. Please ensure `OPENAI_API_KEY` is correct in your .env file."
    except openai.APICallError as e:
        logging.error(f"OpenAI API Error: {e.code} - {e.message}")
        return None, f"Error communicating with OpenAI API: {e.message}. (Code: {e.code})"
    except Exception as e:
        logging.error(f"Failed to generate LLM portfolio: {e}")
        return None, f"An unexpected error occurred while generating the portfolio: {e}"


# UI Components
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

def create_pool_card(pool, strategy, risk, token_data_for_ai, raw_news_data_for_ai):
    """Create enhanced pool card with AI summary button"""
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

    ai_summary_key = f"ai_summary_expander_{pool_id}"
    if ai_summary_key not in st.session_state:
        st.session_state[ai_summary_key] = False

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

        if st.button("✨ Get AI Summary", key=f"ai_btn_{pool_id}"):
            st.session_state[ai_summary_key] = not st.session_state[ai_summary_key]

        if st.session_state[ai_summary_key]:
            with st.spinner("Generating AI Summary... This might take a moment."):
                ai_summary = generate_ai_summary_for_pool(pool, token_data_for_ai, raw_news_data_for_ai)
            st.markdown(f'<div class="ai-summary">{ai_summary}</div>', unsafe_allow_html=True)

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

# MODIFIED: create_portfolio_optimizer no longer takes diversification_level from user
def create_portfolio_optimizer(token_symbol, pools, token_amount, token_price):
    """Create profit-maximizing portfolio optimization interface, powered by LLM"""
    st.subheader("🎯 LLM-Powered Portfolio Constructor")

    total_value = token_amount * token_price

    if total_value == 0:
        st.warning("Please enter a valid token amount and ensure token price is available to construct a portfolio.")
        return []

    # Only one input: Risk Preference
    risk_preference = st.selectbox(
        "Risk Preference",
        ["Conservative", "Balanced", "Aggressive"],
        help="Conservative: Focus on stability and lower risk. Balanced: Optimize for risk-reward. Aggressive: Focus on maximizing potential returns, higher risk."
    )

    # Filter pools to pass to LLM - LLM will do the final selection and allocation
    eligible_pools = [p for p in pools if p.get("tvlUsd", 0) >= 500_000 and p.get("apy", 0) >= 0.5]

    if not eligible_pools:
        st.warning("No eligible pools found for portfolio construction after initial filtering.")
        return []

    # Sort eligible pools by the general score, so the LLM gets better options first
    eligible_pools.sort(key=score_pool, reverse=True)

    if st.button("🚀 Construct Portfolio (Powered by LLM)"):
        with st.spinner("Generating portfolio recommendation with AI... This may take up to 30 seconds."):
            portfolio_allocations_llm, explanation_llm = generate_llm_portfolio_recommendation(
                token_symbol,
                total_value,
                eligible_pools,
                risk_preference # Only risk preference is passed to LLM
            )

        if portfolio_allocations_llm is None:
            st.error(f"Failed to generate portfolio: {explanation_llm}")
            return []
        logging.info(f"LLM Portfolio Allocations: {portfolio_allocations_llm}")

        if portfolio_allocations_llm:
            portfolio = []
            total_expected_return = 0

            for item in portfolio_allocations_llm:
                pool = item["pool"]
                allocation_percent = item["allocation_percent"]
                amount = total_value * (allocation_percent / 100)
                expected_return = amount * (pool["apy"] / 100)

                portfolio.append({
                    "pool": pool,
                    "allocation": allocation_percent / 100, # Store as decimal for calculations
                    "amount": amount,
                    "risk": item["risk"],
                    "apy": item["apy"],
                    "expected_return": expected_return,
                })
                total_expected_return += expected_return

            weighted_apy = (total_expected_return / total_value) * 100 if total_value > 0 else 0

            # Extract diversification strategy from LLM explanation for display
            # Assuming the LLM will state it clearly, e.g., "The chosen diversification strategy is: Single Best Pool"
            diversification_strategy_match = re.search(r"chosen diversification strategy (?:is|was):\s*([^\.\n]+)", explanation_llm, re.IGNORECASE)
            llm_decided_strategy = diversification_strategy_match.group(1).strip() if diversification_strategy_match else "LLM Decided"


            # Portfolio summary with profit focus
            st.markdown(f"""
            <div class="portfolio-summary">
                <h3 style="margin-top: 0;">💰 LLM-Recommended Portfolio Summary</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #4ade80;">${total_expected_return:,.0f}</div>
                        <div>Expected Annual Profit</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold;">{weighted_apy:.2f}%</div>
                        <div>Portfolio APY</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold;">{len(portfolio)}</div>
                        <div>Pools Selected</div>
                    </div>
                     <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold;">${total_value:,.0f}</div>
                        <div>Total Invested</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <strong>LLM Strategy: {llm_decided_strategy} • Risk Profile: {risk_preference}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display LLM's explanation
            st.markdown("### 🧠 LLM's Strategy Explanation")
            st.markdown(explanation_llm)

            # Portfolio breakdown with profit metrics
            st.subheader("💎 Portfolio Allocation (Recommended by LLM)")

            portfolio_display = sorted(portfolio, key=lambda x: x["allocation"], reverse=True)

            for i, item in enumerate(portfolio_display, 1):
                pool = item["pool"]
                allocation = item["allocation"]
                amount = item["amount"]
                expected_return = item["expected_return"]

                if allocation >= 0.4:
                    border_color = "#10b981"
                elif allocation >= 0.2:
                    border_color = "#3b82f6"
                else:
                    border_color = "#6b7280"

                st.markdown(f"""
                <div style="border-left: 4px solid {border_color}; background: white; padding: 1rem;
                            margin: 0.5rem 0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(f"""
                    **#{i} {pool['project'].title()} - {pool['symbol'].upper()}**
                    Chain: {pool['chain'].upper()} | APY: {pool['apy']:.2f}% | Risk: {item['risk']}
                    """)

                with col2:
                    st.metric("Allocation", f"{allocation*100:.1f}%")

                with col3:
                    st.metric("Investment", f"${amount:,.0f}")

                with col4:
                    st.metric("Expected Return", f"${expected_return:,.0f}/yr")

                st.markdown("</div>", unsafe_allow_html=True)

            # Risk vs Return scatter plot (using actual allocations)
            if len(portfolio) > 1:
                fig_scatter = px.scatter(
                    x=[item["apy"] for item in portfolio],
                    y=[item["allocation"] * 100 for item in portfolio],
                    size=[item["amount"] for item in portfolio],
                    color=[item["risk"] for item in portfolio],
                    hover_data={
                        "Project": [item["pool"]["project"] for item in portfolio],
                        "Symbol": [item["pool"]["symbol"] for item in portfolio],
                        "Allocation": [f"{item['allocation']*100:.1f}%" for item in portfolio],
                        "Investment": [f"${item['amount']:,.0f}" for item in portfolio],
                        "Expected Return": [f"${item['expected_return']:,.0f}/yr" for item in portfolio],
                    },
                    title="Portfolio Allocation vs APY",
                    labels={"x": "APY (%)", "y": "Allocation (%)", "color": "Risk Level"},
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Risk distribution
            risk_dist = {"Low": 0, "Medium": 0, "High": 0}
            for item in portfolio:
                risk_dist[item["risk"]] += item["allocation"]

            fig_pie = px.pie(
                values=list(risk_dist.values()),
                names=list(risk_dist.keys()),
                title="Portfolio Risk Distribution",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

            return portfolio
        else:
            st.warning("The LLM could not construct a portfolio based on the provided parameters. Try adjusting your preferences.")
            return []
    return []


# Main Application
def main():
    st.set_page_config(
        page_title="DeFi Yield Explorer",
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
            Discover, analyze, and optimize your DeFi yield farming strategies with AI insights
        </p>
    </div>
    """, unsafe_allow_html=True)

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
        return

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
                create_pool_card(pool, strategy, risk_level, token_data, raw_news_for_ai)
                st.markdown("---")

            if len(risk_pools) > 3:
                toggle_label = "Show Less" if st.session_state[key] else f"Show {len(risk_pools) - 3} More"
                if st.button(f"{toggle_label} {risk} Risk Pools", key=f"toggle_{risk.lower()}"):
                    st.session_state[key] = not st.session_state[key]
                    st.rerun()

    with tab2:
        if token_price > 0:
            # diversification_level is no longer passed here
            portfolio = create_portfolio_optimizer(token, pools, token_amount, token_price)

            if portfolio:
                if st.button("📥 Export Portfolio Data"):
                    portfolio_df = pd.DataFrame([{
                        "Project": item["pool"]["project"],
                        "Symbol": item["pool"]["symbol"],
                        "Chain": item["pool"]["chain"],
                        "APY": item["pool"]["apy"],
                        "TVL": item["pool"]["tvlUsd"],
                        "Risk": item["risk"],
                        "Allocation": f"{item['allocation']*100:.1f}%",
                        "Amount": f"${item['amount']:,.2f}",
                        "Pool_URL": f"https://defillama.com/yields/pool/{item['pool']['pool']}"
                    } for item in portfolio])

                    csv = portfolio_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"{token.upper()}_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("⚠️ Token price data needed for portfolio construction")

    with tab3:
        st.markdown("### 📰 Market Intelligence")
        st.markdown(news_summary_display)

        if token_data:
            market_data = token_data["token_data"].get("market_data", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Market Metrics")
                st.write(f"**All-time High:** ${market_data.get('ath', {}).get('usd', 0):,.4f}")
                st.write(f"**All-time Low:** ${market_data.get('atl', {}).get('usd', 0):,.4f}")
                st.write(f"**Market Cap Rank:** #{market_data.get('market_cap_rank', 'N/A')}")

            with col2:
                st.markdown("#### ⏱️ Performance")
                st.write(f"**7d Change:** {market_data.get('price_change_percentage_7d', 0):.2f}%")
                st.write(f"**30d Change:** {market_data.get('price_change_percentage_30d', 0):.2f}%")
                st.write(f"**1y Change:** {market_data.get('price_change_percentage_1y', 0):.2f}%")


if __name__ == "__main__":
    setup_playwright()

    main()