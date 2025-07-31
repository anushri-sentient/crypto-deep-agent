import os
import logging
import asyncio
import re
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import openai
import streamlit as st

# --- Configuration ---
CRAWLER_API = os.getenv("CRAWLER_API_ENDPOINT", "https://crawl4ai.dev.sentient.xyz/crawl_direct")
CRAWLER_AUTH_TOKEN = os.getenv("CRAWLER_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY not set. AI features will not function.")

config = CrawlerRunConfig(wait_until="networkidle")


# --- Async Crawl Utilities ---
async def crawl_url_async(url: str) -> str:
    """Asynchronously crawls a given URL using crawl4ai and returns cleaned HTML."""
    if not CRAWLER_AUTH_TOKEN:
        logger.error("CRAWLER_AUTH_TOKEN not set. Cannot perform web crawl.")
        return ""

    try:
        async with AsyncWebCrawler(api_key=CRAWLER_AUTH_TOKEN) as crawler:
            logger.info(f"Starting crawl for: {url}")
            result = await crawler.arun(url=url, config=config)
            return result.cleaned_html if result and result.cleaned_html else ""
    except Exception as e:
        logger.error(f"Error during async crawl of {url}: {e}")
        return ""


def crawl_url(url: str) -> str:
    """Synchronously wraps the asynchronous crawl function to be callable from sync contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(crawl_url_async(url), loop)
            return future.result(timeout=60) # Increased timeout to 60 seconds for larger pages
        else:
            return loop.run_until_complete(crawl_url_async(url))
    except Exception as e:
        logger.error(f"Synchronous crawl failed for {url}: {e}")
        return ""


# --- Link Filtering ---
def filter_links(links: list) -> list:
    """Filters out unwanted social media, blockchain explorers, and other less relevant links."""
    blocked_keywords = [
        "twitter.com", "x.com", "discord.gg", "discord.com", "t.me",
        "reddit.com", "youtube.com", "medium.com", "github.com",
        "etherscan.io", "arbiscan.io", "polygonscan.com", "bscscan.com", # Block explorers
        "defillama.com" # Avoid re-crawling DefiLlama itself from extracted links
    ]
    # Use re.IGNORECASE for case-insensitive matching
    filtered = []
    for link in links:
        if not any(re.search(keyword, link, re.IGNORECASE) for keyword in blocked_keywords):
            filtered.append(link)
    return filtered


# --- GPT-4o summarization + URL extraction ---
def gpt4o_summarize_and_extract_links(html: str, pool_name: str = "") -> tuple[str, list]:
    """
    Summarizes HTML content and extracts relevant links using GPT-4o.
    Args:
        html (str): The HTML content to summarize.
        pool_name (str): Identifier for logging/error messages.
    Returns:
        tuple[str, list]: A tuple containing the summarized text and a list of extracted URLs.
    """
    if not openai.api_key:
        logger.warning("OpenAI API key is missing.")
        return "OpenAI client not initialized. API key missing for summarization.", []
    if not html:
        return "No content provided to summarize.", []

    # Truncate HTML for safety and to manage token usage, even for large context models
    truncated_html = html[:50000] # Approx 50k chars, well within most model limits (tokens are ~1:4 chars)

    prompt = (
        f"You are a highly skilled DeFi expert. The following is extracted HTML content from a DeFi pool or project page.\n\n"
        f"**Task:**\n"
        f"1.  **Summarize:** Provide a concise summary (2-4 sentences) of the project's purpose, its key features, how it generates yield, and any immediate risks mentioned or implied within the text. If the text is very generic or primarily points to other resources (e.g., just a list of links), mention that.\n"
        f"2.  **Extract Links:** Identify and list up to 2 **most relevant and official** links from the HTML text. Prioritize official project websites, whitepapers, audit reports, comprehensive documentation, or security analysis reports. Do NOT include social media (Twitter, Discord, Telegram, Reddit, etc.) or blockchain explorers (Etherscan, Arbiscan, etc.) in this list. Identify contracts and github links\n"
        f"Format the output as follows:\n"
        f"SUMMARY: [Your concise summary here]\n"
        f"LINKS: [Link1, Link2 (if found, otherwise leave empty)]\n\n"
        f"**HTML Content:**\n{truncated_html}"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # Using gpt-4o for better summarization quality
            messages=[
                {"role": "system", "content": "You are a concise and accurate DeFi expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Parse the structured output
        summary_match = re.search(r'SUMMARY:\s*(.*?)(?=\nLINKS:|$)', content, re.DOTALL | re.IGNORECASE)
        links_match = re.search(r'LINKS:\s*(.*)', content, re.DOTALL | re.IGNORECASE)

        summary_text = summary_match.group(1).strip() if summary_match else content.strip() # Fallback
        raw_links_str = links_match.group(1).strip() if links_match else ""

        # Extract URLs from the raw links string (they might be comma-separated, etc.)
        extracted_urls = re.findall(r'https?://[^\s)\]\}>"\'<]+', raw_links_str)

        # Apply the explicit filter to avoid unwanted links
        extracted_urls = filter_links(extracted_urls)

        # Ensure unique links and limit to 2
        extracted_urls = list(dict.fromkeys(extracted_urls)) # Removes duplicates while preserving order
        extracted_urls = extracted_urls[:2] # Limit to top 2

        logger.info(f"GPT-4o summarized {pool_name}. Extracted links: {extracted_urls}")
        return summary_text, extracted_urls

    except Exception as e:
        logger.error(f"GPT-4o summarization/link extraction error for {pool_name}: {e}")
        return "Error summarizing content.", []


# --- Final Report Generator (Comprehensive) ---
def final_investor_summary(all_summaries: list) -> str:
    """
    Combines multiple summaries into a final, comprehensive, investor-focused brief.
    Args:
        all_summaries (list): A list of tuples, where each tuple is (source_name, summary_text).
    Returns:
        str: The final, synthesized investment summary.
    """
    if not openai.api_key:
        return "OpenAI client not initialized. API key missing for final summary."
    if not all_summaries:
        return "No summaries provided for final report generation."

    combined_text = "\n\n".join(
        f"--- Summary from {source} ---\n{summary}"
        for source, summary in all_summaries
    )
    
    # Truncate combined_text if it's too long for the prompt (e.g., 80k characters)
    # gpt-4o context window is 128k tokens, but keeping a buffer is good practice.
    if len(combined_text) > 80000:
        combined_text = combined_text[:80000] + "\n\n... [Content truncated due to length]"
        logger.warning("Combined text truncated for final summary prompt due to length.")

    final_prompt = ("You are a highly experienced and cautious DeFi analyst preparing an investment briefing for a potential investor. "
        "Your task is to synthesize the following collected information from various web sources (main pool page, official documentation, etc.) related to a specific DeFi yield pool.\n\n"
        "**Synthesize the information into a single, comprehensive, investor-focused report.** Do not just concatenate or list summaries; genuinely combine and analyze them. The report should be easy to read and provide a balanced perspective.\n\n"
        "**Ensure the report covers the following sections:**\n"
        "1.  **Project, Chain & Pool Overview:** Briefly explain the project's core mission and the specific DeFi pool's function and Chain too (e.g., lending, LP farming, staking). State the Total Value Locked (TVL) and how yield is generated.\n"
        "2.  **Key Features & Differentiators:** What makes this pool/project unique, innovative, or particularly attractive? Highlight any competitive advantages.\n"
        "3.  **Underlying Protocols & Assets:** Describe the underlying blockchain(s), assets (e.g., stablecoins, ETH, specific tokens), and other DeFi protocols (if any) it integrates with.\n"
        "4.  **Risk Analysis (CRITICAL):** This is the most important section. Detail potential risks including: smart contract vulnerabilities (audits?), impermanent loss (if applicable), oracle risks, economic exploits (e.g., flash loan attacks, tokenomics vulnerabilities), potential for rug pulls, and any specific protocol-level or market risks mentioned. Evaluate its overall security posture and decentralization level. Be explicitly cautious.\n"
        "5.  **Audit Status & Security:** Explicitly state if audit information was found and its implications (e.g., audited by CertiK, no recent audit found). Show contract links\n"
        "6.  **Recommendation & Considerations:** Conclude with a clear, balanced, and actionable assessment for the investor. Is this a relatively safe/transparent investment or high-risk? What kind of investor is it suitable for (e.g., conservative, yield-hungry)? What are the key things an investor *must* watch out for and perform due diligence on before considering investment? **Avoid definitive 'invest/don't invest' advice, but provide a strong caution or qualified endorsement based on your risk assessment.** Emphasize diversification and only investing what one can afford to lose.\n\n"
        f"**Collected Information for Synthesis:**\n{combined_text}\n\n Please provide the report directly, without conversational intros.")

    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # Using gpt-4o for higher quality final summary
            messages=[
                {"role": "system", "content": "You are a professional, cautious, and detail-oriented DeFi investment advisor. Provide a balanced and thorough assessment structured by the requested sections."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.4, # Slightly higher temperature for more nuanced synthesis
            max_tokens=1500 # Sufficient tokens for a detailed summary
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Final investor summary generation failed: {e}")
        return "Error generating final summary due to an AI processing issue."


# --- Comprehensive Investment Report Generator (Web Crawl Based) ---
# This function orchestrates crawling multiple pages and generating a comprehensive summary.
@st.cache_data(ttl=3600, show_spinner="Crawling web for comprehensive analysis...")
def generate_comprehensive_investment_report(pool_url: str) -> str:
    """
    Orchestrates crawling web pages related to a DeFi pool and summarizing the information.
    This function's output is cached to prevent redundant API calls and processing.

    Args:
        pool_url (str): The primary URL of the DeFi pool (e.g., DefiLlama yields page).

    Returns:
        str: A comprehensive investor-focused summary of the pool.
    """
    logger.info(f"[INFO] Starting full summary process for: {pool_url}")

    if not pool_url or not pool_url.startswith("http"):
        logger.error(f"Invalid pool URL provided to generate_comprehensive_investment_report: {pool_url}")
        return "Invalid pool URL provided for web crawling."

    all_summaries = []

    # 1. Crawl the main pool page
    main_html = crawl_url(pool_url)
    if not main_html:
        logger.error(f"Failed to retrieve HTML from main pool URL: {pool_url}. Aborting further crawling for this pool.")
        return f"Could not retrieve main pool information from {pool_url}."

    main_summary_text, secondary_links = gpt4o_summarize_and_extract_links(main_html, pool_name=pool_url)
    all_summaries.append((f"Main Pool Page ({pool_url})", main_summary_text))
    
    # 2. Crawl secondary links (up to 2 of the filtered links)
    effective_secondary_links = filter_links(secondary_links) # Re-filter just in case LLM outputs something new
    
    for link_idx, link in enumerate(effective_secondary_links):
        if link_idx >= 2: # Limit secondary crawls to max 2
            break
        try:
            logger.info(f"Crawling secondary link {link_idx+1}: {link}")
            html = crawl_url(link)
            if not html:
                logger.warning(f"Empty HTML returned from secondary crawl of {link}.")
                all_summaries.append((f"Secondary Link ({link})", "No relevant content found for this link."))
                continue
            
            sub_summary_text, _ = gpt4o_summarize_and_extract_links(html, pool_name=link)
            all_summaries.append((f"Secondary Link ({link})", sub_summary_text))
        except Exception as e:
            logger.warning(f"Failed crawling secondary link {link}: {e}")
            all_summaries.append((f"Secondary Link ({link})", f"Error crawling or summarizing: {e}"))

    # 3. Generate final summary
    final_summary_result = final_investor_summary(all_summaries)
    logger.info(f"[INFO] Completed full summary process for: {pool_url}")
    return final_summary_result


# --- In-depth Analysis Report Generator (incorporating external data and web crawl summary) ---
# This is the function that provides the equivalent of your requested "in-depth analysis report".
@st.cache_data(ttl=3600, show_spinner="Generating in-depth AI analysis report...")
def generate_in_depth_ai_analysis_report(pool_data: dict, token_info: str = "", news_data: list = None) -> str:
    """
    Generates a highly structured, AI-driven investment analysis report for a DeFi pool,
    combining provided pool metrics, token information, recent news, and web-crawled summaries.
    This function replaces the previous `generate_pool_breakdown_report`.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set for in-depth AI analysis.")
        return "OpenAI API key is not configured. Cannot generate in-depth report."
    
    if news_data is None:
        news_data = []

    # Construct pool details string robustly
    pool_project = pool_data.get('project', 'N/A')
    pool_symbol = pool_data.get('symbol', 'N/A')
    pool_apy = pool_data.get('apy', 0)
    pool_tvl = pool_data.get('tvlUsd', 0)
    pool_chain = pool_data.get('chain', 'N/A')
    
    # Use formatters for better presentation in the prompt
    fmt_apy = f"{pool_apy:.2f}%" if pool_apy else "0%"
    fmt_tvl = f"${pool_tvl:,.0f}" if pool_tvl else "$0"

    pool_details_str = (
        f"Project: {pool_project}, Symbol: {pool_symbol}, "
        f"APY: {fmt_apy}, TVL: {fmt_tvl}, Chain: {pool_chain}"
    )

    # Get the comprehensive web-crawled summary using the existing function
    pool_defillama_id = pool_data.get('pool', '') # The 'pool' key is often the ID used in DefiLlama URLs
    pool_defillama_url = f"https://defillama.com/yields/pool/{pool_defillama_id}" if pool_defillama_id else ""

    crawled_web_summary = ""
    if pool_defillama_url:
        crawled_web_summary = generate_comprehensive_investment_report(pool_defillama_url)
    else:
        logger.warning(f"Could not construct DefiLlama URL for pool: {pool_data}. Web crawl summary might be limited.")
        crawled_web_summary = "No valid DefiLlama pool URL found, web-crawled information is unavailable."

# Construct pool details string
    pool_project = pool_data.get('project', 'N/A')
    pool_symbol = pool_data.get('symbol', 'N/A')
    pool_apy = pool_data.get('apy', 0)
    pool_tvl = pool_data.get('tvlUsd', 0)
    pool_chain = pool_data.get('chain', 'N/A')

    fmt_apy = f"{pool_apy:.2f}%" if pool_apy else "0%"
    fmt_tvl = f"${pool_tvl:,.0f}" if pool_tvl else "$0"
    pool_details_str = (
        f"Project: {pool_project}, Symbol: {pool_symbol}, "
        f"APY: {fmt_apy}, TVL: {fmt_tvl}, Chain: {pool_chain}"
    )

    # --- 🔍 Enrich token info via CoinGecko ---
    if not token_info and pool_symbol:
        enriched = get_token_info_from_coingecko(pool_symbol)
        if enriched:
            token_info = (
                f"Name: {enriched['name']}\n"
                f"Symbol: {enriched['symbol']}\n"
                f"Current Price: ${enriched['current_price_usd']}\n"
                f"Market Cap: ${enriched['market_cap_usd']:,}\n"
                f"24h Volume: ${enriched['total_volume_usd']:,}\n"
                f"Homepage: {enriched['homepage']}\n"
                f"Contract: {enriched['contract_address']}\n"
                f"CoinGecko URL: {enriched['coingecko_url']}"
            )
        else:
            token_info = "Token info could not be retrieved from CoinGecko."
    elif not token_info:
        token_info = "No specific token information provided."

    # Prepare news summary
    news_titles = [n.get('title', 'No title') for n in news_data if n.get('title')]
    news_summary = "\n".join([f"- {title}" for title in news_titles[:5]]) if news_titles else "No recent news provided."

    prompt = f"""
    You are an experienced DeFi yield strategist providing actionable investment analysis for users seeking to optimize their crypto yields. Focus on practical insights, realistic risk assessment, and clear guidance for position management.

    ** Token Information: **
    {token_info if token_info else "No specific token information provided."}
    
    **DeFi Pool Data:**
    {pool_details_str if pool_details_str else "No specific pool data provided."}

    **Summarized Crawled Information from the official page(s):**
    {crawled_web_summary if crawled_web_summary else "No comprehensive web information could be retrieved."}

    **Recent News (for context):**
    {news_summary if news_summary else "No recent news available."}

    **Analysis Requirements:**
    - Prioritize crawled information as the primary data source where applicable.
    - Provide quantified risk scenarios with specific numbers if possible (e.g., IL impact, TVL changes).
    - Include actionable entry/exit strategies.
    - Compare current metrics to historical performance if implied by data or context.
    - Consider gas costs and transaction efficiency.
    - Assess sustainability of current yields.
    - **IMPORTANT:** Do NOT invent or fabricate any data, risk scores, or numbers.
    - If information is missing or unclear, explicitly write "Data unavailable" or "Not applicable."
    - Cite data sources for all assessments.
    - Ensure the report strictly follows the given structure without deviation.
    - Show contract links (Audits, etc.) if available.
    - Add token information for both tokens (if there are two) in the pool.
    - Talk abou t the chain too (e.g., Ethereum, Arbitrum, etc.) and how it affects the pool.


    ---

    **Example report for reference:**

    ### 💡 Investment Analysis: UNI-V2 ETH/USDC Liquidity Pool

    **🎯 Quick Assessment**
    - **Risk Rating:** Low (3/10) — based on low volatility of USDC stablecoin paired with ETH volatility (source: token_info, pool_details_str)
    - **Recommended Portfolio Allocation:** 1-3% of DeFi holdings to balance safety and yield
    - **Investment Horizon:** 1-3 months, given current moderate APYs and market conditions
    - **Skill Level:** Beginner to Intermediate
    - **Current Market Status:** Neutral — ETH price stable recently, moderate volume (source: crawled_web_summary)

    **📊 Pool Performance Analysis**
    - **Yield Breakdown:**  
    - Trading fees: approx. 0.25% on volume, equating to ~5% APY (source: pool_details_str)  
    - No additional reward tokens currently  
    - **APY Sustainability:** Current APY around 5% aligns with past 30-day average of 4.8% (source: historical data from pool analytics)  
    - **Capital Efficiency:** Standard Uniswap V2 AMM, liquidity evenly distributed; TVL steady at $250M last 30 days  
    - **Volume & Utilization:** 7-day volume steady at $40M, supporting fee income

    **💰 Opportunity Assessment**
    - **Primary Value Proposition:** Low risk stablecoin pairing with ETH exposure and reliable fee income  
    - **Yield Optimization Potential:** Compounding fees weekly recommended; no complex strategies required  
    - **Competitive Advantages:** Large TVL ensures low slippage; established protocol with mature ecosystem  
    - **Market Timing:** Neutral market favors steady income rather than speculation

    **⚠️ Risk Analysis & Mitigation**

    - **Impermanent Loss Scenarios:**  
    - At 10% ETH price divergence: ~1.4% IL loss (calculated using IL formula with price ratio 1.1)  
    - At 25% divergence: ~5.7% IL loss  
    - At 50% divergence: ~13.4% IL loss  
    - **Smart Contract Risk:** Uniswap V2 audited by multiple firms; no critical vulnerabilities reported since launch  
    - **Liquidity Risk:** High TVL and trading volume ensure strong exit liquidity; slippage below 0.5% for $100K withdrawals  
    - **Market Risk:** ETH volatility dominant risk; USDC stable mitigates one side  
    - **Operational Risk:** Gas fees moderate on Ethereum mainnet; minimal complexity in managing LP position  
    - **Other Risks:** No inflationary rewards; governance risks low due to protocol maturity

    **🎯 Strategic Recommendations**

    **Entry Strategy:**  
    - Optimal entry during slight ETH price dips to minimize IL risk  
    - Position sizing: conservative 1-3% of DeFi portfolio  
    - Setup Process: Deposit equal USD value ETH and USDC into Uniswap V2 pool via official interface; no staking required

    **Position Management:**  
    - Monitoring Schedule: Weekly tracking of ETH price, APY, TVL  
    - Rebalancing Triggers: Consider reducing exposure if ETH price moves over 30% from entry to limit IL  
    - Performance Benchmarks: Target net APY >4% after fees and IL

    **Exit Strategy:**  
    - Profit-Taking Levels: Partial exit after 10-15% portfolio gain or APY drops below 3%  
    - Stop-Loss Conditions: Full exit if IL exceeds 15% or major ETH market downturn  
    - Market Change Triggers: Significant protocol upgrade or regulatory concerns

    **👤 Investor Suitability**

    **Ideal For:**  
    - Investors seeking moderate yield with low to medium risk  
    - Beginners and intermediates with DeFi experience

    **Not Suitable For:**  
    - Traders looking for high yield or short-term gains  
    - Investors averse to any market volatility

    **Alternative Strategies:**  
    - Stablecoin-only pools for lower risk but reduced yield  
    - Concentrated liquidity pools for advanced users seeking higher returns

    **📈 Outlook & Final Verdict**  
    - Short-term Outlook: Stable fee income expected; moderate ETH volatility persists  
    - Medium-term Outlook: Dependent on ETH price trends and DeFi ecosystem growth  
    - Overall Recommendation: Hold with moderate confidence  
    - Key Success Factors: Stable volume and ETH price support  
    - Major Risk Concerns: ETH market crashes, unexpected protocol bugs

    **⚡ Action Items**  
    1. Verify official Uniswap V2 documentation and audit reports  
    2. Use impermanent loss calculators before entry  
    3. Set up alerts for ETH price movements and pool TVL changes

    ---

    Please generate the full report now using the data provided above and the exact structure shown in the example.
    """

    logger.info(f"Sending prompt to OpenAI for in-depth analysis of {pool_symbol.upper()}")
    # print(f"AI Summary Prompt:\n{prompt}\n") # For debugging, uncomment if needed
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini", # Using gpt-4o-mini as specified by user
            messages=[
                {"role": "system", "content": "You are an expert DeFi yield strategist providing detailed, actionable, and risk-aware investment analysis following a precise structure."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000, # Increased max_tokens for a more detailed report
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate in-depth AI analysis: {e}")
        return "An error occurred while generating the in-depth analysis report."


# --- Unified Report Interface ---
def generate_pool_analysis(pool_data: dict, token_info: str = "", news_data: list = None, report_type: str = "breakdown") -> str:
    """
    Unified function to generate different types of DeFi pool analysis reports.
    
    Args:
        pool_data (dict): Dictionary containing DeFi pool metrics.
        token_info (str): Additional information about the pool's tokens.
        news_data (list): List of recent news articles.
        report_type (str): Type of report to generate ("comprehensive" or "breakdown").
    
    Returns:
        str: The generated report.
    """
    if report_type == "comprehensive":
        # The 'pool' key from DefiLlama data typically forms part of the URL path
        pool_url = f"https://defillama.com/yields/pool/{pool_data.get('pool', '')}"
        if not pool_data.get('pool'):
            return "Pool ID missing in pool_data for comprehensive report."
        return generate_comprehensive_investment_report(pool_url)
    elif report_type == "breakdown":
        # Call the new in-depth analysis function
        return generate_in_depth_ai_analysis_report(pool_data, token_info, news_data)
    else:
        return f"Invalid report type: {report_type}. Choose 'comprehensive' or 'breakdown'."

# --- Legacy Bridge (Maintains Backward Compatibility) ---
# These aliases ensure that any existing calls to these functions in other parts of the application
# will correctly route to the new/updated implementations.
@st.cache_data(ttl=3600, show_spinner=False)
def summarize_pool_info(pool_url: str) -> str:
    """Legacy alias for generate_comprehensive_investment_report."""
    return generate_comprehensive_investment_report(pool_url)

@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_summary_for_pool(pool, token_info, raw_news_data):
    """
    Legacy alias for generate_in_depth_ai_analysis_report.
    This function now directly calls the new, comprehensive in-depth analysis report function.
    """
    return generate_in_depth_ai_analysis_report(pool, token_info, raw_news_data)

import requests

# --- CoinGecko Token Info Fetcher ---
import requests
import logging

logger = logging.getLogger(__name__)

def get_token_info_from_coingecko(symbol_or_name: str) -> dict:
    """
    Fetches detailed and current token information from CoinGecko by symbol or name.
    This function now directly queries for the specific token's market data,
    which is more efficient than fetching the entire coin list.
    """
    if not symbol_or_name:
        logger.warning("[CoinGecko] No symbol or name provided.")
        return {}

    try:
        # The CoinGecko API is case-insensitive for symbols, but we convert to lower for consistency.
        symbol_lower = symbol_or_name.lower()

        # The /coins/markets endpoint is more direct for getting data for a specific symbol.
        # We still need to get the full list to find the correct 'id' for the given symbol.
        list_response = requests.get("https://api.coingecko.com/api/v3/coins/list")
        list_response.raise_for_status()
        token_list = list_response.json()

        token_id = None
        for token in token_list:
            if token["symbol"].lower() == symbol_lower or token["name"].lower() == symbol_lower:
                token_id = token["id"]
                break

        if not token_id:
            logger.warning(f"[CoinGecko] Token '{symbol_or_name}' not found in the master list.")
            return {}

        # Now, fetch the detailed market data for the found token_id
        # This endpoint provides current, real-time data.
        market_response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": token_id
            }
        )
        market_response.raise_for_status()
        token_data = market_response.json()

        if not token_data:
            logger.warning(f"[CoinGecko] No market data found for token ID: {token_id}")
            return {}

        # Since we get a list back, even for one id, we take the first element.
        data = token_data[0]

        # To get the homepage and contract address, we still need the /coins/{id} endpoint
        details_response = requests.get(f"https://api.coingecko.com/api/v3/coins/{token_id}")
        details_response.raise_for_status()
        details_data = details_response.json()

        return {
            "name": data.get("name", "N/A"),
            "symbol": data.get("symbol", "N/A").upper(),
            "current_price_usd": data.get("current_price", "N/A"),
            "market_cap_usd": data.get("market_cap", "N/A"),
            "total_volume_usd": data.get("total_volume", "N/A"),
            "homepage": details_data.get("links", {}).get("homepage", [""])[0],
            "contract_address": details_data.get("asset_platform_id") and details_data.get("contract_address"),
            "coingecko_url": f"https://www.coingecko.com/en/coins/{token_id}"
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[CoinGecko] API request failed for {symbol_or_name}: {e}")
        return {}
    except (KeyError, IndexError) as e:
        logger.error(f"[CoinGecko] Error parsing data for {symbol_or_name}: {e}")
        return {}
    """
    Fetches detailed token information from CoinGecko by symbol or name.
    """
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/list")
        response.raise_for_status()
        token_list = response.json()

        symbol_or_name = symbol_or_name.lower()
        token_id = None
        for token in token_list:
            if token["symbol"].lower() == symbol_or_name or token["name"].lower() == symbol_or_name:
                token_id = token["id"]
                break

        if not token_id:
            logger.warning(f"[CoinGecko] Token '{symbol_or_name}' not found.")
            return {}

        token_data = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{token_id}",
            params={"localization": "false", "tickers": "false", "market_data": "true"}
        ).json()

        market_data = token_data.get("market_data", {})
        return {
            "name": token_data.get("name", ""),
            "symbol": token_data.get("symbol", ""),
            "current_price_usd": market_data.get("current_price", {}).get("usd", "N/A"),
            "market_cap_usd": market_data.get("market_cap", {}).get("usd", "N/A"),
            "total_volume_usd": market_data.get("total_volume", {}).get("usd", "N/A"),
            "homepage": token_data.get("links", {}).get("homepage", [""])[0],
            "contract_address": token_data.get("platforms", {}).get("ethereum", ""),
            "coingecko_url": f"https://www.coingecko.com/en/coins/{token_id}"
        }

    except Exception as e:
        logger.error(f"[CoinGecko] Error fetching data for {symbol_or_name}: {e}")
        return {}
