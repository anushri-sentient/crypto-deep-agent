import os
import logging
import asyncio
import re
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from openai import OpenAI
import streamlit as st # <--- ADDED: Needed for st.cache_data

# Configuration
# These environment variables are expected to be loaded by the main application's load_dotenv()
# and will be available when this module is imported.
CRAWLER_API = os.getenv("CRAWLER_API_ENDPOINT", "https://crawl4ai.dev.sentient.xyz/crawl_direct")
CRAWLER_AUTH_TOKEN = os.getenv("CRAWLER_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI client
# Initialize the client only if the API key is available
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not found. OpenAI functionality will be limited.")
    client = None # Set to None to handle cases where key is missing

# Crawl configuration
config = CrawlerRunConfig(wait_until="networkidle")

# --- Async Crawl ---
async def crawl_url_async(url: str) -> str:
    """Asynchronously crawls a given URL using crawl4ai and returns cleaned HTML."""
    if not CRAWLER_AUTH_TOKEN:
        logger.error("CRAWLER_AUTH_TOKEN not set. Cannot perform web crawl.")
        return "" # Return empty string if token is missing

    try:
        async with AsyncWebCrawler(api_key=CRAWLER_AUTH_TOKEN) as crawler:
            logger.info(f"Initiating AsyncWebCrawler for {url}")
            result = await crawler.arun(url=url, config=config)
            if result and result.cleaned_html:
                logger.info(f"Successfully crawled {url}")
                return result.cleaned_html
            else:
                logger.warning(f"No cleaned HTML returned for {url}")
                return ""
    except Exception as e:
        logger.error(f"Error during async crawl of {url}: {e}")
        return ""

# --- Sync wrapper for async crawl ---
def crawl_url(url: str) -> str:
    """Synchronously wraps the asynchronous crawl function to be callable from sync contexts."""
    # This pattern safely gets or creates a new event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: # No running loop is currently active
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # If the loop is already running (e.g., in a Streamlit context),
    # schedule the async task to run on it. Otherwise, run until complete.
    if loop.is_running():
        # Schedule the task on the existing loop and get its result
        # This is safe for Streamlit's single-threaded nature as it runs in a thread-safe manner
        future = asyncio.run_coroutine_threadsafe(crawl_url_async(url), loop)
        return future.result() # Wait for the result
    else:
        # Run a new event loop until the coroutine completes
        return loop.run_until_complete(crawl_url_async(url))

# --- Link post-filter to avoid social media and irrelevant links ---
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
    if not client:
        return "OpenAI client not initialized. API key missing for summarization.", []
    if not html:
        return "No content provided to summarize.", []

    # Truncate HTML for safety and to manage token usage, even for large context models
    truncated_html = html[:50000] # Approx 50k chars, well within most model limits (tokens are ~1:4 chars)

    prompt = (
        f"You are a highly skilled DeFi expert. The following is extracted HTML content from a DeFi pool or project page.\n\n"
        f"**Task:**\n"
        f"1.  **Summarize:** Provide a concise summary (2-4 sentences) of the project's purpose, its key features, how it generates yield, and any immediate risks mentioned or implied within the text. If the text is very generic or primarily points to other resources (e.g., just a list of links), mention that.\n"
        f"2.  **Extract Links:** Identify and list up to 2 **most relevant and official** links from the HTML text. Prioritize official project websites, whitepapers, audit reports, comprehensive documentation, or security analysis reports. Do NOT include social media (Twitter, Discord, Telegram, Reddit, etc.) or blockchain explorers (Etherscan, Arbiscan, etc.) in this list.\n"
        f"Format the output as follows:\n"
        f"SUMMARY: [Your concise summary here]\n"
        f"LINKS: [Link1, Link2 (if found, otherwise leave empty)]\n\n"
        f"**HTML Content:**\n{truncated_html}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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

# --- Final investor-focused summary ---
def final_investor_summary(all_summaries: list) -> str:
    """
    Combines multiple summaries into a final, comprehensive, investor-focused brief.
    Args:
        all_summaries (list): A list of tuples, where each tuple is (source_name, summary_text).
    Returns:
        str: The final, synthesized investment summary.
    """
    if not client:
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
        "1.  **Project & Pool Overview:** Briefly explain the project's core mission and the specific DeFi pool's function (e.g., lending, LP farming, staking). State the Total Value Locked (TVL) and how yield is generated.\n"
        "2.  **Key Features & Differentiators:** What makes this pool/project unique, innovative, or particularly attractive? Highlight any competitive advantages.\n"
        "3.  **Underlying Protocols & Assets:** Describe the underlying blockchain(s), assets (e.g., stablecoins, ETH, specific tokens), and other DeFi protocols (if any) it integrates with.\n"
        "4.  **Risk Analysis (CRITICAL):** This is the most important section. Detail potential risks including: smart contract vulnerabilities (audits?), impermanent loss (if applicable), oracle risks, economic exploits (e.g., flash loan attacks, tokenomics vulnerabilities), potential for rug pulls, and any specific protocol-level or market risks mentioned. Evaluate its overall security posture and decentralization level. Be explicitly cautious.\n"
        "5.  **Audit Status & Security:** Explicitly state if audit information was found and its implications (e.g., audited by CertiK, no recent audit found).\n"
        "6.  **Recommendation & Considerations:** Conclude with a clear, balanced, and actionable assessment for the investor. Is this a relatively safe/transparent investment or high-risk? What kind of investor is it suitable for (e.g., conservative, yield-hungry)? What are the key things an investor *must* watch out for and perform due diligence on before considering investment? **Avoid definitive 'invest/don't invest' advice, but provide a strong caution or qualified endorsement based on your risk assessment.** Emphasize diversification and only investing what one can afford to lose.\n\n"
        f"**Collected Information for Synthesis:**\n{combined_text}\n\n Please provide the report directly, without conversational intros.")

    try:
        response = client.chat.completions.create(
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

# --- Public function to import elsewhere ---
@st.cache_data(ttl=3600, show_spinner="Crawling web for more insights...") # <--- ADDED: Caching decorator
def summarize_pool_info(pool_url: str) -> str:
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
        logger.error(f"Invalid pool URL provided to summarize_pool_info: {pool_url}")
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

# --- Optional Main Execution for Testing ---
if __name__ == "__main__":
    # Ensure environment variables are loaded for local testing
    from dotenv import load_dotenv
    load_dotenv()

    # It's good practice to set up a minimal Streamlit app context for caching
    # when testing a cached function directly with `python crawler.py`.
    # In a full `streamlit run main.py`, this setup is managed by Streamlit.
    try:
        # st.set_page_config is necessary for st.cache_data to function outside a full streamlit run
        st.set_page_config(page_title="Crawler Test App") 
        st.header("Direct Crawler Test Output")

        # Example DefiLlama Pool URLs for testing
        test_urls = [
            "https://defillama.com/yields/pool/c4f1e434-3386-4f48-b0ac-dc89782dfa55", # Aave V3 Arbitrum USDC
            "https://defillama.com/yields/pool/bcf3b3e6-1414-4632-9c17-8e62d4957692", # Lido Staked ETH
            # "https://defillama.com/yields/pool/unknown-invalid-id", # Test invalid ID (should return error)
        ]

        for i, url in enumerate(test_urls):
            st.subheader(f"Test {i+1}: {url}")
            with st.spinner(f"Generating full pool summary for {url}..."):
                summary = summarize_pool_info(url)
            st.markdown("--- **Final Summary** ---")
            st.write(summary)
            
            # Test cache hit (should be very fast if the cache is working)
            st.markdown("--- **Testing Cache Hit (should be faster)** ---")
            with st.spinner(f"Generating full pool summary for {url} again (should be cached)..."):
                summary_cached = summarize_pool_info(url)
            st.write(summary_cached)
            st.markdown("---")
            st.success("Test complete.")

    except Exception as e:
        st.error(f"Error during test execution: {e}")
        logger.error(f"Test execution failed: {e}", exc_info=True) # Log full traceback