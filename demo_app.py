import streamlit as st
import os
import pandas as pd
from youtube_scraper import run_youtube_scraper, extract_deFi_insights_with_gemini
from bluechip_landscape import run_bluechip_landscape
from crypto_reddit_analyzer import analyze_reddit_urls  # import your analysis function

# Helper function for DataFrame display, available to all sections
def show_df(df, caption=None):
    st.dataframe(pd.DataFrame(df), use_container_width=True)
    if caption:
        st.caption(caption)

st.set_page_config(page_title="Crypto DeepSearch Demo", layout="wide")

# Sidebar polish: logo, description, and style
with st.sidebar:
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='margin-bottom: 0;'>DeepSearch</h2>
            <p style='font-size: 0.95em; color: #888;'>Your AI-powered crypto research agent</p>
            <hr style='margin: 10px 0;'>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.header("Demo Selector")
    demo_type = st.sidebar.radio("Choose a demo:", ["YouTube Scraper", "Bluechip Landscape", "News/Reddit Insights"])

# Main title and subtitle
st.title("Crypto DeepSearch Agent Demo")
st.markdown("""
    <div style='font-size:1.1em; color:#555; margin-bottom: 1.5em;'>
        Explore DeFi, stablecoin, and crypto yield opportunities with AI-powered insights from YouTube, Reddit, and news sources.<br>
        <span style='color:#888;'>Select a demo from the sidebar to get started.</span>
    </div>
""", unsafe_allow_html=True)

if demo_type == "YouTube Scraper":
    st.header("🎥 YouTube Scraper: DeFi Yield Video Insights")
    st.markdown("""
        <span style='color:#666;'>Scrape YouTube for recent DeFi yield videos and extract protocols, strategies, and watchlist suggestions using Gemini LLM.</span>
    """, unsafe_allow_html=True)
    
    def show_analysis(analyses):
        for a in analyses:
            st.markdown(f"**{a['title']}** ([link]({a['url']}))\n\n{a['analysis']}")

    def show_valid_pools(valid_pools_structured):
        for proto in valid_pools_structured:
            st.markdown(f"### {proto['protocol']}")
            if not proto['found']:
                st.warning(f"Not found on DeFi Llama.")
                continue
            st.markdown(f"**TVL:** {proto['tvl'] if proto['tvl'] is not None else 'N/A'}  ")
            st.markdown(f"**Chains:** {', '.join(proto['chains']) if proto['chains'] else 'N/A'}  ")
            st.markdown(f"**Category:** {proto['category'] if proto['category'] else 'N/A'}  ")
            if proto['pools']:
                pool_df = pd.DataFrame(proto['pools'])
                st.dataframe(pool_df[['rank', 'symbol', 'chain', 'apy_str', 'info']], use_container_width=True)
            else:
                st.info("No yield pools found.")

    st.info("Click the button below to run the YouTube DeFi yield video scraper and analysis.")
    if st.button("🚀 Run YouTube Scraper"):
        with st.spinner("Running YouTube Scraper..."):
            results, analyses, valid_pools_structured = run_youtube_scraper()
            st.success(f"Scraped {len(results)} videos, {len(analyses)} Gemini analyses.")
            if results:
                st.subheader("Video Results")
                show_df(results)
            if analyses:
                st.subheader("Gemini LLM Analyses")
                show_analysis(analyses)
            if valid_pools_structured:
                st.subheader("Valid Pools from Watchlist (DeFi Llama)")
                show_valid_pools(valid_pools_structured)
    else:
        # Try to load from output files if available, in expanders
        with st.expander("Video Results (from file)"):
            if os.path.exists("youtube_yield_candidates.txt"):
                with open("youtube_yield_candidates.txt", "r", encoding="utf-8") as f:
                    st.text(f.read())
            else:
                st.info("No cached video results found.")
        with st.expander("Gemini LLM Analyses (from file)"):
            if os.path.exists("youtube_gemini_analysis.txt"):
                with open("youtube_gemini_analysis.txt", "r", encoding="utf-8") as f:
                    st.text(f.read())
            else:
                st.info("No cached Gemini analyses found.")
        with st.expander("Summary (from file)"):
            if os.path.exists("youtube_gemini_summary.txt"):
                with open("youtube_gemini_summary.txt", "r", encoding="utf-8") as f:
                    st.text(f.read())
            else:
                st.info("No cached summary found.")

elif demo_type == "Bluechip Landscape":
    st.header("💎 Bluechip Landscape: Protocol & Pool Curation")
    st.markdown("""
        <span style='color:#666;'>Fetch and filter DeFi protocols, then score stablecoin pools by risk and yield. Great for discovering top-tier DeFi opportunities.</span>
    """, unsafe_allow_html=True)
    st.info("Click the button below to run the Bluechip Landscape pipeline.")
    if st.button("🏆 Run Bluechip Landscape Pipeline"):
        with st.spinner("Running Bluechip Landscape pipeline..."):
            filtered_protocols, evaluated_pools = run_bluechip_landscape()
            st.success(f"Filtered {len(filtered_protocols)} protocols, evaluated {len(evaluated_pools)} pools.")
            if filtered_protocols:
                st.subheader("Filtered Protocols")
                show_df(filtered_protocols)
            if evaluated_pools:
                st.subheader("Evaluated Stablecoin Pools")
                show_df(evaluated_pools)
    else:
        # Try to load from output files if available, in expanders
        with st.expander("Filtered Protocols (from file)"):
            if os.path.exists("filtered_protocols.csv"):
                df = pd.read_csv("filtered_protocols.csv")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No cached filtered protocols found.")
        with st.expander("Evaluated Stablecoin Pools (from file)"):
            if os.path.exists("evaluated_stablecoin_pools.csv"):
                df = pd.read_csv("evaluated_stablecoin_pools.csv")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No cached evaluated pools found.")

elif demo_type == "News/Reddit Insights":
    st.header("📰 News/Reddit: Gemini-Powered Key Insights for Crypto Investors")
    st.markdown("""
        <span style='color:#666;'>Fetch recent crypto yield/stablecoin news and Reddit posts, then extract key insights for investors using Gemini LLM.</span>
    """, unsafe_allow_html=True)
    from agent.config import Config
    from agent.logger import get_logger
    from agent.keywords import get_stock_symbols, get_financial_keywords
    from agent.sources.reddit import search_newest_crypto_reddit
    from agent.sources.news import search_financial_news
    config = Config()
    logger = get_logger('streamlit', 'INFO')
    stock_symbols = get_stock_symbols()
    financial_keywords = get_financial_keywords()

    reddit_results = search_newest_crypto_reddit(config, logger, stock_symbols, financial_keywords)
    news_results = search_financial_news(config, logger, stock_symbols, financial_keywords)

    tabs = st.tabs(["Reddit", "News"])

    def gemini_news_prompt(text):
        return (
            "You're analyzing a news article or Reddit post relevant to DeFi, stablecoins, or crypto yields. "
            "Extract key actionable insights for a crypto investor. "
            "Summarize the main points, highlight any protocols, coins, or yield strategies mentioned, and note any risks or opportunities.\n\n"
            f"Content:\n\n{text}"
        )

    with tabs[0]:
        st.subheader("Reddit Posts 🗨️")
        st.markdown("<span style='color:#888;'>Latest Reddit posts about DeFi, stablecoins, and crypto yields.</span>", unsafe_allow_html=True)
        if reddit_results:
            df = pd.DataFrame(reddit_results)[['title', 'author', 'subreddit', 'score', 'num_comments', 'created_at', 'url']]
            show_df(df)
            reddit_urls = [row['url'] for row in reddit_results if 'reddit.com' in row['url']]
            if st.button("🤖 Analyze All Reddit Posts with Gemini"):
                with st.spinner("Analyzing Reddit posts for crypto relevance and summaries..."):
                    analysis_results = analyze_reddit_urls(reddit_urls)
                for result in analysis_results:
                    st.markdown(f"### [{result['url']}]({result['url']})")
                  
                    st.markdown(f"- **Summary:**\n\n{result['summary']}")
        else:
            st.info("No relevant Reddit posts found.")

    st.markdown("<hr style='margin: 1.5em 0;'>", unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("News Articles 📰")
        st.markdown("<span style='color:#888;'>Latest news articles about DeFi, stablecoins, and crypto yields.</span>", unsafe_allow_html=True)
        if news_results:
            df = pd.DataFrame(news_results)[['title', 'author', 'source_name', 'published_at', 'description', 'url']]
            show_df(df)
            selected = st.multiselect("Select News articles to analyze with Gemini:", df.index, format_func=lambda i: df.loc[i, 'title'])
            if selected:
                st.subheader("Gemini Insights for Selected News Articles")
                for idx in selected:
                    row = news_results[idx]
                    content = f"{row['title']}\n\n{row.get('description', '')}\n\nURL: {row['url']}"
                    api_key = os.getenv('GEMINI_API_KEY')
                    if api_key:
                        with st.spinner(f"Analyzing News article: {row['title']}"):
                            summary = extract_deFi_insights_with_gemini(gemini_news_prompt(content), api_key)
                        st.markdown(f"**{row['title']}** ([link]({row['url']}))\n\n{summary}")
                    else:
                        st.warning("No Gemini API key set.")
        else:
            st.info("No relevant news articles found.")