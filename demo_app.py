import streamlit as st
import os
import pandas as pd
import re
from youtube_scraper import run_youtube_scraper, extract_deFi_insights_with_gemini, CURATED_YOUTUBERS
from bluechip_landscape import run_bluechip_landscape
from crypto_reddit_analyzer import analyze_reddit_urls  # import your analysis function
from playwright.sync_api import sync_playwright
import subprocess
import sys

# Helper function for DataFrame display, available to all sections
def show_df(df, caption=None):
    st.dataframe(pd.DataFrame(df), use_container_width=True)
    if caption:
        st.caption(caption)

def extract_coins_from_analysis(analysis_text):
    """Extract coin names from Gemini analysis text"""
    coins = []
    
    # Common words to exclude (not actual coins)
    exclude_words = {
        'APR', 'APY', 'TVL', 'USD', 'USDC', 'USDT', 'DAI', 'ETH', 'BTC', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'BASE',
        'DEFI', 'CRYPTO', 'YIELD', 'STABLE', 'COIN', 'TOKEN', 'PROTOCOL', 'PLATFORM', 'STRATEGY', 'RISK', 'REWARD',
        'LENDING', 'BORROWING', 'LIQUIDITY', 'FARMING', 'STAKING', 'VAULT', 'POOL', 'SWAP', 'DEX', 'AMM', 'LP',
        'CURVE', 'AAVE', 'COMPOUND', 'YEARN', 'CONVEX', 'BALANCER', 'UNISWAP', 'SUSHISWAP', 'PENDLE', 'TOKEMAK',
        'KAMINO', 'MARGINFI', 'RAYDIUM', 'GAMMA', 'FRANCIUM', 'ORCA', 'SAVE', 'SPARKLEND', 'MULTIPLI', 'MERKL',
        'STAKEDAO', 'DOLOMITE', 'GEARBOX', 'EXACTLY', 'LATCH', 'SANDCLOCK', 'HYPERDRIVE', 'INTEGRAL', 'CLIPPER'
    }
    
    # Common patterns for coin mentions
    patterns = [
        r'\b[A-Z]{2,10}\b',  # 2-10 letter uppercase tokens
        r'\b[A-Z][a-z]+[A-Z][a-z]*\b',  # CamelCase tokens
        r'\$[A-Z]{2,10}\b',  # $SYMBOL format
    ]
    
    # Extract all potential matches
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, analysis_text)
        all_matches.extend(matches)
    
    # Process matches
    for match in all_matches:
        clean_match = match.replace('$', '').strip()
        if (len(clean_match) >= 2 and 
            clean_match not in exclude_words and
            not clean_match.isdigit()):
            coins.append(clean_match)
    
    # Filter out common non-coin words that might slip through
    filtered_coins = []
    for coin in coins:
        if (coin.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'into', 'during', 'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon'] and
            len(coin) >= 2):
            filtered_coins.append(coin)
    
    return filtered_coins

def display_video_with_thumbnail(video, index):
    """Display a video with thumbnail and selection checkbox"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if 'thumbnail_url' in video:
            st.image(video['thumbnail_url'], width=120)
        else:
            st.image("https://via.placeholder.com/120x90?text=No+Thumbnail", width=120)
    
    with col2:
        st.markdown(f"**{index + 1}. {video['title']}**")
        st.markdown(f"**Channel:** {video.get('channel', 'Unknown')}")
        st.markdown(f"**Published:** {video.get('published_at', 'Unknown')[:10]}")
        st.markdown(f"**URL:** [{video['url']}]({video['url']})")
        if 'description' in video:
            st.markdown(f"**Description:** {video['description'][:200]}...")
    
    return st.checkbox(f"Select video {index + 1}", key=f"video_{index}")

def setup_playwright():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        print("✅ Playwright browsers are already installed")
        return True
    except Exception as e:
        print(f"⚠️ Playwright browsers not found: {e}")
        print("🔧 Installing Playwright browsers...")
        try:
            subprocess.run([
                sys.executable, "-m", "playwright", "install", "chromium"
            ], check=True, capture_output=True)
            print("✅ Playwright browsers installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Playwright browsers: {e}")
            return False

def scrape_euler_strategies():
    if not setup_playwright():
        print("❌ Cannot proceed without Playwright browsers")
        return []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"})
        url = "https://app.euler.finance/strategies?collateralAsset=DAI%2CUSDT%2CUSDC&asset=USDT%2CDAI%2CUSDC&network=ethereum"
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            html = page.content()
            print("Failed to load page. HTML content (first 1000 chars):")
            print(html[:1000])
            raise

        page.wait_for_selector("div.asCards.MuiBox-root.css-1253gju", timeout=30000)

        cards_container = page.query_selector("div.asCards.MuiBox-root.css-1253gju")
        if not cards_container:
            print("Warning: No cards_container found!")
            return []
        strategy_divs = cards_container.query_selector_all("div.MuiBox-root.css-1oarypt")
        if not strategy_divs:
            print("Warning: No strategy_divs found!")
            return []

        data = []
        for div in strategy_divs:
            text = div.inner_text().strip()
            lines = text.split('\n')

            # Defensive: avoid index errors
            strategy = {
                "name": lines[0] if len(lines) > 0 else "",
                "tokens": lines[1] if len(lines) > 1 else "",
                "max_roe": None,
                "max_multiplier": None,
                "correlated": None,
                "liquidity": None,
            }
            if len(lines) < 2:
                print(f"Warning: Unexpected card format, lines: {lines}")

            # Find max ROE value (next line after "Max ROE")
            try:
                max_roe_index = lines.index("Max ROE") + 1
                strategy["max_roe"] = lines[max_roe_index]
            except ValueError:
                pass

            # Find max multiplier value (next line after "Max multiplier")
            try:
                max_mult_index = lines.index("Max multiplier") + 1
                strategy["max_multiplier"] = lines[max_mult_index]
            except ValueError:
                pass

            # Correlated value (next line after "Correlated")
            try:
                corr_index = lines.index("Correlated") + 1
                strategy["correlated"] = lines[corr_index]
            except ValueError:
                pass

            # Liquidity value (next line after "Liquidity")
            try:
                liq_index = lines.index("Liquidity") + 1
                strategy["liquidity"] = lines[liq_index]
            except ValueError:
                pass

            data.append(strategy)

        browser.close()
        return data

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
    demo_type = st.sidebar.radio(
        "Choose a demo:",
        ["YouTube Scraper", "Bluechip Landscape", "News/Reddit Insights", "Euler Strategies", "Exponential.fi Risk A & B Pools"]
    )

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
        <span style='color:#666;'>Scrape YouTube for recent DeFi yield videos from curated YouTubers and extract protocols, strategies, and watchlist suggestions using Gemini LLM.</span>
    """, unsafe_allow_html=True)
    
    # YouTube Scraper Options
    scraper_option = st.radio(
        "Choose scraping method:",
        ["Curated YouTubers (Recommended)", "Search-based (Original)"],
        help="Curated YouTubers focuses on trusted DeFi channels, while search-based finds videos by keywords."
    )
    
    if scraper_option == "Curated YouTubers (Recommended)":
        st.subheader("📺 Curated DeFi YouTubers")
        
        # Display YouTubers with selection
        st.markdown("**Select YouTubers to search:**")
        youtuber_cols = st.columns(2)
        selected_youtubers = []
        
        for i, (name, info) in enumerate(CURATED_YOUTUBERS.items()):
            with youtuber_cols[i % 2]:
                if st.checkbox(f"**{name}**", key=f"youtuber_{i}"):
                    selected_youtubers.append((name, info))
                st.markdown(f"🔗 [{info['url']}]({info['url']})")
                st.markdown(f"*{', '.join(info.get('search_terms', [name]))}*")
        
        if not selected_youtubers:
            st.info("Please select at least one YouTuber to search.")
            st.stop()
        
        # Video selection interface
        if st.button("🔍 Search for Videos"):
            with st.spinner("Searching for videos from selected YouTubers..."):
                try:
                    from googleapiclient.discovery import build
                    import datetime
                    
                    youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
                    
                    all_videos = []
                    for name, info in selected_youtubers:
                        st.markdown(f"**Searching: {name}**")
                        
                        # Use search terms to find videos
                        search_terms = info.get('search_terms', [name])
                        for term in search_terms:
                            search_response = youtube.search().list(
                                q=f"{term} stablecoin yield defi",
                                part='snippet',
                                type='video',
                                maxResults=5,
                                order='date',
                                publishedAfter=(datetime.datetime.utcnow() - datetime.timedelta(days=90)).isoformat("T") + "Z"
                            ).execute()
                            
                            videos = search_response.get('items', [])
                            for v in videos:
                                video_id = v['id']['videoId']
                                title = v['snippet']['title']
                                channel = v['snippet']['channelTitle']
                                published_at = v['snippet']['publishedAt']
                                thumbnail_url = v['snippet']['thumbnails']['medium']['url']
                                description = v['snippet']['description']
                                url = f"https://youtube.com/watch?v={video_id}"
                                
                                all_videos.append({
                                    'video_id': video_id,
                                    'title': title,
                                    'channel': channel,
                                    'published_at': published_at,
                                    'thumbnail_url': thumbnail_url,
                                    'description': description,
                                    'url': url
                                })
                    
                    # Remove duplicates
                    unique_videos = []
                    seen_ids = set()
                    for video in all_videos:
                        if video['video_id'] not in seen_ids:
                            unique_videos.append(video)
                            seen_ids.add(video['video_id'])
                    
                    st.session_state['available_videos'] = unique_videos
                    st.success(f"Found {len(unique_videos)} unique videos!")
                    
                except Exception as e:
                    st.error(f"Error searching for videos: {e}")
                    st.info("Make sure you have set the YOUTUBE_API_KEY environment variable.")
                    st.stop()
        
        # Display videos for selection
        if 'available_videos' in st.session_state and st.session_state['available_videos']:
            st.subheader("📹 Available Videos")
            st.markdown("**Select videos to analyze:**")
            
            selected_videos = []
            for i, video in enumerate(st.session_state['available_videos']):
                if display_video_with_thumbnail(video, i):
                    selected_videos.append(video)
            
            if selected_videos:
                st.subheader("🎯 Selected Videos for Analysis")
                st.markdown(f"**{len(selected_videos)} videos selected**")
                
                if st.button("🤖 Analyze Selected Videos"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    analyses = []
                    all_coins = set()
                    
                    for i, video in enumerate(selected_videos):
                        # Update progress
                        progress = (i + 1) / len(selected_videos)  # Value between 0.0 and 1.0
                        progress_bar.progress(progress)
                        status_text.text(f"Analyzing video {i+1} of {len(selected_videos)}: {video['title']}")
                        
                        st.markdown(f"### 📺 {video['title']}")
                        st.markdown(f"**Channel:** {video['channel']} | **URL:** [{video['url']}]({video['url']})")
                        
                        # Get actual transcript using AssemblyAI
                        transcript = ""
                        assemblyai_key = os.getenv('ASSEMBLYAI_API_KEY')
                        
                        if assemblyai_key:
                            with st.spinner(f"Extracting transcript for: {video['title']}"):
                                try:
                                    from youtube_scraper import get_transcript_assemblyai
                                    transcript = get_transcript_assemblyai(video['url'], assemblyai_key, lang='en')
                                    
                                    if transcript:
                                        st.success("✅ Transcript extracted successfully (AssemblyAI)")
                                        # Show transcript preview
                                        with st.expander("📝 View Transcript"):
                                            st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
                                    else:
                                        st.warning("⚠️ AssemblyAI failed, trying YouTube Transcript API...")
                                        # Fallback to YouTube Transcript API
                                        try:
                                            from youtube_scraper import get_transcript_youtube_api
                                            video_id = video['url'].split('=')[-1]
                                            transcript = get_transcript_youtube_api(video_id)
                                            if transcript:
                                                st.success("✅ Transcript extracted successfully (YouTube API)")
                                                with st.expander("📝 View Transcript"):
                                                    st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
                                            else:
                                                st.warning("⚠️ No transcript available for this video")
                                                transcript = f"Video transcript for: {video['title']}\n\nNo transcript could be extracted for this video. This could be due to:\n- Transcripts disabled for this video\n- Video is too long or has audio issues\n- Both AssemblyAI and YouTube API failed"
                                        except Exception as e2:
                                            st.error(f"❌ YouTube API also failed: {e2}")
                                            transcript = f"Video transcript for: {video['title']}\n\nError extracting transcript: {e2}"
                                except Exception as e:
                                    st.error(f"❌ Error with AssemblyAI: {e}")
                                    # Fallback to YouTube Transcript API
                                    try:
                                        st.info("🔄 Trying YouTube Transcript API as fallback...")
                                        from youtube_scraper import get_transcript_youtube_api
                                        video_id = video['url'].split('=')[-1]
                                        transcript = get_transcript_youtube_api(video_id)
                                        if transcript:
                                            st.success("✅ Transcript extracted successfully (YouTube API fallback)")
                                            with st.expander("📝 View Transcript"):
                                                st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
                                        else:
                                            st.warning("⚠️ No transcript available for this video")
                                            transcript = f"Video transcript for: {video['title']}\n\nNo transcript could be extracted for this video."
                                    except Exception as e2:
                                        st.error(f"❌ Both methods failed: {e2}")
                                        transcript = f"Video transcript for: {video['title']}\n\nError extracting transcript: {e2}"
                        else:
                            st.info("ℹ️ No AssemblyAI API key, trying YouTube Transcript API...")
                            try:
                                from youtube_scraper import get_transcript_youtube_api
                                video_id = video['url'].split('=')[-1]
                                transcript = get_transcript_youtube_api(video_id)
                                if transcript:
                                    st.success("✅ Transcript extracted successfully (YouTube API)")
                                    with st.expander("📝 View Transcript"):
                                        st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
                                else:
                                    st.warning("⚠️ No transcript available for this video")
                                    transcript = f"Video transcript for: {video['title']}\n\nNo transcript could be extracted for this video. To get better results, please set the ASSEMBLYAI_API_KEY environment variable."
                            except Exception as e:
                                st.error(f"❌ YouTube API failed: {e}")
                                transcript = f"Video transcript for: {video['title']}\n\nError extracting transcript: {e}"
                        
                        # Analyze with Gemini
                        api_key = os.getenv('GEMINI_API_KEY')
                        if api_key:
                            try:
                                analysis = extract_deFi_insights_with_gemini(transcript, api_key)
                                analyses.append({
                                    'title': video['title'],
                                    'url': video['url'],
                                    'analysis': analysis
                                })
                                
                                # Extract coins
                                coins_info = extract_coins_from_analysis(analysis)
                                all_coins.update(coins_info)
                                
                                # Display analysis
                                st.markdown("**🔍 Analysis:**")
                                st.markdown(analysis)
                                
                                if coins_info:
                                    st.markdown("**🪙 Coins/Tokens Mentioned:**")
                                    coin_tags = " ".join([f"`{coin}`" for coin in coins_info])
                                    st.markdown(coin_tags)
                                
                                st.markdown("---")
                                
                            except Exception as e:
                                st.error(f"Error analyzing video: {e}")
                        else:
                            st.warning("No Gemini API key set. Please set GEMINI_API_KEY environment variable.")
                    
                    # Create a simple DataFrame for better display
                    coin_df = pd.DataFrame({
                        'Coin/Token': sorted(all_coins),
                        'Source Videos': [len([a for a in analyses if coin in a['analysis']]) for coin in sorted(all_coins)]
                    })
                    st.dataframe(coin_df, use_container_width=True)
                    
                    # Save results to session state
                    st.session_state['analyses'] = analyses
                    st.session_state['all_coins'] = list(all_coins)
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    st.success(f"🎉 Successfully analyzed {len(selected_videos)} videos and found {len(all_coins)} unique coins/tokens!")
            
            else:
                st.info("Please select at least one video to analyze.")
    
    else:  # Original search-based method
        st.subheader("🔍 Search-based YouTube Scraper")
        st.info("Click the button below to run the original YouTube DeFi yield video scraper and analysis.")
        
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

        if st.button("🚀 Run YouTube Scraper"):
            with st.spinner("Running YouTube Scraper..."):
                results, analyses, valid_pools_structured = run_youtube_scraper()
                st.success(f"Scraped {len(results)} videos, {len(analyses)} Gemini analyses.")
                if results:
                    st.subheader("Video Results")
                    show_df(results)
                if analyses:
                    st.subheader("LLM Analyses")
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
            with st.expander("LLM Analyses (from file)"):
                if os.path.exists("youtube_gemini_analysis.txt"):
                    with open("youtube_gemini_analysis.txt", "r", encoding="utf-8") as f:
                        st.text(f.read())
                else:
                    st.info("No cached analyses found.")
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
    st.header("📰 News/Reddit: Key Insights for Crypto Investors")
    st.markdown("""
        <span style='color:#666;'>Fetch recent crypto yield/stablecoin news and Reddit posts, then extract key insights for investors using LLM.</span>
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
            if st.button("🤖 Analyze All Reddit Posts "):
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
            selected = st.multiselect("Select News articles to analyze :", df.index, format_func=lambda i: df.loc[i, 'title'])
            if selected:
                st.subheader(" Insights for Selected News Articles")
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

elif demo_type == "Euler Strategies":
    st.header("🏦 Euler Strategies: DeFi Lending Opportunities")
    st.markdown("""
        <span style='color:#666;'>Scrape and analyze Euler Finance lending strategies to discover high-yield DeFi opportunities with risk metrics and liquidity information.</span>
    """, unsafe_allow_html=True)
    
    st.info("Click the button below to scrape Euler Finance strategies from their web app.")
    
    if st.button("🔍 Scrape Euler Strategies"):
        with st.spinner("Scraping Euler Finance strategies..."):
            try:
                strategies = scrape_euler_strategies()
                st.success(f"Successfully scraped {len(strategies)} Euler strategies!")
                
                if strategies:
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(strategies)
                    
                    # Display the data
                    st.subheader("📊 Euler Finance Strategies")
                    st.dataframe(df, use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader("📈 Strategy Summary")
                    
                    # Count strategies by name
                    strategy_counts = df['name'].value_counts()
                    st.markdown("**Strategy Types:**")
                    for strategy, count in strategy_counts.items():
                        st.markdown(f"- **{strategy}**: {count} instances")
                    
                    # Show tokens used
                    if 'tokens' in df.columns:
                        all_tokens = []
                        for tokens in df['tokens'].dropna():
                            all_tokens.extend([t.strip() for t in tokens.split('/')])
                        
                        token_counts = pd.Series(all_tokens).value_counts()
                        if not token_counts.empty:
                            st.markdown("**Most Common Tokens:**")
                            for token, count in token_counts.head(10).items():
                                st.markdown(f"- **{token}**: {count} times")
                    
                    # Save to session state for potential further analysis
                    st.session_state['euler_strategies'] = strategies
                    st.session_state['euler_strategies_df'] = df
                    
                    # Option to analyze with Gemini
                    if st.button("🤖 Analyze Strategies"):
                        api_key = os.getenv('GEMINI_API_KEY')
                        if api_key:
                            with st.spinner("Analyzing strategies..."):
                                # Create a summary of the strategies for analysis
                                strategy_summary = f"""
                                Euler Finance Strategies Analysis:
                                
                                Total strategies found: {len(strategies)}
                                
                                Strategy breakdown:
                                {strategy_counts.to_string()}
                                
                                Detailed strategy data:
                                {df.to_string()}
                                
                                Please analyze these Euler Finance strategies and provide insights on:
                                1. Which strategies look most promising for yield farming
                                2. Risk assessment based on the metrics shown
                                3. Recommendations for DeFi investors
                                4. Any notable patterns or trends in the data
                                """
                                
                                analysis = extract_deFi_insights_with_gemini(strategy_summary, api_key)
                                st.markdown("**🔍 Analysis:**")
                                st.markdown(analysis)
                        else:
                            st.warning("No Gemini API key set. Please set GEMINI_API_KEY environment variable.")
                
            except ModuleNotFoundError as e:
                st.error(f"Playwright not installed: {e}")
                st.info("Make sure you have Playwright installed: `pip install playwright` and run `playwright install chromium`")
            except Exception as e:
                st.error(f"Error scraping Euler strategies: {e}")
                st.info("This is likely a parsing or data error, not a Playwright error.")
    else:
        # Try to load from session state if available
        if 'euler_strategies' in st.session_state:
            st.subheader("📊 Previously Scraped Euler Strategies")
            df = st.session_state['euler_strategies_df']
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No cached Euler strategies found. Click the button above to scrape fresh data.")

elif demo_type == "Exponential.fi Risk A & B Pools":
    st.header("📊 Exponential.fi Pools: Risk A & B, Sorted by APY (with DeFi Llama match)")
    st.markdown("""
        <span style='color:#666;'>Scrape and display pools from Exponential.fi with risk ratings A & B, sorted by highest APY. Pools are matched to DeFi Llama for additional data.</span>
    """, unsafe_allow_html=True)
    if st.button("Run Exponential.fi + DeFi Llama Integration"):
        with st.spinner("Scraping Exponential.fi pools and matching to DeFi Llama..."):
            try:
                from exponential_scraper import scrape_and_match_exponential_to_defillama
                df = scrape_and_match_exponential_to_defillama()
            except Exception as e:
                st.error(f"Failed to scrape Exponential.fi or match to DeFi Llama: {e}")
                df = None
        if df is not None and not df.empty:
            st.write(f"Showing all pools by APY (with DeFi Llama match):")
            st.dataframe(df[['Pool', 'Chain', 'TVL', 'Risk', 'Yield', 'APY', 'DeFiLlama Project', 'DeFiLlama TVL', 'DeFiLlama APY', 'DeFiLlama URL']])
        else:
            st.warning("No pools found or failed to scrape data.")