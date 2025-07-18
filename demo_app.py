# Standard library
import os
import re
import sys
import subprocess

# Third-party
import streamlit as st
import pandas as pd
import requests

# Local
from youtube_scraper import run_youtube_scraper, extract_deFi_insights_with_gemini, CURATED_YOUTUBERS
from bluechip_landscape import run_bluechip_landscape
from crypto_reddit_analyzer import analyze_reddit_urls  # import your analysis function
from playwright.sync_api import sync_playwright
from functools import lru_cache

# DeFiLlama Yield Rankings Integration
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_yield_rankings():
    """Fetch and cache DeFiLlama yield rankings dataset"""
    url = "https://datasets.llama.fi/yields/yield_rankings.csv"
    try:
        df = pd.read_csv(url)
        # Debug: Print column information
        print(f"Dataset loaded with {len(df)} rows and columns: {list(df.columns)}")
        return df
    except Exception as e:
        st.error(f"Failed to fetch yield rankings: {e}")
        return None

def filter_rankings_by_token(df, token_symbol, il_risk_filter=None):
    """Filter yield rankings by token symbol and IL risk"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Convert token symbol to uppercase for case-insensitive matching
    token_upper = token_symbol.upper()
    
    # Filter pools that contain the token in their symbol or underlying tokens
    filtered_df = df[
        df['symbol'].str.contains(token_upper, case=False, na=False) |
        df['underlyingTokens'].str.contains(token_upper, case=False, na=False)
    ].copy()
    
    # Apply IL risk filter if specified
    if il_risk_filter and il_risk_filter != "All":
        if il_risk_filter == "No IL Risk":
            filtered_df = filtered_df[filtered_df['ilRisk'] == 'no']
        elif il_risk_filter == "Has IL Risk":
            filtered_df = filtered_df[filtered_df['ilRisk'] == 'yes']
        elif il_risk_filter == "Unknown IL Risk":
            filtered_df = filtered_df[filtered_df['ilRisk'].isna() | (filtered_df['ilRisk'] != 'yes') & (filtered_df['ilRisk'] != 'no')]
    
    return filtered_df

def classify_pool_from_rankings(row):
    """Classify pool as Staking or Pooling based on rankings data"""
    symbol = str(row.get('symbol', '')).lower()
    underlying_tokens = str(row.get('underlyingTokens', '')).lower()
    pool_meta = str(row.get('poolMeta', '')).lower()
    
    # Check for single asset indicators
    if (
        'single' in pool_meta or
        'staking' in pool_meta or
        'lend' in pool_meta or
        len(str(underlying_tokens).split(',')) == 1 or
        'ilRisk' in row and row['ilRisk'] == 'no'
    ):
        return 'Staking'
    return 'Pooling'

def classify_il_risk(row):
    """Classify IL risk based on ilRisk field"""
    il_risk = row.get('ilRisk', '')
    if pd.isna(il_risk) or il_risk == '':
        return 'Unknown'
    elif il_risk == 'no':
        return 'No IL Risk'
    elif il_risk == 'yes':
        return 'Has IL Risk'
    else:
        return 'Unknown'

def render_rankings_protocol_card(protocol_name):
    """Render protocol card for yield rankings data"""
    # Try to fetch protocol info from DeFiLlama API
    protocol_info = cached_fetch_protocol_info(protocol_name)
    
    if protocol_info and not protocol_info.get('error'):
        cols = st.columns([1, 4])
        with cols[0]:
            if protocol_info.get('logo'): 
                st.image(protocol_info['logo'], width=80)
            else:
                st.image("https://via.placeholder.com/80x80?text=?", width=80)
        with cols[1]:
            st.markdown(f"### {protocol_info.get('name', protocol_name).title()}")
            st.markdown(f"**Category:** {protocol_info.get('category', 'N/A')}")
            st.markdown(f"**Chains:** {', '.join(protocol_info.get('chains', []))}")
            if protocol_info.get('description'):
                st.markdown(f"<span style='color:#555'>{protocol_info['description']}</span>", unsafe_allow_html=True)
            if protocol_info.get('url'):
                st.markdown(f"[Website]({protocol_info['url']})", unsafe_allow_html=True)
            if protocol_info.get('twitter'):
                st.markdown(f"[Twitter]({protocol_info['twitter']})", unsafe_allow_html=True)
    else:
        st.markdown(f"### {protocol_name.title()}")
        st.info("Protocol information not available")

def format_rankings_data(df):
    """Format yield rankings data for display"""
    if df.empty:
        return pd.DataFrame()
    
    # Add classification and formatting
    df['category'] = df.apply(classify_pool_from_rankings, axis=1)
    df['il_risk_classified'] = df.apply(classify_il_risk, axis=1)
    df['apy_formatted'] = df['apy'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    df['tvl_formatted'] = df['tvlUsd'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    
    # Define display columns and check which ones exist
    display_columns = {
        'symbol': 'Pool',
        'chain': 'Chain',
        'project': 'Protocol',
        'apy_formatted': 'APY',
        'tvl_formatted': 'TVL',
        'category': 'Type',
        'il_risk_classified': 'IL Risk',
        'pool': 'Pool ID'
    }
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in display_columns.keys() if col in df.columns]
    filtered_display_columns = {col: display_columns[col] for col in available_columns}
    
    return df[available_columns].rename(columns=filtered_display_columns)

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
        for i, div in enumerate(strategy_divs):
            try:
                # Initialize strategy with default values
                strategy = {
                    "name": f"Strategy {i+1}",
                    "tokens": "",
                    "max_roe": "",
                    "max_multiplier": "",
                    "correlated": "",
                    "liquidity": ""
                }
                
                # Get the full text content of the card for debugging
                full_text = div.inner_text().strip()
                print(f"Strategy {i+1} full text: {full_text[:200]}...")
                
                # Try to extract strategy name from various possible elements
                name_selectors = [
                    "h6", "h5", "h4", 
                    ".MuiTypography-h6", ".MuiTypography-h5", ".MuiTypography-h4",
                    "[data-testid='strategy-name']", "[class*='strategy-name']",
                    "div[class*='title']", "div[class*='name']"
                ]
                
                for selector in name_selectors:
                    name_elem = div.query_selector(selector)
                    if name_elem:
                        name_text = name_elem.inner_text().strip()
                        if name_text and name_text != "Max ROE" and name_text != "Points":
                            strategy["name"] = name_text
                            print(f"Found name: {name_text}")
                            break
                
                # Try to extract all text elements and parse them
                all_elements = div.query_selector_all("*")
                text_elements = []
                for elem in all_elements:
                    text = elem.inner_text().strip()
                    if text and len(text) > 1:
                        text_elements.append(text)
                
                # Look for patterns in the text elements
                for j, text in enumerate(text_elements):
                    text_lower = text.lower()
                    
                    # Look for percentage values (likely APY/ROE)
                    if '%' in text or 'APY' in text_lower or 'ROE' in text_lower:
                        if 'max' in text_elements[j-1].lower() if j > 0 else False:
                            strategy["max_roe"] = text
                        elif 'roe' in text_elements[j-1].lower() if j > 0 else False:
                            strategy["max_roe"] = text
                    
                    # Look for multiplier values
                    if 'x' in text or 'multiplier' in text_lower:
                        strategy["max_multiplier"] = text
                    
                    # Look for correlated values
                    if 'correlated' in text_lower:
                        if j + 1 < len(text_elements):
                            strategy["correlated"] = text_elements[j + 1]
                    
                    # Look for liquidity values
                    if 'liquidity' in text_lower:
                        if j + 1 < len(text_elements):
                            strategy["liquidity"] = text_elements[j + 1]
                    
                    # Look for token pairs (containing common tokens)
                    if any(token in text.upper() for token in ['USDC', 'USDT', 'DAI', 'ETH', 'BTC']):
                        if '/' in text or '\\' in text:
                            strategy["tokens"] = text
                
                # Try to extract label-value pairs more systematically
                rows = div.query_selector_all("div.MuiBox-root")
                for row in rows:
                    spans = row.query_selector_all("span.MuiTypography-root")
                    if len(spans) >= 2:
                        label = spans[0].inner_text().strip().replace(":", "").lower()
                        value = spans[1].inner_text().strip()
                        
                        # Map common labels to strategy fields
                        if "max roe" in label or "roe" in label:
                            strategy["max_roe"] = value
                        elif "max multiplier" in label or "multiplier" in label:
                            strategy["max_multiplier"] = value
                        elif "correlated" in label:
                            strategy["correlated"] = value
                        elif "liquidity" in label:
                            strategy["liquidity"] = value
                        elif "tokens" in label or "asset" in label or "pair" in label:
                            strategy["tokens"] = value
                        elif "strategy" in label or "name" in label:
                            strategy["name"] = value
                
                # Clean up empty values
                for key in strategy:
                    if strategy[key] == "":
                        strategy[key] = "N/A"
                
                data.append(strategy)
                print(f"Processed strategy {i+1}: {strategy}")
                
            except Exception as e:
                print(f"Error processing strategy {i+1}: {e}")
                # Add a fallback strategy entry
                data.append({
                    "name": f"Strategy {i+1} (Error)",
                    "tokens": "Error parsing",
                    "max_roe": "Error",
                    "max_multiplier": "Error",
                    "correlated": "Error",
                    "liquidity": "Error"
                })

        browser.close()
        return data

def fetch_pool_details(pool_id):
    url = f"https://yields.llama.fi/pools/{pool_id}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Failed to fetch details"}

def fetch_pool_chart(pool_id):
    url = f"https://yields.llama.fi/chart/{pool_id}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Failed to fetch chart data"}

def fetch_protocol_info(protocol_slug):
    url = f"https://api.llama.fi/protocol/{protocol_slug}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Failed to fetch protocol info"}

def render_protocol_card(protocol_slug):
    info = fetch_protocol_info(protocol_slug)
    if info and not info.get('error'):
        cols = st.columns([1, 4])
        with cols[0]:
            if info.get('logo'): st.image(info['logo'], width=80)
        with cols[1]:
            st.markdown(f"### {info.get('name', protocol_slug).title()}")
            st.markdown(f"**Category:** {info.get('category', 'N/A')}")
            st.markdown(f"**Chains:** {', '.join(info.get('chains', []))}")
            if info.get('description'):
                st.markdown(f"<span style='color:#555'>{info['description']}</span>", unsafe_allow_html=True)
            if info.get('url'):
                st.markdown(f"[Website]({info['url']})", unsafe_allow_html=True)
            if info.get('twitter'):
                st.markdown(f"[Twitter]({info['twitter']})", unsafe_allow_html=True)
            if info.get('audit_links'):
                for link in info['audit_links']:
                    st.markdown(f"[Audit]({link})", unsafe_allow_html=True)
    else:
        st.info("No protocol info available.")

@st.cache_data(show_spinner=False)
def cached_fetch_pool_details(pool_id):
    return fetch_pool_details(pool_id)

@st.cache_data(show_spinner=False)
def cached_fetch_pool_chart(pool_id):
    return fetch_pool_chart(pool_id)

@st.cache_data(show_spinner=False)
def cached_fetch_protocol_info(protocol_slug):
    return fetch_protocol_info(protocol_slug)

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
        ["YouTube Scraper", "Bluechip Landscape", "News/Reddit Insights", "Euler Strategies", "Exponential.fi Risk A & B Pools", "Find Pools by Token", "DeFiLlama Yield Rankings"]
    )
    # Add user mode selector
    user_mode = st.sidebar.radio(
        "User Mode:",
        ["Beginner", "Advanced"],
        help="Beginner: Guided, simple. Advanced: Full filters and analytics."
    )
    st.session_state['user_mode'] = user_mode

# Onboarding logic
if st.session_state.get('user_mode', 'Beginner') == 'Beginner':
    st.info("""
        👋 **Welcome, Beginner!**
        
        This app will guide you through finding the best DeFi yield and staking opportunities.
        
        **How would you like to get started?**
    """)
    onboarding_choice = st.radio(
        "Choose an option:",
        ["I have a token and want to find staking/yield options", "Show me the best opportunities for beginners", "Learn about DeFi yields and risks"]
    )
    st.session_state['onboarding_choice'] = onboarding_choice
    st.write(f"You selected: {onboarding_choice}")
    # (Later: Use onboarding_choice to drive personalized flows)
else:
    st.success("Advanced mode: All filters and analytics enabled.")
    # (Later: Show advanced filter widgets and analytics)

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
                                    transcript = f"Video transcript for: {video['title']}\n\nError extracting transcript: {e}"
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
                    
                    # Count strategies by name (handle missing column gracefully)
                    if 'name' in df.columns:
                        strategy_counts = df['name'].value_counts()
                        st.markdown("**Strategy Types:**")
                        for strategy, count in strategy_counts.items():
                            st.markdown(f"- **{strategy}**: {count} instances")
                    else:
                        st.info("Strategy names not available in the scraped data.")
                    
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
                                {strategy_counts.to_string() if 'name' in df.columns else 'Strategy names not available'}
                                
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

elif demo_type == "Find Pools by Token":
    st.header("🔎 Find Pools by Token (DeFiLlama API)")
    st.markdown("""
        Enter a token (e.g., USDC, ETH, DAI) to find all yield, staking, and LP pools from DeFiLlama that include your token.
        
        **Note:** Most pools are LPs (liquidity pools), but some are single-asset staking or lending. Results are grouped by protocol. Click a pool to see full analytics, protocol info, and APY history.
    """)
    st.info("💡 **Research Tip:** Use the protocol card, pool summary, and APY chart to compare risk, yield, and platform reputation. Click the DeFiLlama link for even deeper analytics.")
    token = st.text_input("Enter token symbol (e.g., USDC, ETH, DAI):", "USDC")
    if st.button("Find Pools"):
        from exponential_scraper import get_all_yield_pools_from_defillama
        with st.spinner(f"Fetching pools for {token} from DeFiLlama..."):
            pools = get_all_yield_pools_from_defillama(token_filter=token)
        if pools:
            def classify_pool(p):
                pool_type = p.extra.get('poolMeta', '').lower() if hasattr(p, 'extra') and p.extra.get('poolMeta') else ''
                exposure = p.extra.get('exposure', '')
                il_risk = p.extra.get('ilRisk', '')
                tokens = p.extra.get('underlyingTokens', [])
                if (
                    exposure == 'single'
                    or any(x in pool_type for x in ['staking', 'lend', 'single'])
                    or il_risk == 'no'
                    or (tokens and len(tokens) == 1)
                ):
                    return 'Staking'
                return 'Pooling'
            def pool_row_dict(p):
                predictions = p.extra.get('predictions', {}) if p.extra.get('predictions') else {}
                pool_id = p.extra.get('pool')
                details_url = f"https://defillama.com/yields/pool/{pool_id}" if pool_id else ''
                category = classify_pool(p)
                return {
                    'category': category,
                    'chain': p.chain,
                    'project': p.protocol,
                    'symbol': p.pool_name,
                    'tvlUsd': p.extra.get('tvlUsd'),
                    'apyBase': p.extra.get('apyBase'),
                    'apyReward': p.extra.get('apyReward'),
                    'apy': p.apy,
                    'rewardTokens': ', '.join([str(t) for t in p.extra.get('rewardTokens', []) if t]) if p.extra.get('rewardTokens') else '',
                    'pool': pool_id,
                    'poolMeta': p.extra.get('poolMeta'),
                    'exposure': p.extra.get('exposure'),
                    'ilRisk': p.extra.get('ilRisk'),
                    'underlyingTokens': ', '.join([str(t) for t in p.extra.get('underlyingTokens', []) if t]) if p.extra.get('underlyingTokens') else '',
                    'stablecoin': p.extra.get('stablecoin'),
                    'apyBase7d': p.extra.get('apyBase7d'),
                    'apyMean30d': p.extra.get('apyMean30d'),
                    'volumeUsd1d': p.extra.get('volumeUsd1d'),
                    'volumeUsd7d': p.extra.get('volumeUsd7d'),
                    'predictedClass': predictions.get('predictedClass'),
                    'predictedProbability': predictions.get('predictedProbability'),
                    'binnedConfidence': predictions.get('binnedConfidence'),
                    'outlier': p.extra.get('outlier'),
                    'count': p.extra.get('count'),
                    'mu': p.extra.get('mu'),
                    'sigma': p.extra.get('sigma'),
                    'url': p.url if p.url else f"https://defillama.com/protocol/{p.protocol}",
                    'details_url': f"[details]({details_url})" if details_url else '',
                    'il7d': p.extra.get('il7d'),
                    'apyPct1D': p.extra.get('apyPct1D'),
                    'apyPct7D': p.extra.get('apyPct7D'),
                    'apyPct30D': p.extra.get('apyPct30D'),
                    'lockup': p.extra.get('lockup', p.duration) if hasattr(p, 'extra') else p.duration,
                }
            # Group by protocol
            import pandas as pd
            pool_dicts = [pool_row_dict(p) for p in pools]
            df = pd.DataFrame(pool_dicts)
            protocols = df['project'].unique().tolist()
            tabs = st.tabs([f"{p.title()}" for p in protocols])
            for i, protocol in enumerate(protocols):
                with tabs[i]:
                    protocol_pools = df[df['project'] == protocol]
                    protocol_slug = protocol
                    st.markdown("---")
                    render_protocol_card(protocol_slug)
                    st.markdown("---")
                    st.markdown("#### Pools for this Protocol")
                    # Show top 10 by TVL or APY
                    if 'tvlUsd' in protocol_pools.columns and protocol_pools['tvlUsd'].notnull().any():
                        protocol_pools = protocol_pools.sort_values(by='tvlUsd', ascending=False)
                    elif 'apy' in protocol_pools.columns:
                        protocol_pools = protocol_pools.sort_values(by='apy', ascending=False)
                    protocol_pools = protocol_pools.head(10).copy()
                    pool_options = [
                        f"{row['symbol']} | {row['category']} | TVL: {row['tvlUsd']}" for _, row in protocol_pools.iterrows()
                    ]
                    selected_idx = st.selectbox(f"Select a pool to view details (Top 10 for {protocol.title()})", options=list(range(len(pool_options))), format_func=lambda i: pool_options[i] if i < len(pool_options) else "", key=f"select_{protocol}")
                    selected_row = protocol_pools.iloc[selected_idx]
                    pool_id = selected_row['pool']
                    # Pool summary
                    st.markdown("#### Pool Summary")
                    st.dataframe(pd.DataFrame(selected_row).T, use_container_width=True)
                    # APY chart
                    st.markdown("#### APY History")
                    chart = cached_fetch_pool_chart(pool_id)
                    if chart and 'data' in chart and chart['data']:
                        chart_df = pd.DataFrame(chart['data'])
                        if 'apy' in chart_df.columns and 'timestamp' in chart_df.columns:
                            chart_df['date'] = pd.to_datetime(chart_df['timestamp'])
                            st.line_chart(chart_df.set_index('date')['apy'], use_container_width=True)
                        else:
                            st.info("No APY chart data available for this pool.")
                    else:
                        st.info("No APY chart data available for this pool.")
                    # Pool details
                    st.markdown("#### Pool Details (API)")
                    details = cached_fetch_pool_details(pool_id)
                    with st.expander("Show raw API details"):
                        if details and not details.get('error'):
                            st.json(details)
                        else:
                            st.warning("No additional API details available.")
                    # DeFiLlama link
                    st.markdown(f"[View on DeFiLlama](https://defillama.com/yields/pool/{pool_id})")
        else:
            st.warning(f"No pools found for token {token}.")

elif demo_type == "DeFiLlama Yield Rankings":
    st.header("📊 DeFiLlama Yield Rankings Dataset")
    st.markdown("""
        Explore DeFiLlama's comprehensive yield rankings dataset to find the best staking and yield opportunities across all protocols.
        
        **Features:**
        - Real-time yield rankings from DeFiLlama's datasets
        - Filter by token to find pools containing your assets
        - Grouped by protocol with detailed analytics
        - Cached data for fast performance
        - Direct links to DeFiLlama analytics
    """)
    
    st.info("💡 **Research Tip:** This dataset provides comprehensive yield rankings across all DeFi protocols. Use the protocol cards and pool details to compare opportunities.")
    
    # IL Risk explanation
    with st.expander("ℹ️ About IL Risk (Impermanent Loss)"):
        st.markdown("""
        **Impermanent Loss (IL)** occurs when the price of tokens in a liquidity pool changes relative to each other.
        
        - **No IL Risk**: Single-asset staking, lending, or stablecoin pairs (like USDC/DAI)
        - **Has IL Risk**: Most LP pools where token prices can diverge (like ETH/USDC)
        - **Unknown**: Risk level not specified in the data
        
        **For beginners**: Start with "No IL Risk" pools for safer yield farming.
        """)
    
    # Token input
    token = st.text_input("Enter token symbol to filter pools (e.g., USDC, ETH, DAI):", "USDC")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort pools by:",
        ["TVL (Total Value Locked)", "APY (Annual Percentage Yield)", "Protocol Name"],
        help="Choose how to sort the pools for comparison"
    )
    
    # IL risk filter
    il_risk_filter = st.selectbox(
        "Filter by IL risk:",
        ["All", "No IL Risk", "Has IL Risk", "Unknown IL Risk"],
        help="Choose how to filter the pools based on IL risk"
    )
    
    if st.button("🔍 Load Yield Rankings"):
        with st.spinner("Fetching DeFiLlama yield rankings dataset..."):
            # Fetch the rankings data
            rankings_df = fetch_yield_rankings()
            
            if rankings_df is not None and not rankings_df.empty:
                st.success(f"✅ Loaded {len(rankings_df)} total pools from DeFiLlama yield rankings!")
                
                # Filter by token and IL risk
                filtered_df = filter_rankings_by_token(rankings_df, token, il_risk_filter)
                
                if not filtered_df.empty:
                    st.success(f"Found {len(filtered_df)} pools containing {token.upper()}")
                    
                    # Sort the data
                    if sort_by == "TVL (Total Value Locked)":
                        filtered_df = filtered_df.sort_values('tvlUsd', ascending=False, na_position='last')
                    elif sort_by == "APY (Annual Percentage Yield)":
                        filtered_df = filtered_df.sort_values('apy', ascending=False, na_position='last')
                    elif sort_by == "Protocol Name":
                        filtered_df = filtered_df.sort_values('project')
                    
                    # Group by protocol
                    protocols = filtered_df['project'].unique().tolist()
                    
                    if len(protocols) > 1:
                        st.markdown(f"### 📈 Results for {token.upper()} ({len(filtered_df)} pools across {len(protocols)} protocols)")
                        
                        # Create tabs for each protocol
                        tabs = st.tabs([f"{p.title()}" for p in protocols])
                        
                        for i, protocol in enumerate(protocols):
                            with tabs[i]:
                                protocol_pools = filtered_df[filtered_df['project'] == protocol]
                                
                                st.markdown("---")
                                render_rankings_protocol_card(protocol)
                                st.markdown("---")
                                
                                st.markdown(f"#### 📊 Pools for {protocol.title()}")
                                
                                # Format data for display
                                display_df = format_rankings_data(protocol_pools)
                                
                                if not display_df.empty:
                                    # Show IL risk summary for this protocol
                                    il_risk_summary = protocol_pools['ilRisk'].value_counts()
                                    if not il_risk_summary.empty:
                                        st.markdown("**IL Risk Distribution:**")
                                        for risk_type, count in il_risk_summary.items():
                                            risk_label = "No IL Risk" if risk_type == 'no' else "Has IL Risk" if risk_type == 'yes' else "Unknown"
                                            st.markdown(f"- {risk_label}: {count} pools")
                                    
                                    # Show top pools
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Pool selection for details
                                    if len(display_df) > 1:
                                        pool_options = [
                                            f"{row['Pool']} | {row['Type']} | IL: {row['IL Risk']} | APY: {row['APY']} | TVL: {row['TVL']}" 
                                            for _, row in display_df.iterrows()
                                        ]
                                        
                                        selected_idx = st.selectbox(
                                            f"Select a pool to view details ({len(display_df)} pools available)",
                                            options=list(range(len(pool_options))),
                                            format_func=lambda i: pool_options[i] if i < len(pool_options) else "",
                                            key=f"rankings_select_{protocol}"
                                        )
                                        
                                        if selected_idx < len(display_df):
                                            selected_row = display_df.iloc[selected_idx]
                                            original_row = protocol_pools.iloc[selected_idx]
                                            
                                            st.markdown("#### 📋 Pool Details")
                                            
                                            # Display key metrics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("APY", selected_row['APY'])
                                            with col2:
                                                st.metric("TVL", selected_row['TVL'])
                                            with col3:
                                                st.metric("Type", selected_row['Type'])
                                            
                                            # Show additional details
                                            st.markdown("#### 📊 Additional Pool Information")
                                            details_data = [
                                                ['Chain', original_row.get('chain', 'N/A')],
                                                ['Protocol', original_row.get('project', 'N/A')],
                                                ['Pool ID', original_row.get('pool', 'N/A')],
                                                ['IL Risk', classify_il_risk(original_row)]
                                            ]
                                            
                                            # Add URL if available
                                            if 'url' in original_row and original_row.get('url'):
                                                details_data.append(['URL', f"[View on DeFiLlama]({original_row.get('url')})"])
                                            else:
                                                details_data.append(['URL', 'N/A'])
                                            
                                            details_df = pd.DataFrame(details_data, columns=['Metric', 'Value'])
                                            st.dataframe(details_df, use_container_width=True)
                                            
                                            # Try to fetch APY chart if pool ID is available
                                            pool_id = original_row.get('pool')
                                            if pool_id:
                                                st.markdown("#### 📈 APY History")
                                                chart = cached_fetch_pool_chart(pool_id)
                                                if chart and 'data' in chart and chart['data']:
                                                    chart_df = pd.DataFrame(chart['data'])
                                                    if 'apy' in chart_df.columns and 'timestamp' in chart_df.columns:
                                                        chart_df['date'] = pd.to_datetime(chart_df['timestamp'])
                                                        st.line_chart(chart_df.set_index('date')['apy'], use_container_width=True)
                                                    else:
                                                        st.info("No APY chart data available for this pool.")
                                                else:
                                                    st.info("No APY chart data available for this pool.")
                                            
                                            # Raw data expander
                                            with st.expander("🔍 Raw Pool Data"):
                                                st.json(original_row.to_dict())
                                    else:
                                        st.info(f"Only one pool found for {protocol.title()}")
                                        st.dataframe(display_df, use_container_width=True)
                                else:
                                    st.warning(f"No pools found for {protocol.title()}")
                    else:
                        st.info(f"All pools for {token.upper()} are from a single protocol: {protocols[0]}")
                        protocol_pools = filtered_df
                        display_df = format_rankings_data(protocol_pools)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Show summary statistics
                        st.markdown("#### 📊 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Pools", len(filtered_df))
                        with col2:
                            avg_apy = filtered_df['apy'].mean()
                            st.metric("Average APY", f"{avg_apy:.2f}%" if not pd.isna(avg_apy) else "N/A")
                        with col3:
                            total_tvl = filtered_df['tvlUsd'].sum()
                            st.metric("Total TVL", f"${total_tvl:,.0f}" if not pd.isna(total_tvl) else "N/A")
                        with col4:
                            # Count IL risk distribution
                            il_risk_counts = filtered_df['ilRisk'].value_counts()
                            no_il_pools = il_risk_counts.get('no', 0)
                            st.metric("No IL Risk Pools", no_il_pools)
                else:
                    st.warning(f"No pools found containing {token.upper()} in the yield rankings dataset.")
                    
                    # Show some sample data to help user
                    st.markdown("#### 💡 Sample Pools from Dataset")
                    sample_df = rankings_df.head(10)[['symbol', 'project', 'chain', 'apy', 'tvlUsd']]
                    sample_df['apy'] = sample_df['apy'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    sample_df['tvlUsd'] = sample_df['tvlUsd'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                    st.dataframe(sample_df, use_container_width=True)
                    st.caption("Try searching for a different token or explore the sample data above.")
            else:
                st.error("Failed to load yield rankings dataset. Please try again later.")
                st.info("The dataset might be temporarily unavailable or there might be a network issue.")