import logging
import requests
import datetime
import os
import csv
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- YouTube API Setup ---
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
print(YOUTUBE_API_KEY)
if not YOUTUBE_API_KEY:
    logging.warning("YouTube API key not set. Set the YOUTUBE_API_KEY environment variable.")
else:
    logging.info("YouTube API key set.")
    print(YOUTUBE_API_KEY)

YOUTUBE_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'

# --- Search Parameters ---
QUERIES = [
    'best stable-coin yields',
    'top DeFi yields',
    'best stablecoin APY',
    'top DeFi APY',
]
CHAINS = ['Ethereum', 'Arbitrum', 'Optimism', 'Base', 'Polygon', 'Solana']
REPUTABLE_CHANNELS = {'DeFi Dojo', 'The Defiant', 'Bankless', 'Finematics'}
MAX_RESULTS = 10
MONTHS_LOOKBACK = 3

# --- LLM Extraction ---
try:
    import google.generativeai as genai
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    genai = None
    YouTubeTranscriptApi = None
    logging.warning("google-generativeai or youtube_transcript_api not installed. Gemini LLM extraction will not work.")

# --- Helper Functions ---
def get_recent_date_iso(months=3):
    today = datetime.datetime.utcnow()
    delta = datetime.timedelta(days=30*months)
    print(today - delta)
    return (today - delta).isoformat("T") + "Z"

def search_youtube(query, published_after):
    if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY_HERE':
        logging.warning("No YouTube API key set. Please set YOUTUBE_API_KEY as an environment variable.")
        return []
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': MAX_RESULTS,
        'order': 'date',
        'publishedAfter': published_after,
        'key': YOUTUBE_API_KEY
    }
    try:
        resp = requests.get(YOUTUBE_SEARCH_URL, params=params)
        resp.raise_for_status()
        return resp.json().get('items', [])
    except Exception as e:
        logging.error(f"YouTube API error for query '{query}': {e}")
        return []

def get_transcript(video_id, max_chars=1000, max_retries=3):
    if not YouTubeTranscriptApi:
        return ''
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = ' '.join([t['text'] for t in transcript])
            return full_text[:max_chars]
        except Exception as e:
            if '429' in str(e) or 'Too Many Requests' in str(e):
                logging.warning(f"429 Too Many Requests for {video_id}, attempt {attempt}/{max_retries}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logging.warning(f"No transcript for {video_id}: {e}")
                break
    logging.error(f"Failed to retrieve transcript for {video_id} after {max_retries} attempts.")
    return ''

def list_gemini_models(api_key):
    if not genai:
        print("google-generativeai not installed.")
        return []
    genai.configure(api_key=api_key)
    models = list(genai.list_models())
    for m in models:
        print(f"Model: {m.name}, Supported methods: {m.supported_generation_methods}")
    return models

def get_gemini_generation_model(api_key):
    models = list_gemini_models(api_key)
    # Prefer the latest recommended models
    preferred = [
        'models/gemini-1.5-flash',
        'models/gemini-1.5-flash-latest',
        'models/gemini-1.5-pro-latest',
        'models/gemini-1.5-pro',
    ]
    for p in preferred:
        for m in models:
            if m.name == p and 'generateContent' in m.supported_generation_methods:
                print(f"Using preferred Gemini model: {m.name}")
                return m.name
    # Otherwise, pick the first non-deprecated model with generateContent and not vision
    for m in models:
        if 'generateContent' in m.supported_generation_methods and 'vision' not in m.name.lower():
            print(f"Using Gemini model: {m.name}")
            return m.name
    print("No suitable Gemini model found. Please check your API access.")
    return None

def extract_deFi_insights_with_gemini(transcript, api_key, model_name="models/gemini-1.5-flash"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = (
        "You're analyzing a transcript from a DeFi-focused video. Your task is to extract useful insights related to stablecoin yields, DeFi protocols, and yield strategies.\n\n"
        "Please extract and present the following:\n"
        "1. **Coins**: List all cryptocurrency coins or tokens mentioned (e.g., USDC, ETH, SOL).\n"
        "2. **Protocols**: List any DeFi protocols or platforms referenced (e.g., Aave, Kamino, Marginfi).\n"
        "3. **Strategies**: Summarize any interesting or novel stablecoin yield strategies discussed.\n"
        "4. **Watchlist Additions**: Suggest any protocols, platforms, or strategies that should be considered for inclusion in a stablecoin yield farming watchlist.\n\n"
        "Format your response clearly under each heading.\n\n"
        f"Transcript:\n\n{transcript}"
    )
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)

def run_youtube_scraper():
    published_after = get_recent_date_iso(MONTHS_LOOKBACK)
    found_protocols = set()
    results = []
    analysis_outputs = []
    logging.info("Searching YouTube for recent DeFi yield videos...")
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model_name = None
    video_count = 0
    MAX_VIDEOS = 5
    for base_query in QUERIES:
        for chain in CHAINS:
            if video_count >= MAX_VIDEOS:
                break
            query = f"{base_query} {chain}"
            videos = search_youtube(query, published_after)
            for v in videos:
                if video_count >= MAX_VIDEOS:
                    break
                snippet = v['snippet']
                title = snippet['title']
                channel = snippet['channelTitle']
                published = snippet['publishedAt']
                video_id = v['id']['videoId']
                url = f"https://youtube.com/watch?v={video_id}"
                is_reputable = channel in REPUTABLE_CHANNELS
                protocols = [word for word in title.split() if word[0].isupper() and len(word) > 2]
                for p in protocols:
                    found_protocols.add(p)
                results.append({
                    'title': title,
                    'channel': channel,
                    'published': published,
                    'url': url,
                    'reputable': is_reputable,
                    'protocols': ', '.join(protocols)
                })
                # --- Gemini LLM DeFi analysis for each video ---
                print(f"\nFetching transcript and analyzing with Gemini LLM for video: {title} ({url})")
                transcript = get_transcript(video_id, max_chars=2000)
                if transcript:
                    analysis = extract_deFi_insights_with_gemini(transcript, gemini_api_key, gemini_model_name)
                    print(f"\n"); print(f"Gemini Analysis Output for {title}:\n{analysis}\n")
                    analysis_outputs.append({
                        'title': title,
                        'url': url,
                        'analysis': analysis
                    })
                else:
                    print(f"No transcript available for this video: {title}")
                video_count += 1
            if video_count >= MAX_VIDEOS:
                break
        if video_count >= MAX_VIDEOS:
            break
    # Save results to text file
    with open('youtube_yield_candidates.txt', 'w', encoding='utf-8') as f:
        for r in results:
            rep = 'REPUTABLE' if r['reputable'] else 'other'
            f.write(f"[{rep}] {r['title']} | {r['channel']} | {r['published']} | {r['url']} | Protocols: {r['protocols']}\n")
    # Save analysis outputs to a file
    with open('youtube_gemini_analysis.txt', 'w', encoding='utf-8') as f:
        for a in analysis_outputs:
            f.write(f"Video: {a['title']}\nURL: {a['url']}\nAnalysis:\n{a['analysis']}\n\n{'-'*60}\n\n")
    # Summarize all outputs using Gemini
    if analysis_outputs:
        print("doing summary")
        all_analyses = '\n\n'.join([a['analysis'] for a in analysis_outputs])
        summary_prompt = (
            "You are an expert DeFi analyst. Given the following analyses of several DeFi YouTube videos, "
            "summarize the most important coins, protocols, strategies, and watchlist suggestions. "
            "Highlight any recurring themes or especially promising opportunities.\n\n"
            f"Analyses:\n\n{all_analyses}"
        )
        summary = extract_deFi_insights_with_gemini(summary_prompt, gemini_api_key, gemini_model_name)
        print("\n================ SUMMARY OF ALL VIDEOS ================\n")
        print(summary)
        with open('youtube_gemini_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
    else:
        print("No analysis outputs to summarize.")
    return results, analysis_outputs

def main():
    run_youtube_scraper()

if __name__ == "__main__":
    main() 