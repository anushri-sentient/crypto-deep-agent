import os
import logging
import requests
import datetime
import time
import re
import subprocess

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Your API keys (set as environment variables for security)
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Gemini LLM import (optional, only if you have google-generativeai installed)
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("google-generativeai not installed. Gemini LLM analysis will be skipped.")

# --- Settings ---
MAX_VIDEOS = 1  # Increased to show more options
LANG = 'en'

# Curated YouTuber list with channel IDs
CURATED_YOUTUBERS = {
    "Stephen TCG | DeFi Dojo": {
        "channel_id": None,  # Will be searched by name
        "url": "https://www.youtube.com/@TheCalculatorGuy",
        "search_terms": ["TheCalculatorGuy", "Stephen TCG", "DeFi Dojo"]
    },
    "CryptoLabs Research | Defi Passive Income | Crypto": {
        "channel_id": None,  # Will be searched by name
        "url": "https://www.youtube.com/@CryptolabsResearch",
        "search_terms": ["CryptolabsResearch", "CryptoLabs Research"]
    },
    "Jake Call | DeFi Income": {
        "channel_id": None,  # Will be searched by name
        "url": "https://www.youtube.com/@jakeacall.",
        "search_terms": ["jakeacall", "Jake Call"]
    },
    "TokenGuy": {
        "channel_id": None,  # Will be searched by name
        "url": "https://www.youtube.com/@TokenGuySol",
        "search_terms": ["TokenGuySol", "TokenGuy"]
    }
}

# Helper to get ISO date string N months ago
def get_recent_date_iso(months=3):
    today = datetime.datetime.utcnow()
    delta = datetime.timedelta(days=30*months)
    return (today - delta).isoformat("T") + "Z"

# Get channel ID from URL
def get_channel_id_from_url(url):
    """Extract channel ID from YouTube URL"""
    try:
        # For @username format, we need to make an API call to get channel ID
        if '@' in url:
            username = url.split('@')[1].split('/')[0]
            # This would require an API call to convert username to channel ID
            # For now, return None and we'll handle it in the main function
            return None
        return None
    except Exception as e:
        logging.warning(f"Could not extract channel ID from {url}: {e}")
        return None

# Fetch videos from specific channels
def fetch_channel_videos(youtube, channel_id, max_results=10):
    """Fetch recent videos from a specific channel"""
    try:
        # Get channel's uploads playlist
        channels_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channels_response.get('items'):
            return []
            
        uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get videos from uploads playlist
        playlist_response = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part='snippet',
            maxResults=max_results
        ).execute()
        
        videos = []
        for item in playlist_response.get('items', []):
            video_id = item['snippet']['resourceId']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            published_at = item['snippet']['publishedAt']
            thumbnail_url = item['snippet']['thumbnails']['medium']['url']
            
            videos.append({
                'video_id': video_id,
                'title': title,
                'description': description,
                'published_at': published_at,
                'thumbnail_url': thumbnail_url,
                'url': f"https://youtube.com/watch?v={video_id}"
            })
        
        return videos
    except Exception as e:
        logging.error(f"Error fetching videos from channel {channel_id}: {e}")
        return []

# Display video options with thumbnails
def display_video_options(videos):
    """Display videos with thumbnails and return user selection"""
    print("\n" + "="*80)
    print("AVAILABLE VIDEOS FOR ANALYSIS")
    print("="*80)
    
    for i, video in enumerate(videos, 1):
        print(f"\n{i}. {video['title']}")
        print(f"   Published: {video['published_at'][:10]}")
        print(f"   URL: {video['url']}")
        print(f"   Thumbnail: {video['thumbnail_url']}")
        print(f"   Description: {video['description'][:100]}...")
        print("-" * 80)
    
    print(f"\nEnter video numbers to analyze (comma-separated, e.g., 1,3,5) or 'all' for all videos:")
    selection = input().strip()
    
    if selection.lower() == 'all':
        return list(range(len(videos)))
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        return [i for i in selected_indices if 0 <= i < len(videos)]
    except ValueError:
        print("Invalid selection. Please enter numbers separated by commas.")
        return []

# Clean subtitle text from VTT format (remove timestamps and metadata)
def clean_subtitle_text(sub_text):
    lines = sub_text.splitlines()
    filtered = []
    for line in lines:
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} -->", line):
            continue
        if line.strip() == "" or line.startswith("WEBVTT"):
            continue
        filtered.append(line.strip())
    return " ".join(filtered)

# Fallback: get transcript with youtube-transcript-api
def get_transcript_youtube_api(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[LANG])
        return " ".join([t['text'] for t in transcript_list])
    except TranscriptsDisabled:
        logging.warning(f"Transcripts disabled for video {video_id}")
    except Exception as e:
        logging.warning(f"Error fetching transcript via YouTube API for {video_id}: {e}")
    return ''

# Gemini LLM extraction
def extract_deFi_insights_with_gemini(transcript, api_key, model_name="models/gemini-1.5-flash"):
    if not genai:
        logging.error("Gemini LLM library not installed.")
        return "Gemini LLM not available."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = (
        "You're analyzing a transcript from a DeFi-focused video. Extract useful insights about stablecoin yields, DeFi protocols, and yield strategies.\n\n"
        "1. Coins mentioned\n"
        "2. Protocols/platforms\n"
        "3. Yield strategies summarized\n"
        "4. Watchlist suggestions\n\n"
        f"Transcript:\n\n{transcript}"
    )
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)

# --- AssemblyAI transcript extraction ---
def get_transcript_assemblyai(video_url, api_key, lang='en'):
    # Download audio
    filename = f"{video_url.split('=')[-1]}.mp3"
    command = ['yt-dlp', '-x', '--audio-format', 'mp3', '-o', filename, video_url]
    subprocess.run(command, check=True)
    transcript = ''
    try:
        # Upload to AssemblyAI
        headers = {'authorization': api_key}
        with open(filename, 'rb') as f:
            response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, files={'file': f})
        upload_url = response.json().get('upload_url')
        if not upload_url:
            return ''
        # Request transcription
        endpoint = 'https://api.assemblyai.com/v2/transcript'
        json_data = {
            "audio_url": upload_url,
            "language_code": lang
        }
        headers = {'authorization': api_key, 'content-type': 'application/json'}
        response = requests.post(endpoint, json=json_data, headers=headers)
        resp_json = response.json()
        transcript_id = resp_json.get('id')
        if not transcript_id:
            return ''
        # Poll until complete
        endpoint = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
        headers = {'authorization': api_key}
        for _ in range(60):  # up to 5 minutes
            poll_resp = requests.get(endpoint, headers=headers).json()
            if poll_resp['status'] == 'completed':
                transcript = poll_resp['text']
                break
            elif poll_resp['status'] == 'error':
                return ''
            time.sleep(5)
        return transcript
    finally:
        # Always try to delete the mp3 file
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            logging.warning(f"Could not delete temporary mp3 file {filename}: {e}")

# Main runner with channel selection
def run_scraper_with_channels():
    from googleapiclient.discovery import build

    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    print("\n" + "="*60)
    print("CURATED DEFI YOUTUBERS")
    print("="*60)
    
    for i, (name, info) in enumerate(CURATED_YOUTUBERS.items(), 1):
        print(f"{i}. {name}")
        print(f"   URL: {info['url']}")
    
    print(f"\nEnter YouTuber numbers to search (comma-separated, e.g., 1,3) or 'all' for all:")
    youtuber_selection = input().strip()
    
    if youtuber_selection.lower() == 'all':
        selected_youtubers = list(CURATED_YOUTUBERS.items())
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in youtuber_selection.split(',')]
            selected_youtubers = [list(CURATED_YOUTUBERS.items())[i] for i in selected_indices if 0 <= i < len(CURATED_YOUTUBERS)]
        except (ValueError, IndexError):
            print("Invalid selection. Using all YouTubers.")
            selected_youtubers = list(CURATED_YOUTUBERS.items())
    
    all_videos = []
    
    # Search for videos from these channels using their names and search terms
    for name, info in selected_youtubers:
        print(f"\nSearching for videos from: {name}")
        
        # Use search terms to find videos from this channel
        search_terms = info.get('search_terms', [name])
        for term in search_terms:
            search_response = youtube.search().list(
                q=f"{term} stablecoin yield defi",
                part='snippet',
                type='video',
                maxResults=MAX_VIDEOS,
                order='date',
                publishedAfter=get_recent_date_iso(3)
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
    
    if not all_videos:
        print("No videos found from selected channels.")
        return [], []
    
    # Remove duplicates based on video_id
    unique_videos = []
    seen_ids = set()
    for video in all_videos:
        if video['video_id'] not in seen_ids:
            unique_videos.append(video)
            seen_ids.add(video['video_id'])
    
    # Display options and get user selection
    selected_indices = display_video_options(unique_videos)
    selected_videos = [unique_videos[i] for i in selected_indices if 0 <= i < len(unique_videos)]
    
    if not selected_videos:
        print("No videos selected for analysis.")
        return [], []
    
    results = []
    analyses = []

    for video in selected_videos:
        video_id = video['video_id']
        title = video['title']
        url = video['url']
        print(f"\nGetting transcript and analyzing: {title}")

        transcript = get_transcript_assemblyai(url, os.getenv('ASSEMBLYAI_API_KEY'), lang=LANG)

        if not transcript:
            print(f"No transcript available for video: {title}")
            analyses.append({'title': title, 'url': url, 'analysis': 'No transcript available.'})
            continue

        # Gemini analysis (if API key set)
        if GEMINI_API_KEY:
            analysis = extract_deFi_insights_with_gemini(transcript, GEMINI_API_KEY)
        else:
            analysis = "No Gemini API key set; skipping analysis."

        print(f"Gemini Analysis for {title}:\n{analysis}")
        analyses.append({'title': title, 'url': url, 'analysis': analysis})

        results.append({'title': title, 'channel': video['channel'], 'url': url})

    return results, analyses

# Original search-based runner (kept for compatibility)
def run_scraper():
    from googleapiclient.discovery import build

    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    published_after = get_recent_date_iso(3)
    query = 'best stable-coin yields'

    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=MAX_VIDEOS,
        order='date',
        publishedAfter=published_after
    ).execute()

    videos = search_response.get('items', [])
    results = []
    analyses = []

    for v in videos:
        video_id = v['id']['videoId']
        title = v['snippet']['title']
        channel = v['snippet']['channelTitle']
        url = f"https://youtube.com/watch?v={video_id}"
        print(f"\nGetting transcript and analyzing: {title}")

        transcript = get_transcript_assemblyai(url, os.getenv('ASSEMBLYAI_API_KEY'), lang=LANG)

        if not transcript:
            print(f"No transcript available for video: {title}")
            analyses.append({'title': title, 'url': url, 'analysis': 'No transcript available.'})
            continue

        # Gemini analysis (if API key set)
        if GEMINI_API_KEY:
            analysis = extract_deFi_insights_with_gemini(transcript, GEMINI_API_KEY)
        else:
            analysis = "No Gemini API key set; skipping analysis."

        print(f"Gemini Analysis for {title}:\n{analysis}")
        analyses.append({'title': title, 'url': url, 'analysis': analysis})

        results.append({'title': title, 'channel': channel, 'url': url})

    return results, analyses


def main():
    if not YOUTUBE_API_KEY:
        logging.error("Missing YouTube API key. Please set YOUTUBE_API_KEY environment variable.")
        return
    if not GEMINI_API_KEY:
        logging.warning("No Gemini API key found. Analysis steps will be skipped.")

    print("Choose scraping method:")
    print("1. Search-based (original)")
    print("2. Curated YouTubers (new)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        results, analyses = run_scraper_with_channels()
    else:
        results, analyses = run_scraper()

    # Save results & analyses to file
    with open('video_results.txt', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"{r['title']} | {r['channel']} | {r['url']}\n")

    with open('video_analyses.txt', 'w', encoding='utf-8') as f:
        for a in analyses:
            f.write(f"Video: {a['title']}\nURL: {a['url']}\nAnalysis:\n{a['analysis']}\n\n{'-'*60}\n\n")

def run_youtube_scraper():
    """Wrapper for Streamlit app compatibility. Also validates pools from the watchlist."""
    from validate_watchlist_llama import run_watchlist_validation
    results, analyses = run_scraper()
    _, valid_pools_structured = run_watchlist_validation()
    return results, analyses, valid_pools_structured

if __name__ == "__main__":
    main()
