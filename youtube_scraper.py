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
MAX_VIDEOS = 1
LANG = 'en'

# Helper to get ISO date string N months ago
def get_recent_date_iso(months=3):
    today = datetime.datetime.utcnow()
    delta = datetime.timedelta(days=30*months)
    return (today - delta).isoformat("T") + "Z"

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

# Main runner
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
