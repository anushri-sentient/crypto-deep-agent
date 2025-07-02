import os
import requests
import google.generativeai as genai
import praw
from urllib.parse import urlparse
from dotenv import load_dotenv

# ========== Load API KEYS ==========
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')

# ========== Configure Gemini ==========
genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
except Exception as e:
    raise RuntimeError(f"[ERROR] Could not load Gemini model: {e}")

# ========== Reddit Setup ==========
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent='crypto-classifier-agent'
)

# ========== Helper Functions ==========

def get_reddit_post_and_top_comments(url, comment_limit=5):
    try:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)
        content = f"Title: {submission.title}\n\nPost:\n{submission.selftext}\n\nTop Comments:\n"
        for top_comment in submission.comments[:comment_limit]:
            content += f"- {top_comment.body}\n"
        return content
    except Exception as e:
        print(f"[ERROR] Reddit fetch failed for {url}: {e}")
        return None

def summarize_with_gemini(text):
    prompt = "Summarize this crypto-related article:\n\n" + text[:4000]
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini summarization failed: {e}")
        return "Summary unavailable."

# ========== Exported Main Function ==========

def analyze_reddit_urls(url_list):
    results = []

    for url in url_list:
        print(f"[URL] {url}")
        if "reddit.com" not in url:
            print("   -> [Skipped] Not a Reddit URL.\n")
            continue

        content = get_reddit_post_and_top_comments(url)
        if not content:
            print("   -> [Skipped] Unable to extract content.\n")
            continue

        summary = summarize_with_gemini(content)

        result = {
            "url": url,
            "summary": summary if summary else "Not crypto-related"
        }

        results.append(result)


        print("   -> 📝 Summary:\n" + summary + "\n")

    return results
