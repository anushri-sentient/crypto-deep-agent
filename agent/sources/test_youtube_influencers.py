import sys
import os
from agent.sources.youtube import get_influencer_videos

if __name__ == "__main__":
    print("YouTube Influencer Video Fetch Test")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        api_key = input("Enter your YouTube Data API key: ").strip()
    channel_ids = input("Enter comma-separated YouTube channel IDs: ").strip().split(",")
    channel_ids = [cid.strip() for cid in channel_ids if cid.strip()]
    if not channel_ids:
        print("No channel IDs provided. Exiting.")
        sys.exit(1)
    max_results = input("How many recent videos per channel? (default 5): ").strip()
    try:
        max_results = int(max_results) if max_results else 5
    except ValueError:
        max_results = 5
    print(f"Fetching up to {max_results} recent videos for each channel...")
    videos = get_influencer_videos(api_key, channel_ids, max_results)
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"- {v['title']} (Channel: {v['channelId']}, Published: {v['publishedAt']})")
        print(f"  URL: {v['url']}") 