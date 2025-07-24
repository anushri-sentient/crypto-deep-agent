import requests
from agent.filter import contains_financial_content

def get_trending_youtube_videos(config, logger, stock_symbols, financial_keywords):
    if not config.youtube_api_key:
        logger.warning("YouTube API key not provided")
        return []
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'snippet,statistics',
        'chart': 'mostPopular',
        'regionCode': config.youtube_region_code,
        'videoCategoryId': config.youtube_category_id,
        'maxResults': config.youtube_max_results,
        'key': config.youtube_api_key
    }
    logger.info(f"YouTube API request params: {params}")
    try:
        response = requests.get(url, params=params)
        logger.info(f"YouTube API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"YouTube API error response: {response.text}")
            return []
        data = response.json()
        logger.info(f"YouTube API returned {len(data.get('items', []))} total videos")
        sample_titles = [video['snippet']['title'] for video in data.get('items', [])[:5]]
        logger.info(f"Sample video titles: {sample_titles}")
        financial_videos = []
        for video in data.get('items', []):
            title = video['snippet']['title'].lower()
            description = video['snippet']['description'].lower()
            logger.debug(f"Checking video: {video['snippet']['title']}")
            if contains_financial_content(title + ' ' + description, stock_symbols, financial_keywords, logger):
                financial_videos.append({
                    'id': video['id'],
                    'title': video['snippet']['title'],
                    'channel': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'view_count': video['statistics'].get('viewCount', 0),
                    'url': f"https://www.youtube.com/watch?v={video['id']}",
                    'source': 'YouTube'
                })
                logger.debug(f"Added financial video: {video['snippet']['title']}")
            else:
                logger.debug(f"Video not financial: {video['snippet']['title']}")
        logger.info(f"Found {len(financial_videos)} videos with financial content")
        return financial_videos
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching YouTube data: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in YouTube API call: {e}")
        return []

def get_influencer_videos(api_key, channel_ids, max_results=5, logger=None):
    """
    Fetch recent videos from a list of YouTube influencer channel IDs.
    Args:
        api_key (str): YouTube Data API key.
        channel_ids (list): List of YouTube channel IDs.
        max_results (int): Number of recent videos to fetch per channel.
        logger (optional): Logger for debug/info output.
    Returns:
        list: List of video dicts with id, title, description, publishedAt, channelId, and url.
    """
    all_videos = []
    for channel_id in channel_ids:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'key': api_key,
            'channelId': channel_id,
            'part': 'snippet',
            'order': 'date',
            'maxResults': max_results,
            'type': 'video'
        }
        if logger:
            logger.info(f"Fetching videos for channel: {channel_id}")
        try:
            response = requests.get(url, params=params)
            if logger:
                logger.info(f"YouTube API response status: {response.status_code}")
            if response.status_code != 200:
                if logger:
                    logger.error(f"YouTube API error response: {response.text}")
                continue
            data = response.json()
            for item in data.get("items", []):
                video = {
                    "id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "channelId": channel_id,
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "source": "YouTubeInfluencer"
                }
                all_videos.append(video)
        except Exception as e:
            if logger:
                logger.error(f"Error fetching videos for channel {channel_id}: {e}")
            continue
    return all_videos

def search_youtube_videos(config, logger, query, max_results=10):
    """
    Search YouTube for videos matching the query.
    Args:
        config: Config object with API key and region.
        logger: Logger for debug/info output.
        query (str): The search query.
        max_results (int): Number of results to fetch.
    Returns:
        list: List of video dicts with id, title, description, publishedAt, channel, url, and source.
    """
    if not config.youtube_api_key:
        logger.warning("YouTube API key not provided")
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'regionCode': getattr(config, 'youtube_region_code', 'US'),
        'maxResults': max_results,
        'key': config.youtube_api_key
    }
    logger.info(f"YouTube search params: {params}")
    try:
        response = requests.get(url, params=params)
        logger.info(f"YouTube API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"YouTube API error response: {response.text}")
            return []
        data = response.json()
        videos = []
        for item in data.get("items", []):
            video = {
                "id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"].get("description", ""),
                "publishedAt": item["snippet"].get("publishedAt"),
                "channel": item["snippet"].get("channelTitle"),
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "source": "YouTube"
            }
            videos.append(video)
        logger.info(f"YouTube search returned {len(videos)} videos")
        return videos
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        return []
