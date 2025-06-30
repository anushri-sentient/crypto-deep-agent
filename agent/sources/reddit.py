import requests
import datetime
from agent.filter import contains_financial_content

def search_reddit_finance(config, logger, stock_symbols, financial_keywords):
    if not config.reddit_client_id or not config.reddit_client_secret:
        logger.warning("Reddit credentials not provided")
        return []
    posts = []
    try:
        auth = requests.auth.HTTPBasicAuth(config.reddit_client_id, config.reddit_client_secret)
        data = {'grant_type': 'client_credentials'}
        headers = {'User-Agent': 'FinancialAgent/1.0'}
        token_response = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
        token_response.raise_for_status()
        token = token_response.json()['access_token']
        headers['Authorization'] = f'bearer {token}'
        for subreddit in config.reddit_subreddits:
            subreddit = subreddit.strip()
            if not subreddit:
                continue
            url = f'https://oauth.reddit.com/r/{subreddit}/hot'
            params = {'limit': config.reddit_limit}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            for post in data['data']['children']:
                post_data = post['data']
                title_and_text = f"{post_data['title']} {post_data.get('selftext', '')}"
                if contains_financial_content(title_and_text, stock_symbols, financial_keywords, logger):
                    posts.append({
                        'id': post_data['id'],
                        'title': post_data['title'],
                        'author': post_data['author'],
                        'created_at': datetime.datetime.fromtimestamp(post_data['created_utc']).isoformat(),
                        'score': post_data['score'],
                        'num_comments': post_data['num_comments'],
                        'url': f"https://reddit.com{post_data['permalink']}",
                        'subreddit': post_data['subreddit'],
                        'source': 'Reddit'
                    })
        return posts
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Reddit data: {e}")
        return []
