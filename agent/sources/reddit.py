import requests
import datetime
from agent.filter import contains_financial_content

def search_newest_crypto_reddit(config, logger, stock_symbols, financial_keywords):
    if not config.reddit_client_id or not config.reddit_client_secret:
        logger.warning("Reddit credentials not provided")
        return []

    posts = []
    try:
        auth = requests.auth.HTTPBasicAuth(config.reddit_client_id, config.reddit_client_secret)
        data = {'grant_type': 'client_credentials'}
        headers = {'User-Agent': 'CryptoAgent/1.0'}
        token_response = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
        token_response.raise_for_status()
        token = token_response.json()['access_token']
        headers['Authorization'] = f'bearer {token}'

        # Example crypto-related subreddits — adjust config.reddit_subreddits accordingly if you want
        crypto_subreddits = ['CryptoCurrency', 'defi', 'ethfinance', 'bitcoin','CryptoMoonShots']

        for subreddit in crypto_subreddits:
            url = f'https://oauth.reddit.com/r/{subreddit}/new'  # 'new' sorting
            params = {'limit': 20}  # fetch more to filter later
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            for post in data['data']['children']:
                post_data = post['data']
                title_and_text = f"{post_data['title']} {post_data.get('selftext', '')}"

                # Filter for financial/crypto content
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

        # Sort all collected posts by newest
        posts_sorted = sorted(posts, key=lambda x: x['created_at'], reverse=True)

        # Return top 10 newest crypto-related posts
        return posts_sorted[:10]

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Reddit data: {e}")
        return []

def search_reddit_posts(config, logger, query, max_results=10):
    """
    Search Reddit for posts matching the query using Reddit's search API.
    Args:
        config: Config object with Reddit credentials.
        logger: Logger for debug/info output.
        query (str): The search query.
        max_results (int): Number of results to fetch.
    Returns:
        list: List of post dicts with id, title, author, created_at, score, num_comments, url, subreddit, and source.
    """
    if not config.reddit_client_id or not config.reddit_client_secret:
        logger.warning("Reddit credentials not provided")
        return []
    posts = []
    try:
        auth = requests.auth.HTTPBasicAuth(config.reddit_client_id, config.reddit_client_secret)
        data = {'grant_type': 'client_credentials'}
        headers = {'User-Agent': 'CryptoAgent/1.0'}
        token_response = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
        token_response.raise_for_status()
        token = token_response.json()['access_token']
        headers['Authorization'] = f'bearer {token}'
        # Search in a set of crypto-related subreddits
        crypto_subreddits = ['CryptoCurrency', 'defi', 'ethfinance', 'bitcoin','CryptoMoonShots']
        for subreddit in crypto_subreddits:
            url = f'https://oauth.reddit.com/r/{subreddit}/search'
            params = {'q': query, 'restrict_sr': 1, 'sort': 'new', 'limit': max_results}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            for post in data['data']['children']:
                post_data = post['data']
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
        # Sort all collected posts by newest
        posts_sorted = sorted(posts, key=lambda x: x['created_at'], reverse=True)
        return posts_sorted[:max_results]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching Reddit: {e}")
        return []
