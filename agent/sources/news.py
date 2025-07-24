import requests
from agent.filter import contains_financial_content
from agent.sources.rss import fetch_rss_feeds

def search_financial_news(config, logger, stock_symbols, financial_keywords):
    news_articles = []
    if config.news_api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': ' OR '.join(stock_symbols[:10]) + ' OR ' + ' OR '.join(['stocks', 'market', 'earnings']),
                'sources': config.news_sources,
                'language': config.news_language,
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': config.news_api_key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            for article in data.get('articles', []):
                if contains_financial_content(article['title'] + ' ' + (article['description'] or ''), stock_symbols, financial_keywords, logger):
                    news_articles.append({
                        'id': article['url'].split('/')[-1][:20],
                        'title': article['title'],
                        'description': article['description'],
                        'author': article.get('author', 'Unknown'),
                        'published_at': article['publishedAt'],
                        'url': article['url'],
                        'source_name': article['source']['name'],
                        'source': 'News'
                    })
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {e}")
    if not news_articles and not config.news_api_key:
        news_articles = fetch_rss_feeds(logger, stock_symbols, financial_keywords)
    return news_articles

def search_news_articles(config, logger, query, max_results=10):
    """
    Search news articles using the News API with a query.
    Args:
        config: Config object with News API key and sources.
        logger: Logger for debug/info output.
        query (str): The search query.
        max_results (int): Number of results to fetch.
    Returns:
        list: List of article dicts with id, title, description, author, published_at, url, source_name, and source.
    """
    news_articles = []
    if config.news_api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'sources': config.news_sources,
                'language': config.news_language,
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'apiKey': config.news_api_key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            for article in data.get('articles', []):
                news_articles.append({
                    'id': article['url'].split('/')[-1][:20],
                    'title': article['title'],
                    'description': article['description'],
                    'author': article.get('author', 'Unknown'),
                    'published_at': article['publishedAt'],
                    'url': article['url'],
                    'source_name': article['source']['name'],
                    'source': 'News'
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching news: {e}")
    return news_articles[:max_results]
