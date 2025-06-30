import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from agent.filter import contains_financial_content

def fetch_rss_feeds(logger, stock_symbols, financial_keywords):
    rss_feeds = [
        'https://feeds.bloomberg.com/markets/news.rss',
        'https://moxie.foxbusiness.com/google-publisher/markets.xml',
        'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'https://feeds.reuters.com/reuters/businessNews'
    ]
    articles = []
    for feed_url in rss_feeds:
        try:
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            for item in items[:10]:
                title_elem = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                link_elem = item.find('link') or item.find('{http://www.w3.org/2005/Atom}link')
                desc_elem = item.find('description') or item.find('{http://www.w3.org/2005/Atom}summary')
                date_elem = item.find('pubDate') or item.find('{http://www.w3.org/2005/Atom}published')
                if title_elem is not None:
                    title = title_elem.text or ''
                    description = desc_elem.text if desc_elem is not None else ''
                    if contains_financial_content(title + ' ' + description, stock_symbols, financial_keywords, logger):
                        articles.append({
                            'id': link_elem.text.split('/')[-1][:20] if link_elem is not None else f"rss_{len(articles)}",
                            'title': title,
                            'description': description[:200] + '...' if len(description) > 200 else description,
                            'published_at': date_elem.text if date_elem is not None else datetime.now().isoformat(),
                            'url': link_elem.text if link_elem is not None else '',
                            'source_name': feed_url.split('/')[2],
                            'source': 'RSS'
                        })
        except Exception as e:
            logger.warning(f"Error fetching RSS feed {feed_url}: {e}")
            continue
    return articles
