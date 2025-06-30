from agent.config import Config
from agent.logger import get_logger
from agent.keywords import get_stock_symbols, get_financial_keywords
from agent.sources.youtube import get_trending_youtube_videos, get_influencer_videos
from agent.sources.reddit import search_reddit_finance
from agent.sources.news import search_financial_news
from agent.filter import filter_high_priority_content
from agent.emailer import send_email_alert
import time
from datetime import datetime

def run_scan(config, logger, stock_symbols, financial_keywords, processed_content):
    logger.info(f"Starting financial scan at {datetime.now()}")
    all_content = []
    logger.info("Fetching YouTube videos...")
    youtube_videos = get_trending_youtube_videos(config, logger, stock_symbols, financial_keywords)
    all_content.extend(youtube_videos)
    logger.info(f"Found {len(youtube_videos)} relevant YouTube videos")
    logger.info("Fetching Reddit posts...")
    reddit_posts = search_reddit_finance(config, logger, stock_symbols, financial_keywords)
    all_content.extend(reddit_posts)
    logger.info(f"Found {len(reddit_posts)} relevant Reddit posts")
    logger.info("Fetching financial news...")
    news_articles = search_financial_news(config, logger, stock_symbols, financial_keywords)
    all_content.extend(news_articles)
    logger.info(f"Found {len(news_articles)} relevant news articles")
    logger.info("Fetching influencer YouTube videos...")
    influencer_videos = get_influencer_videos(
        config.youtube_api_key,
        config.INFLUENCER_CHANNEL_IDS,
        max_results=5,
        logger=logger
    )
    logger.info(f"Found {len(influencer_videos)} influencer videos")
    # Only filter for new influencer videos, not by priority
    new_influencer_videos = []
    for item in influencer_videos:
        content_id = f"{item['source']}_{item['id']}"
        if content_id not in processed_content:
            new_influencer_videos.append(item)
        processed_content.add(content_id)
    high_priority = filter_high_priority_content(all_content, config.min_priority_score, logger, stock_symbols, financial_keywords)
    logger.info(f"Identified {len(high_priority)} high-priority items")
    new_content = []
    for item in high_priority:
        content_id = f"{item['source']}_{item['id']}"
        if content_id not in processed_content:
            new_content.append(item)
        processed_content.add(content_id)
    logger.info(f"Found {len(new_content)} new high-priority items")
    logger.info(f"Found {len(new_influencer_videos)} new influencer videos")
    if new_content or new_influencer_videos:
        send_email_alert(config, logger, new_content, new_influencer_videos)
    else:
        logger.info("No new high-priority content to report")
    return new_content, new_influencer_videos

def main():
    config = Config()
    logger = get_logger(__name__, config.log_level, config.log_file_path)
    missing = config.validate()
    if missing:
        logger.warning(f"Missing required configuration fields: {', '.join(missing)}")
        logger.warning("Some features may not work properly. Check your .env file.")
    stock_symbols = get_stock_symbols(config.custom_symbols)
    financial_keywords = get_financial_keywords(config.custom_keywords)
    logger.info(f"Monitoring {len(stock_symbols)} stock symbols: {', '.join(stock_symbols[:10])}{'...' if len(stock_symbols) > 10 else ''}")
    logger.info(f"Monitoring {len(financial_keywords)} financial keywords: {', '.join(financial_keywords[:10])}{'...' if len(financial_keywords) > 10 else ''}")
    processed_content = set()
    if config.continuous_mode:
        logger.info("Starting continuous monitoring...")
        while True:
            try:
                run_scan(config, logger, stock_symbols, financial_keywords, processed_content)
                logger.info(f"Waiting {config.scan_interval} seconds before next scan...")
                time.sleep(config.scan_interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)
    else:
        logger.info("Running single scan...")
        results, influencer_results = run_scan(config, logger, stock_symbols, financial_keywords, processed_content)
        logger.info(f"Scan completed. Found {len(results)} new items and {len(influencer_results)} new influencer videos.")

if __name__ == "__main__":
    main()