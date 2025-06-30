import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.email_config = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('EMAIL_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'recipient': os.getenv('RECIPIENT_EMAIL')
        }
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '1800'))
        self.min_priority_score = int(os.getenv('MIN_PRIORITY_SCORE', '2'))
        self.max_items_per_email = int(os.getenv('MAX_ITEMS_PER_EMAIL', '10'))
        self.youtube_max_results = int(os.getenv('YOUTUBE_MAX_RESULTS', '50'))
        self.youtube_region_code = os.getenv('YOUTUBE_REGION_CODE', 'US')
        self.youtube_category_id = os.getenv('YOUTUBE_CATEGORY_ID', '25')
        self.reddit_subreddits = os.getenv('REDDIT_SUBREDDITS', 'stocks,investing,SecurityAnalysis,ValueInvesting,StockMarket').split(',')
        self.reddit_limit = int(os.getenv('REDDIT_LIMIT', '25'))
        self.news_sources = os.getenv('NEWS_SOURCES', 'bloomberg,reuters,financial-times,the-wall-street-journal,cnbc')
        self.news_language = os.getenv('NEWS_LANGUAGE', 'en')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.log_file_path = os.getenv('LOG_FILE_PATH')
        self.subject_prefix = os.getenv('EMAIL_SUBJECT_PREFIX', 'Financial Alert')
        self.continuous_mode = os.getenv('CONTINUOUS_MODE', 'false').lower() == 'true'
        self.custom_symbols = [s.strip().upper() for s in os.getenv('CUSTOM_STOCK_SYMBOLS', '').split(',') if s.strip()]
        self.custom_keywords = [k.strip().lower() for k in os.getenv('CUSTOM_KEYWORDS', '').split(',') if k.strip()]
        self.INFLUENCER_CHANNEL_IDS = ['UCRvqjQPSeaWn-uEx-w0XOIg','UCVzs1JD3i6sPfG4kUzFPqhA']
        # to add more
        self.env_example_vars = {
            'YOUTUBE_API_KEY': self.youtube_api_key,
            'EMAIL_USERNAME': self.email_config['username'],
            'EMAIL_PASSWORD': self.email_config['password'],
            'RECIPIENT_EMAIL': self.email_config['recipient'],
            'REDDIT_CLIENT_ID': self.reddit_client_id,
            'REDDIT_CLIENT_SECRET': self.reddit_client_secret,
            'NEWS_API_KEY': self.news_api_key,
            'EMAIL_SMTP_SERVER': self.email_config['smtp_server'],
            'EMAIL_PORT': self.email_config['port'],
            'SCAN_INTERVAL': self.scan_interval,
            'MIN_PRIORITY_SCORE': self.min_priority_score,
            'MAX_ITEMS_PER_EMAIL': self.max_items_per_email,
            'YOUTUBE_MAX_RESULTS': self.youtube_max_results,
            'YOUTUBE_REGION_CODE': self.youtube_region_code,
            'YOUTUBE_CATEGORY_ID': self.youtube_category_id,
            'REDDIT_SUBREDDITS': self.reddit_subreddits,
            'REDDIT_LIMIT': self.reddit_limit,
            'NEWS_SOURCES': self.news_sources,
            'NEWS_LANGUAGE': self.news_language,
            'LOG_LEVEL': self.log_level,
            'LOG_FILE_PATH': self.log_file_path,
            'EMAIL_SUBJECT_PREFIX': self.subject_prefix,
            'CONTINUOUS_MODE': self.continuous_mode,
            'CUSTOM_STOCK_SYMBOLS': self.custom_symbols,
            'CUSTOM_KEYWORDS': self.custom_keywords,
            'INFLUENCER_CHANNEL_IDS': self.INFLUENCER_CHANNEL_IDS
        }

    def validate(self):
        required = [
            ('YOUTUBE_API_KEY', self.youtube_api_key),
            ('EMAIL_USERNAME', self.email_config['username']),
            ('EMAIL_PASSWORD', self.email_config['password']),
            ('RECIPIENT_EMAIL', self.email_config['recipient'])
        ]
        missing = [name for name, val in required if not val]
        return missing
