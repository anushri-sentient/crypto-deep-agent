import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_email_alert(config, logger, content, influencer_videos=None):
    if not content and not influencer_videos:
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = config.email_config['username']
        msg['To'] = config.email_config['recipient']
        total_items = (len(content) if content else 0) + (len(influencer_videos) if influencer_videos else 0)
        msg['Subject'] = f"{config.subject_prefix} - {total_items} High Priority Items"
        body = f"""
Financial Agent Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        if content:
            body += f"\nFound {len(content)} high-priority financial content items:\n\n"
            for i, item in enumerate(content[:config.max_items_per_email], 1):
                if item['source'] == 'YouTube':
                    body += f"""
{i}. YouTube Video (Priority: {item['priority_score']})
   Title: {item['title']}
   Channel: {item.get('channel', item.get('channelId', 'N/A'))}
   Views: {item.get('view_count', 'N/A')}
   URL: {item['url']}
   
"""
                elif item['source'] == 'Reddit':
                    body += f"""
{i}. Reddit Post (Priority: {item['priority_score']})
   Title: {item['title']}
   Subreddit: r/{item['subreddit']}
   Author: u/{item['author']}
   Score: {item.get('score', 0)} | Comments: {item.get('num_comments', 0)}
   URL: {item['url']}
   
"""
                elif item['source'] in ['News', 'RSS']:
                    body += f"""
{i}. News Article (Priority: {item['priority_score']})
   Title: {item['title']}
   Source: {item.get('source_name', 'Unknown')}
   Description: {item.get('description', 'N/A')[:150]}{'...' if len(item.get('description', '')) > 150 else ''}
   URL: {item['url']}
   
"""
        if influencer_videos:
            body += f"\nNew videos from influencers ({len(influencer_videos)}):\n\n"
            for i, item in enumerate(influencer_videos[:config.max_items_per_email], 1):
                body += f"""
{i}. 📺 Influencer Video
   Title: {item['title']}
   Channel ID: {item.get('channelId', 'N/A')}
   Published: {item.get('publishedAt', 'N/A')}
   URL: {item['url']}
   
"""
        body += "\n---\nFinancial Agent MVP v1.0"
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(config.email_config['smtp_server'], config.email_config['port'])
        server.starttls()
        server.login(config.email_config['username'], config.email_config['password'])
        server.send_message(msg)
        server.quit()
        logger.info(f"Email alert sent with {total_items} items")
    except Exception as e:
        logger.error(f"Error sending email: {e}")
