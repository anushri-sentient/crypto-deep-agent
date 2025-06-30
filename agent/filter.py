from typing import List, Dict

def contains_financial_content(text: str, stock_symbols, financial_keywords, logger=None) -> bool:
    text_lower = text.lower()
    if logger:
        logger.debug(f"Checking text: '{text_lower[:100]}...'")
    for symbol in stock_symbols:
        if f"${symbol.lower()}" in text_lower or f" {symbol.lower()} " in text_lower:
            if logger:
                logger.debug(f"Found stock symbol match: {symbol}")
            return True
    for keyword in financial_keywords:
        if keyword in text_lower:
            if logger:
                logger.debug(f"Found financial keyword match: '{keyword}' in text")
            return True
    if logger:
        logger.debug("No financial content found")
    return False

def filter_high_priority_content(content: List[Dict], min_priority_score: int, logger, stock_symbols, financial_keywords) -> List[Dict]:
    high_priority = []
    for item in content:
        priority_score = 0
        if item['source'] == 'YouTube':
            view_count = int(item.get('view_count', 0))
            if view_count > 10000:
                priority_score += 2
            elif view_count > 1000:
                priority_score += 1
        elif item['source'] == 'Reddit':
            score = item.get('score', 0)
            comments = item.get('num_comments', 0)
            if score > 100 or comments > 50:
                priority_score += 2
            elif score > 20 or comments > 10:
                priority_score += 1
        elif item['source'] in ['News', 'RSS']:
            priority_score += 1
            source_name = item.get('source_name', '').lower()
            if any(trusted in source_name for trusted in ['bloomberg', 'reuters', 'wsj', 'cnbc']):
                priority_score += 1
        text_to_check = (
            item.get('title', '') + ' ' + 
            item.get('text', '') + ' ' + 
            item.get('description', '')
        ).lower()
        urgent_keywords = ['breaking', 'alert', 'crash', 'surge', 'earnings', 'fed']
        for keyword in urgent_keywords:
            if keyword in text_to_check:
                priority_score += 1
                break
        if priority_score >= min_priority_score:
            high_priority.append({**item, 'priority_score': priority_score})
    return sorted(high_priority, key=lambda x: x['priority_score'], reverse=True)
