# Crypto DeepSearch Agent

A modular Python agent that monitors YouTube, Reddit, and news sources for high-priority financial and crypto content, and sends email alerts.

## Features
- Monitors trending YouTube videos, Reddit posts, and financial news
- Filters for financial/crypto relevance and high engagement
- Sends email alerts for high-priority items
- Highly configurable via `.env` file
- Modular, testable codebase

## Project Structure
```
crypto-deepsearch-agent/
│
├── agent/
│   ├── config.py         # Loads and validates environment/config
│   ├── logger.py         # Centralized logger setup
│   ├── keywords.py       # Default and custom keywords/symbols
│   ├── filter.py         # Filtering and priority logic
│   ├── emailer.py        # Email sending logic
│   └── sources/
│       ├── youtube.py    # YouTube trending fetcher
│       ├── reddit.py     # Reddit finance fetcher
│       ├── news.py       # News API fetcher
│       └── rss.py        # RSS fallback fetcher
│
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment config
└── README.md             # This file
```

## Setup
1. **Clone the repo**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Copy and edit the environment file**:
   ```bash
   cp .env.example .env
   # Edit .env and fill in your API keys and email credentials
   ```

## Configuration
All configuration is via environment variables in `.env`. See `.env.example` for all options. At minimum, set:
- `YOUTUBE_API_KEY`
- `EMAIL_USERNAME`, `EMAIL_PASSWORD`, `RECIPIENT_EMAIL`

Optional: Reddit and News API keys, custom keywords, etc.

## Usage
- **Single scan:**
  ```bash
  python main.py
  ```
- **Continuous monitoring:**
  Set `CONTINUOUS_MODE=true` in `.env`.

## Extending
- Add new sources: create a new file in `agent/sources/` and import in `main.py`.
- Add new keywords or symbols: edit `agent/keywords.py` or use `CUSTOM_KEYWORDS`/`CUSTOM_STOCK_SYMBOLS` in `.env`.
