# Crypto DeepSearch Agent: Stablecoin Yield Pipeline

A modular Python pipeline for end-to-end DeFi stablecoin yield research. This project automates the process of:

1. **Mapping the "blue-chip" DeFi landscape**
2. **Scoring stablecoin pools by reward vs. risk**
3. **Hunting for fresh ideas on social channels (YouTube)**
4. **Validating every newcomer with live DeFiLlama data**

## Features
- Fetches and filters protocols from DeFiLlama for the busiest chains and key categories
- Scores and tags stablecoin pools by risk and yield
- Scrapes YouTube for recent DeFi yield videos and extracts new protocols/strategies
- Validates new candidates using DeFiLlama and risk/yield logic
- Outputs comprehensive CSV and text reports for each step

## Project Structure
```
crypto-deepsearch-agent/
│
├── agent/                       # Modular agent components (YouTube, Reddit, News, etc.)
├── bluechip_landscape.py        # Step 1 & 2: Protocol and pool curation, risk/yield scoring
├── youtube_scraper.py           # Step 3: Social channel idea mining (YouTube + Gemini LLM)
├── validate_watchlist_llama.py  # Step 4: Validate newcomers, cross-check with DeFiLlama
├── stablecoin_yield_pipeline.py # Orchestrator: Runs the full 4-step pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup
1. **Clone the repo**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**:
   - `YOUTUBE_API_KEY` (for YouTube scraping)
   - `GEMINI_API_KEY` (for Gemini LLM analysis)
   - See `.env.example` for more options if needed

## Usage
Run the full pipeline with:
```bash
python stablecoin_yield_pipeline.py
```
This will execute all four steps and generate output files in the `output/` directory.

## Output Files (in `output/` directory)
- **filtered_protocols.csv**: All filtered DeFi protocols (blue-chip landscape)
- **evaluated_stablecoin_pools.csv**: All scored/tagged stablecoin pools with risk/yield assessment
- **youtube_yield_candidates.txt**: List of YouTube videos scraped for DeFi yield ideas
- **youtube_gemini_analysis.txt**: Gemini LLM analysis of each video (insights, protocols, strategies)
- **youtube_gemini_summary.txt**: Summary of all Gemini analyses (key coins, protocols, strategies, watchlist suggestions)
- **youtube_yield_candidates.csv**: (If present) CSV version of YouTube video results
- **watchlist_llama_results.txt**: Validation of watchlist protocols/strategies using live DeFiLlama data

## The Four-Step Pipeline

### 1. Map the Blue-Chip Landscape
- Fetches all DeFi protocols from DeFiLlama
- Filters for busiest chains (Ethereum, Arbitrum, Optimism, Base, Polygon, Solana) and key categories (lending, DEX, yield)
- Always includes Kamino, Marginfi, Raydium, Gamma
- Output: `filtered_protocols.csv`

### 2. Score Each Pool's Reward vs. Risk
- Evaluates all stablecoin pools for filtered protocols
- Tags by coin quality, risk, and yield
- Output: `evaluated_stablecoin_pools.csv`

### 3. Hunt for Fresh Ideas on Social Channels
- Scrapes YouTube for recent DeFi yield videos (last 1-3 months)
- Runs Gemini LLM to extract protocols, strategies, and watchlist suggestions
- Outputs: `youtube_yield_candidates.txt`, `youtube_gemini_analysis.txt`, `youtube_gemini_summary.txt`

### 4. Validate Every Newcomer
- Cross-checks new protocols/strategies with live DeFiLlama data
- Outputs a validation report: `watchlist_llama_results.txt`

## Extending
- Add more social sources (e.g., Twitter, Reddit) by extending the agent modules
- Customize risk/yield scoring logic in `bluechip_landscape.py`
- Adjust YouTube scraping and LLM prompts in `youtube_scraper.py`

## License
MIT
