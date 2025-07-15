import os
import re
import json
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CryptoStrategyAgent:
    """
    Crypto Strategy Agent that extracts strategies and recommendations from YouTube summaries
    and provides live DeFi data for mentioned coins/tokens/pools.
    """
    
    def __init__(self):
        self.defi_llama_base_url = "https://api.llama.fi"
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Common DeFi protocols and their identifiers
        self.defi_protocols = {
            'aave': 'Aave',
            'compound': 'Compound',
            'curve': 'Curve',
            'uniswap': 'Uniswap',
            'sushiswap': 'SushiSwap',
            'balancer': 'Balancer',
            'yearn': 'Yearn Finance',
            'convex': 'Convex Finance',
            'pendle': 'Pendle',
            'tokemak': 'Tokemak',
            'kamino': 'Kamino Finance',
            'marginfi': 'MarginFi',
            'raydium': 'Raydium',
            'orca': 'Orca',
            'francium': 'Francium',
            'multipli': 'Multipli',
            'merkl': 'Merkl',
            'stakedao': 'Stake DAO',
            'dolomite': 'Dolomite',
            'gearbox': 'Gearbox',
            'exactly': 'Exactly',
            'latch': 'Latch',
            'sandclock': 'Sandclock',
            'hyperdrive': 'Hyperdrive',
            'integral': 'Integral',
            'clipper': 'Clipper'
        }
        
        # Strategy keywords to look for
        self.strategy_keywords = [
            'yield farming', 'liquidity mining', 'staking', 'lending', 'borrowing',
            'liquidity provision', 'LP', 'vault', 'strategy', 'pool', 'farm',
            'compound', 'reinvest', 'harvest', 'claim rewards', 'auto-compound',
            'leverage', 'short', 'long', 'hedge', 'arbitrage', 'flash loan',
            'collateral', 'debt', 'interest', 'APY', 'APR', 'yield', 'returns'
        ]
    
    def extract_strategies_and_recommendations(self, youtube_summary: str) -> Dict:
        """
        Extract strategies and recommendations from YouTube summary using Gemini LLM
        """
        if not self.gemini_api_key:
            logger.warning("No Gemini API key found. Using basic extraction.")
            return self._basic_extraction(youtube_summary)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            
            prompt = """
            Analyze this YouTube video summary about DeFi/crypto and extract strategies and recommendations.
            
            Return ONLY a valid JSON object with this exact structure:
            {
                "strategies": [
                    {
                        "name": "Strategy name",
                        "description": "Brief description",
                        "execution_steps": ["Step 1", "Step 2", "Step 3"],
                        "protocols": ["protocol1", "protocol2"],
                        "risks": ["risk1", "risk2"]
                    }
                ],
                "recommendations": [
                    {
                        "asset": "Asset name/symbol",
                        "type": "coin/token/pool",
                        "protocol": "Protocol name",
                        "expected_return": "APY/APR if mentioned",
                        "confidence": "high/medium/low"
                    }
                ]
            }
            
            Extract:
            1. STRATEGIES: yield strategies, farming methods, investment approaches with execution steps
            2. RECOMMENDATIONS: specific coin/token/pool recommendations with protocols and expected returns
            
            YouTube Summary:
            """ + youtube_summary
            
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Try to parse JSON response
            try:
                # Clean up the response text to extract JSON
                # Remove any markdown formatting
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    result_text = result_text.split('```')[1]
                
                # Remove any leading/trailing whitespace and newlines
                result_text = result_text.strip()
                
                return json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.warning(f"Response text: {result_text[:200]}...")
                return self._basic_extraction(youtube_summary)
                
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return self._basic_extraction(youtube_summary)
    
    def _basic_extraction(self, youtube_summary: str) -> Dict:
        """
        Basic extraction without Gemini LLM
        """
        strategies = []
        recommendations = []
        
        # Extract strategies based on keywords
        summary_lower = youtube_summary.lower()
        
        # Look for strategy patterns
        for keyword in self.strategy_keywords:
            if keyword in summary_lower:
                # Find context around the keyword
                start = max(0, summary_lower.find(keyword) - 100)
                end = min(len(summary_lower), summary_lower.find(keyword) + 200)
                context = youtube_summary[start:end]
                
                strategies.append({
                    "name": f"{keyword.title()} Strategy",
                    "description": context.strip(),
                    "execution_steps": [f"Research {keyword} opportunities", f"Identify suitable protocols", "Execute strategy"],
                    "protocols": [],
                    "risks": ["Market volatility", "Smart contract risk", "Impermanent loss"]
                })
        
        # Extract coin/token mentions with better filtering
        coin_patterns = [
            r'\b[A-Z]{2,10}\b',  # 2-10 letter uppercase tokens
            r'\b[A-Z][a-z]+[A-Z][a-z]*\b',  # CamelCase tokens
            r'\$[A-Z]{2,10}\b',  # $SYMBOL format
        ]
        
        exclude_words = {
            'APR', 'APY', 'TVL', 'USD', 'USDC', 'USDT', 'DAI', 'ETH', 'BTC', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'BASE',
            'DEFI', 'CRYPTO', 'YIELD', 'STABLE', 'COIN', 'TOKEN', 'PROTOCOL', 'PLATFORM', 'STRATEGY', 'RISK', 'REWARD',
            'LENDING', 'BORROWING', 'LIQUIDITY', 'FARMING', 'STAKING', 'VAULT', 'POOL', 'SWAP', 'DEX', 'AMM', 'LP'
        }
        
        found_coins = set()
        for pattern in coin_patterns:
            matches = re.findall(pattern, youtube_summary)
            for match in matches:
                clean_match = match.replace('$', '').strip()
                if (len(clean_match) >= 2 and 
                    clean_match not in exclude_words and
                    not clean_match.isdigit() and
                    clean_match not in found_coins):
                    
                    # Try to extract APY/APR information
                    expected_return = "Not specified"
                    apy_match = re.search(rf'{clean_match}.*?(\d+\.?\d*)\s*%?\s*APY', youtube_summary, re.IGNORECASE)
                    if apy_match:
                        expected_return = f"{apy_match.group(1)}% APY"
                    
                    # Try to identify protocol
                    protocol = "Unknown"
                    for proto_name, proto_display in self.defi_protocols.items():
                        if proto_name.lower() in youtube_summary.lower():
                            protocol = proto_display
                            break
                    
                    recommendations.append({
                        "asset": clean_match,
                        "type": "token",
                        "protocol": protocol,
                        "expected_return": expected_return,
                        "confidence": "medium" if expected_return != "Not specified" else "low"
                    })
                    found_coins.add(clean_match)
        
        return {
            "strategies": strategies,
            "recommendations": recommendations
        }
    
    def get_live_defi_data(self, asset_name: str, protocol_name: str = None) -> Optional[Dict]:
        """
        Get live DeFi data for a specific asset/protocol from DeFi Llama
        """
        try:
            # Search for protocols
            if protocol_name:
                protocols_response = requests.get(f"{self.defi_llama_base_url}/protocols")
                if protocols_response.status_code == 200:
                    protocols = protocols_response.json()
                    
                    # Find matching protocol
                    for protocol in protocols:
                        if protocol_name.lower() in protocol['name'].lower():
                            protocol_slug = protocol['slug']
                            
                            # Get protocol details
                            protocol_response = requests.get(f"{self.defi_llama_base_url}/protocol/{protocol_slug}")
                            if protocol_response.status_code == 200:
                                protocol_data = protocol_response.json()
                                
                                # Get pools/tokens for this protocol
                                pools_response = requests.get(f"{self.defi_llama_base_url}/protocol/{protocol_slug}/pools")
                                if pools_response.status_code == 200:
                                    pools_data = pools_response.json()
                                    
                                    # Find pools containing the asset
                                    matching_pools = []
                                    for pool in pools_data:
                                        if asset_name.lower() in str(pool).lower():
                                            matching_pools.append(pool)
                                    
                                    return {
                                        "protocol": protocol_data,
                                        "pools": matching_pools,
                                        "asset": asset_name
                                    }
            
            # If no protocol specified, search for the asset across all protocols
            all_protocols_response = requests.get(f"{self.defi_llama_base_url}/protocols")
            if all_protocols_response.status_code == 200:
                all_protocols = all_protocols_response.json()
                
                results = []
                for protocol in all_protocols[:20]:  # Limit to first 20 for performance
                    try:
                        pools_response = requests.get(f"{self.defi_llama_base_url}/protocol/{protocol['slug']}/pools")
                        if pools_response.status_code == 200:
                            pools = pools_response.json()
                            for pool in pools:
                                if asset_name.lower() in str(pool).lower():
                                    results.append({
                                        "protocol": protocol['name'],
                                        "protocol_slug": protocol['slug'],
                                        "pool": pool
                                    })
                    except Exception as e:
                        continue
                
                if results:
                    return {
                        "asset": asset_name,
                        "found_in": results
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching DeFi data for {asset_name}: {e}")
            return None
    
    def get_token_price_data(self, token_symbol: str) -> Optional[Dict]:
        """
        Get token price data from CoinGecko API
        """
        try:
            # First, search for the token
            search_url = f"https://api.coingecko.com/api/v3/search?query={token_symbol}"
            search_response = requests.get(search_url)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                if search_data['coins']:
                    # Get the first match
                    coin_id = search_data['coins'][0]['id']
                    
                    # Get detailed price data
                    price_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                    price_response = requests.get(price_url)
                    
                    if price_response.status_code == 200:
                        return price_response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching price data for {token_symbol}: {e}")
            return None
    
    def analyze_youtube_summary(self, youtube_summary: str) -> Dict:
        """
        Main function to analyze YouTube summary and provide actionable insights
        """
        logger.info("Starting YouTube summary analysis...")
        
        # Extract strategies and recommendations
        extraction_result = self.extract_strategies_and_recommendations(youtube_summary)
        
        # Get live data for recommendations
        live_data = {}
        for recommendation in extraction_result.get('recommendations', []):
            asset = recommendation['asset']
            protocol = recommendation.get('protocol', 'Unknown')
            
            # Get DeFi data
            defi_data = self.get_live_defi_data(asset, protocol)
            if defi_data:
                live_data[asset] = {
                    'defi_data': defi_data,
                    'recommendation': recommendation
                }
            
            # Get price data
            price_data = self.get_token_price_data(asset)
            if price_data:
                if asset not in live_data:
                    live_data[asset] = {}
                live_data[asset]['price_data'] = price_data
        
        return {
            'extraction': extraction_result,
            'live_data': live_data,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def format_results_for_display(self, analysis_result: Dict) -> str:
        """
        Format analysis results for easy reading
        """
        output = []
        output.append("=" * 80)
        output.append("CRYPTO STRATEGY AGENT ANALYSIS RESULTS")
        output.append("=" * 80)
        output.append(f"Analysis Time: {analysis_result['analysis_timestamp']}")
        output.append("")
        
        # Strategies Section
        strategies = analysis_result['extraction'].get('strategies', [])
        if strategies:
            output.append("🎯 STRATEGIES IDENTIFIED:")
            output.append("-" * 40)
            for i, strategy in enumerate(strategies, 1):
                output.append(f"{i}. {strategy['name']}")
                output.append(f"   Description: {strategy['description'][:200]}...")
                output.append("   Execution Steps:")
                for step in strategy['execution_steps']:
                    output.append(f"     • {step}")
                if strategy['risks']:
                    output.append("   Risks:")
                    for risk in strategy['risks']:
                        output.append(f"     ⚠️ {risk}")
                output.append("")
        
        # Recommendations Section
        recommendations = analysis_result['extraction'].get('recommendations', [])
        if recommendations:
            output.append("💎 RECOMMENDATIONS:")
            output.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                output.append(f"{i}. {rec['asset']} ({rec['type']})")
                output.append(f"   Protocol: {rec['protocol']}")
                output.append(f"   Expected Return: {rec['expected_return']}")
                output.append(f"   Confidence: {rec['confidence']}")
                output.append("")
        
        # Live Data Section
        live_data = analysis_result.get('live_data', {})
        if live_data:
            output.append("📊 LIVE DATA:")
            output.append("-" * 40)
            for asset, data in live_data.items():
                output.append(f"🔸 {asset}:")
                
                # Price data
                if 'price_data' in data:
                    price_info = data['price_data']
                    if 'market_data' in price_info:
                        market_data = price_info['market_data']
                        output.append(f"   💰 Current Price: ${market_data.get('current_price', {}).get('usd', 'N/A')}")
                        output.append(f"   📈 24h Change: {market_data.get('price_change_percentage_24h', 'N/A')}%")
                        output.append(f"   💎 Market Cap: ${market_data.get('market_cap', {}).get('usd', 'N/A'):,}")
                
                # DeFi data
                if 'defi_data' in data:
                    defi_info = data['defi_data']
                    if 'protocol' in defi_info:
                        protocol = defi_info['protocol']
                        output.append(f"   🏦 Protocol: {protocol.get('name', 'N/A')}")
                        output.append(f"   📊 TVL: ${protocol.get('tvl', 'N/A'):,}")
                    
                    if 'pools' in defi_info and defi_info['pools']:
                        output.append(f"   🏊 Found {len(defi_info['pools'])} pools")
                        for pool in defi_info['pools'][:3]:  # Show first 3 pools
                            if isinstance(pool, dict):
                                apy = pool.get('apy', 'N/A')
                                output.append(f"     • APY: {apy}%")
                
                output.append("")
        
        return "\n".join(output)


def main():
    """
    Test function for the Crypto Strategy Agent
    """
    # Example YouTube summary for testing
    test_summary = """
    In this video, I'm covering the best DeFi yield farming strategies for 2024. 
    We're looking at Curve Finance pools which are offering 8-12% APY on stablecoin pairs.
    I recommend checking out the USDC/USDT pool on Curve, which currently has 9.5% APY.
    Another great opportunity is Aave lending where you can earn 4-6% APY on USDC deposits.
    For higher yields, consider Yearn Finance vaults which can provide 15-20% APY through
    automated strategies. The YFI token has been performing well and the vaults are very safe.
    Also look into Convex Finance for Curve LP token staking - you can earn additional CVX rewards.
    """
    
    agent = CryptoStrategyAgent()
    
    print("Testing Crypto Strategy Agent...")
    print("=" * 50)
    
    # Analyze the test summary
    results = agent.analyze_youtube_summary(test_summary)
    
    # Display formatted results
    formatted_output = agent.format_results_for_display(results)
    print(formatted_output)
    
    # Save results to file
    with open("crypto_agent_results.txt", "w") as f:
        f.write(formatted_output)
    
    print("\nResults saved to crypto_agent_results.txt")


if __name__ == "__main__":
    main() 