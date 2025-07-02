import requests
import re

# Use the content from the document instead of reading from file
SUMMARY_TEXT = '''Based on the aggregated analyses of multiple DeFi YouTube video transcripts, here's a summary of the key findings:

**1. Coins:**

* USDC:  The most frequently mentioned stablecoin, consistently appearing across various analyses.
* USDT:  Another frequently mentioned stablecoin, often paired with USDC in liquidity pools.
* BTC:  Bitcoin, mentioned in several contexts, including yield-generating products offered by centralized exchanges.
* ETH: Ethereum, featured in strategies involving lending and staking.
* SOL: Solana's native token, prominent in several Solana-based DeFi strategies.
* PLS:  PulseChain's native token, featured in a specific yield farming strategy.
* mSOL:  Marinade staked SOL, highlighting liquid staking strategies.
* SRUSD: A stablecoin mentioned in a leveraged yield farming strategy.
* D: A stablecoin on PulseChain.
* HEX: Mentioned in conjunction with PulseChain.
* PYUSD, SUSD, USDH:  Various stablecoins featured in high-yield lending strategies.
* WBTC: Wrapped Bitcoin, used in a range-bound yield farming strategy.


**2. Protocols:**

* Morpho:  A DeFi lending protocol highlighted for leveraged yield farming strategies with SRUSD.
* Jupiter:  A DEX on Solana with a significant TVL, offering liquidity provision strategies.
* BMX: Another DEX offering liquidity provision and index token minting.
* Moonwell: A protocol built on Morpho, offering stablecoin yields and integration with a spending card.
* Aave: A prominent lending protocol, mentioned in comparison to others.
* Uphold: A platform offering yield on stablecoins and USD (not strictly a DeFi protocol).
* Coinbase & OKX: Centralized exchanges offering yield products.
* SuperForm: A platform offering high stablecoin yield on the Base network.
* Base: A blockchain utilized as a platform for SuperForm.
* LayerZero & Hyperlink: Bridge protocols utilized in a cross-chain strategy.
* MakerDAO:  Mentioned in the context of yield market trading.
* Spectra (formerly API):  A protocol designed for yield trading.
* Pine: A protocol used by Spectra.
* PulseChain & PulseX: A blockchain and its associated DEX, used for liquidity provision strategies.
* Ether.fi & Upshift: Platforms offering high-yield ETH vaults.
* Marinade Finance:  A liquid staking protocol for SOL.
* Camino Finance:  An automated yield farming protocol on Solana.
* Save: A Solana protocol (partial name provided), possibly focused on capital preservation.
* Orca, Meteora: Prominent Solana DEXs.
* Solend, Marginfi, Drift Protocol, Exponent Finance, Rayax, Asgard Finance: Protocols offering high-yield stablecoin lending.
* Nectari: A Solana-based platform offering savings and lending services for stablecoins.
* Raydium: A Solana-based DEX.
* Pumpf Fun: A Solana launchpad (higher risk).
* InsurAce & Nexus Mutual: Insurance protocols, offering yield through liquidity provision for insurance premiums.


**3. Strategies:**

* **Leveraged Yield Farming:**  Borrowing stablecoins at a lower rate and lending them at a higher rate to amplify returns (Morpho example).  High risk due to liquidation potential.
* **Liquidity Provision on DEXs:** Providing liquidity to DEXs like Jupiter, BMX, Orca, Meteora, and Raydium to earn trading fees.  Risk of impermanent loss.
* **Liquid Staking:** Staking assets (SOL with Marinade) to receive liquid tokens that can be used in other DeFi activities while earning staking rewards.
* **Automated Yield Farming:**  Using platforms like Camino to automate yield farming across multiple protocols, optimizing returns and managing risk.
* **Passive Stablecoin Yield Farming:**  Earning interest on stablecoins deposited on platforms like Uphold, SuperForm, and Nectari.
* **Yield Trading:**  Speculating on the variation of yield offered by different DeFi protocols (Spectra).
* **Range-bound Yield Farming:**  Profiting from price movements within a defined range (Orca example).
* **High-Yield Stablecoin Lending:** Lending stablecoins on platforms like Solend, Marginfi, Drift Protocol, and Exponent Finance for high APYs.
* **Fixed-Term Deposits:**  Depositing stablecoins for a fixed term to earn potentially higher returns (Rayax, when active).
* **ETH Yield Farming Vaults:** Utilizing DeFi vaults (Ether.fi, Upshift) to generate returns on ETH and related tokens.


**4. Watchlist Additions:**

Several protocols and strategies repeatedly appear as promising candidates for a stablecoin yield farming watchlist.  However,  it's crucial to perform thorough due diligence before investing in any of them.  Key factors to consider include:

* **Protocol Track Record & Security Audits:** Established protocols with a history of reliable performance and security audits should be prioritized.
* **TVL & Trading Volume:** Higher TVL and trading volume generally indicate greater liquidity and stability.
* **Risk Assessment:**  Carefully assess the risks associated with each strategy and platform (e.g., impermanent loss, liquidation risk, smart contract vulnerabilities).
* **APY vs. Risk:** Balance high APYs with acceptable levels of risk.  Extremely high yields often signal elevated risk.
* **Stablecoin Stability:**  Consider the stability and reputation of the stablecoins used in each strategy.
* **Diversification:**  Diversify across multiple protocols and strategies to reduce overall risk.

**Specific Watchlist Candidates (requiring further research):**

* **Morpho:** For leveraged yield farming.
* **Jupiter & BMX:** For liquidity provision strategies on Solana.
* **Moonwell:** For stablecoin yields.
* **Aave:** As a benchmark for stablecoin lending.
* **SuperForm:**  For its high-yield stablecoin offering on Base.
* **PulseX:** For its high APY opportunities, but requires careful monitoring of impermanent loss.
* **Marinade Finance & Camino Finance:** For liquid staking and automated yield farming on Solana, respectively.
* **Orca, Meteora, Raydium:** Solana DEXs for stablecoin liquidity provision.
* **Solend, Marginfi, Drift Protocol, Exponent Finance:** High-yield stablecoin lending protocols.
* **Nectari:** For stablecoin savings and lending.
* **Spectra:** For its novel yield trading approach.

This watchlist should be continuously updated and monitored as the DeFi landscape evolves.  Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.  The information provided here is not financial advice.'''

# --- Extract Watchlist from Summary ---
def extract_watchlist(text):
    """Extract watchlist protocols from the summary text"""
    
    # Debug: Let's find the exact section first
    print("🔍 DEBUG: Looking for watchlist section...")
    
    # Look for the section more broadly
    lines = text.split('\n')
    watchlist_start = -1
    watchlist_end = -1
    
    for i, line in enumerate(lines):
        if 'Specific Watchlist Candidates' in line:
            watchlist_start = i
            print(f"Found watchlist section at line {i}: {line}")
            break
    
    if watchlist_start == -1:
        print('❌ No watchlist section found.')
        return []
    
    # Find the end of the section
    for i in range(watchlist_start + 1, len(lines)):
        if lines[i].strip().startswith('This watchlist should'):
            watchlist_end = i
            break
    
    if watchlist_end == -1:
        watchlist_end = len(lines)
    
    # Extract the section
    section_lines = lines[watchlist_start:watchlist_end]
    print(f"📝 DEBUG: Extracted {len(section_lines)} lines from watchlist section")
    
    protocols = []
    
    # Process each line in the section
    for line in section_lines:
        line = line.strip()
        if line.startswith('*') and '**' in line and ':' in line:
            print(f"🔍 Processing line: {line}")
            
            # The format is actually **Protocol:** not **Protocol**:
            # So we need to match content between ** and **: 
            pattern = r'\*\s*\*\*([^*]+):\*\*'
            protocol_match = re.search(pattern, line)
            
            if protocol_match:
                protocol_names = protocol_match.group(1).strip()
                print(f"✅ Found protocol(s): '{protocol_names}'")
                
                # Handle multiple protocols separated by & or ,
                if '&' in protocol_names:
                    for name in protocol_names.split('&'):
                        name = name.strip()
                        if name:
                            protocols.append(name)
                            print(f"  ➕ Added: {name}")
                elif ',' in protocol_names:
                    for name in protocol_names.split(','):
                        name = name.strip()
                        if name:
                            protocols.append(name)
                            print(f"  ➕ Added: {name}")
                else:
                    protocols.append(protocol_names)
                    print(f"  ➕ Added: {protocol_names}")
            else:
                print(f"❌ No match for line: {line}")
                # Let's try a simpler approach - just extract everything between ** **
                simple_pattern = r'\*\*([^*]+)\*\*'
                simple_match = re.search(simple_pattern, line)
                if simple_match:
                    content = simple_match.group(1).strip()
                    if ':' in content:
                        # Remove the colon and everything after it
                        protocol_names = content.split(':')[0].strip()
                        print(f"✅ Found protocol(s) with simple pattern: '{protocol_names}'")
                        
                        # Handle multiple protocols separated by & or ,
                        if '&' in protocol_names:
                            for name in protocol_names.split('&'):
                                name = name.strip()
                                if name:
                                    protocols.append(name)
                                    print(f"  ➕ Added: {name}")
                        elif ',' in protocol_names:
                            for name in protocol_names.split(','):
                                name = name.strip()
                                if name:
                                    protocols.append(name)
                                    print(f"  ➕ Added: {name}")
                        else:
                            protocols.append(protocol_names)
                            print(f"  ➕ Added: {protocol_names}")
    
    return protocols

# --- DeFi Llama API ---
def fetch_protocols():
    """Fetch all protocols from DeFi Llama API"""
    try:
        resp = requests.get("https://api.llama.fi/protocols", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error fetching protocols: {e}")
        return []

def fetch_yield_pools():
    """Fetch yield pools from DeFi Llama API"""
    try:
        resp = requests.get("https://yields.llama.fi/pools", timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.RequestException as e:
        print(f"Error fetching yield pools: {e}")
        return []

def find_protocol(protocols, name):
    """Find a protocol by name with fuzzy matching"""
    name_lower = name.lower()
    
    # Try exact match first
    for p in protocols:
        if name_lower == p['name'].lower():
            return p
    
    # Try partial match
    for p in protocols:
        protocol_name = p['name'].lower()
        if name_lower in protocol_name or protocol_name in name_lower:
            return p
    
    # Try matching with common variations
    name_variations = [
        name_lower.replace(' finance', ''),
        name_lower.replace('finance', ''),
        name_lower + ' finance',
        name_lower.replace(' protocol', ''),
        name_lower.replace('protocol', ''),
    ]
    
    for variation in name_variations:
        for p in protocols:
            if variation == p['name'].lower():
                return p
    
    return None

def find_yield_pools(pools, protocol_name):
    """Find yield pools for a given protocol"""
    matching_pools = []
    protocol_name_lower = protocol_name.lower()
    
    for pool in pools:
        project = pool.get('project', '').lower()
        if protocol_name_lower in project or project in protocol_name_lower:
            matching_pools.append(pool)
    
    return matching_pools

def format_number(num):
    """Format large numbers with appropriate suffixes"""
    if num is None:
        return "N/A"
    
    try:
        num = float(num)
        if num >= 1_000_000_000:
            return f"${num/1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"${num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"${num/1_000:.2f}K"
        else:
            return f"${num:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def run_watchlist_validation():
    print("🔍 Extracting watchlist from summary...")
    watchlist = extract_watchlist(SUMMARY_TEXT)
    
    if not watchlist:
        print('❌ No protocols found in watchlist.')
        return watchlist, []
    
    print(f'✅ Found {len(watchlist)} protocols in watchlist:')
    for i, name in enumerate(watchlist, 1):
        print(f"  {i}. {name}")
    
    print("\n🌐 Fetching data from DeFi Llama...")
    protocols = fetch_protocols()
    pools = fetch_yield_pools()
    
    if not protocols:
        print("❌ Failed to fetch protocol data.")
        return watchlist, []
    
    print(f"✅ Fetched {len(protocols)} protocols and {len(pools)} yield pools from DeFi Llama.")
    print("\n" + "="*80)
    print("📊 WATCHLIST ANALYSIS")
    print("="*80)
    
    found_count = 0
    output_lines = []
    output_lines.append(f"WATCHLIST ANALYSIS\n{'='*80}")
    for name in watchlist:
        print(f"\n🔍 {name.upper()}:")
        print("-" * 50)
        output_lines.append(f"\n{name.upper()}\n{'-'*50}")
        
        proto = find_protocol(protocols, name)
        if not proto:
            print(f"❌ Protocol '{name}' not found on DeFi Llama.")
            output_lines.append(f"❌ Protocol '{name}' not found on DeFi Llama.")
            continue
        
        found_count += 1
        
        # Protocol info
        tvl = proto.get('tvl')
        tvl_str = format_number(tvl)
        print(f"💰 TVL: {tvl_str}")
        output_lines.append(f"TVL: {tvl_str}")
        
        chains = proto.get('chains', [])
        if chains:
            chains_str = ', '.join(chains[:5])
            print(f"🔗 Chains: {chains_str}")
            output_lines.append(f"Chains: {chains_str}")
            if len(chains) > 5:
                more_chains = f"... and {len(chains) - 5} more"
                print(f"    {more_chains}")
                output_lines.append(f"    {more_chains}")
        
        category = proto.get('category', 'N/A')
        print(f"📁 Category: {category}")
        output_lines.append(f"Category: {category}")
        
        # Find yield pools
        proto_pools = find_yield_pools(pools, proto['name'])
        if not proto_pools:
            print("📈 No yield pools found.")
            output_lines.append("No yield pools found.")
            continue
        
        print(f"📈 Top Yield Opportunities ({len(proto_pools)} total pools):")
        output_lines.append(f"Top Yield Opportunities ({len(proto_pools)} total pools):")
        
        # Sort pools by APY (descending) and show top 3
        valid_pools = [p for p in proto_pools if p.get('apy') is not None]
        sorted_pools = sorted(valid_pools, key=lambda x: x.get('apy', 0), reverse=True)
        
        for i, pool in enumerate(sorted_pools[:3], 1):
            apy = pool.get('apy')
            apy_str = f"{apy*100:.2f}%" if apy is not None else 'N/A'
            symbol = pool.get('symbol', 'N/A')
            chain = pool.get('chain', 'N/A')
            pool_meta = pool.get('poolMeta', '')
            print(f"    {i}. {symbol} on {chain}")
            print(f"       APY: {apy_str}")
            if pool_meta:
                print(f"       Info: {pool_meta}")
            output_lines.append(f"  {i}. {symbol} on {chain}")
            output_lines.append(f"     APY: {apy_str}")
            if pool_meta:
                output_lines.append(f"     Info: {pool_meta}")
    
    print(f"\n📊 Summary: Found {found_count}/{len(watchlist)} protocols on DeFi Llama")
    output_lines.append(f"\nSummary: Found {found_count}/{len(watchlist)} protocols on DeFi Llama")
    
    # Store results in a file
    with open('watchlist_llama_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print("\n✅ Results saved to 'watchlist_llama_results.txt'")
    return watchlist, output_lines

def main():
    run_watchlist_validation()

if __name__ == '__main__':
    main()