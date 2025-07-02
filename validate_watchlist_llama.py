import requests
import re
import os

# Dynamically load SUMMARY_TEXT from video_analyses.txt if available
VIDEO_ANALYSES_PATH = os.path.join(os.path.dirname(__file__), 'video_analyses.txt')
try:
    with open(VIDEO_ANALYSES_PATH, 'r', encoding='utf-8') as f:
        SUMMARY_TEXT = f.read()
except Exception as e:
    SUMMARY_TEXT = 'No video analysis summary available. Please ensure video_analyses.txt exists.'

# --- Extract Watchlist from Summary ---
def extract_watchlist(text):
    """Extract watchlist protocols from the summary text"""
    
    print("🔍 DEBUG: Looking for watchlist section...")
    lines = text.split('\n')
    watchlist_start = -1
    watchlist_end = -1
    # Accept multiple possible section headers
    possible_headers = [
        'Specific Watchlist Candidates',
        'Watchlist Suggestions',
        '4. Watchlist Suggestions',
        '4. Specific Watchlist Candidates',
    ]
    for i, line in enumerate(lines):
        for header in possible_headers:
            if header in line:
                watchlist_start = i
                print(f"Found watchlist section at line {i}: {line}")
                break
        if watchlist_start != -1:
            break
    if watchlist_start == -1:
        print('❌ No watchlist section found.')
        return []
    # Find the end of the section
    for i in range(watchlist_start + 1, len(lines)):
        if lines[i].strip().startswith('**Important Note') or lines[i].strip().startswith('This watchlist should'):
            watchlist_end = i
            break
    if watchlist_end == -1:
        watchlist_end = len(lines)
    section_lines = lines[watchlist_start:watchlist_end]
    print(f"📝 DEBUG: Extracted {len(section_lines)} lines from watchlist section")
    protocols = []
    for line in section_lines:
        line = line.strip()
        if line.startswith('*') and '**' in line and ':' in line:
            print(f"🔍 Processing line: {line}")
            pattern = r'\*\s*\*\*([^*]+):\*\*'
            protocol_match = re.search(pattern, line)
            if protocol_match:
                protocol_names = protocol_match.group(1).strip()
                print(f"✅ Found protocol(s): '{protocol_names}'")
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
                simple_pattern = r'\*\*([^*]+)\*\*'
                simple_match = re.search(simple_pattern, line)
                if simple_match:
                    content = simple_match.group(1).strip()
                    if ':' in content:
                        protocol_names = content.split(':')[0].strip()
                        print(f"✅ Found protocol(s) with simple pattern: '{protocol_names}'")
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
    structured = []
    output_lines.append(f"WATCHLIST ANALYSIS\n{'='*80}")
    for name in watchlist:
        print(f"\n🔍 {name.upper()}:")
        print("-" * 50)
        output_lines.append(f"\n{name.upper()}\n{'-'*50}")
        proto = find_protocol(protocols, name)
        if not proto:
            print(f"❌ Protocol '{name}' not found on DeFi Llama.")
            output_lines.append(f"❌ Protocol '{name}' not found on DeFi Llama.")
            structured.append({
                'protocol': name,
                'found': False,
                'tvl': None,
                'chains': [],
                'category': None,
                'pools': []
            })
            continue
        found_count += 1
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
        proto_pools = find_yield_pools(pools, proto['name'])
        if not proto_pools:
            print("📈 No yield pools found.")
            output_lines.append("No yield pools found.")
            structured.append({
                'protocol': proto['name'],
                'found': True,
                'tvl': tvl,
                'chains': chains,
                'category': category,
                'pools': []
            })
            continue
        print(f"📈 Top Yield Opportunities ({len(proto_pools)} total pools):")
        output_lines.append(f"Top Yield Opportunities ({len(proto_pools)} total pools):")
        valid_pools = [p for p in proto_pools if p.get('apy') is not None]
        sorted_pools = sorted(valid_pools, key=lambda x: x.get('apy', 0), reverse=True)
        top_pools = []
        for i, pool in enumerate(sorted_pools[:3], 1):
            apy = pool.get('apy')
            apy_str = f"{apy:.2f}%" if apy is not None else 'N/A'
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
            top_pools.append({
                'rank': i,
                'symbol': symbol,
                'chain': chain,
                'apy': apy,
                'apy_str': apy_str,
                'info': pool_meta
            })
        structured.append({
            'protocol': proto['name'],
            'found': True,
            'tvl': tvl,
            'chains': chains,
            'category': category,
            'pools': top_pools
        })
    print(f"\n📊 Summary: Found {found_count}/{len(watchlist)} protocols on DeFi Llama")
    output_lines.append(f"\nSummary: Found {found_count}/{len(watchlist)} protocols on DeFi Llama")
    with open('watchlist_llama_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print("\n✅ Results saved to 'watchlist_llama_results.txt'")
    return watchlist, structured

def main():
    run_watchlist_validation()

if __name__ == '__main__':
    main()