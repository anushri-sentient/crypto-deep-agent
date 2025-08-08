import requests
import time
from typing import Dict, List, Optional


def get_dexscreener_pair_data(symbol: str, chain: str = None, debug: bool = False) -> Optional[Dict]:
    """Get detailed pair data from DexScreener API by token pair"""
    try:
        # Split symbols
        clean_symbol = symbol.replace(" ", "").upper()
        if "-" in clean_symbol:
            base_target, quote_target = clean_symbol.split("-")
        else:
            base_target = clean_symbol
            quote_target = ""

        if debug:
            print(f"🔍 DexScreener Search: Base = {base_target}, Quote = {quote_target}, Chain = {chain or 'Any'}")

        # Search DexScreener
        url = f"https://api.dexscreener.com/latest/dex/search?q={base_target}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pairs = data.get("pairs", [])
        if debug:
            print(f"📊 Found {len(pairs)} pairs for {base_target}")
        
        if not pairs:
            if debug:
                print(f"❌ No pairs found for {base_target}")
            return None

        # Look for best match
        best_match = None
        for i, pair in enumerate(pairs[:5]):  # Only check first 5 to avoid spam
            pair_chain = pair.get("chainId", "").lower()
            base_symbol = pair.get("baseToken", {}).get("symbol", "").upper()
            quote_symbol = pair.get("quoteToken", {}).get("symbol", "").upper()

            if debug:
                print(f"  {i+1}. {base_symbol}-{quote_symbol} on {pair_chain}")

            # Match both base and quote symbols
            if (
                (not chain or pair_chain == chain.lower()) and
                (
                    (base_target == base_symbol and quote_target == quote_symbol) or
                    (base_target == quote_symbol and quote_target == base_symbol)
                )
            ):
                best_match = pair
                if debug:
                    print(f"✅ Found exact match: {base_symbol}-{quote_symbol} on {pair_chain}")
                break

        # Fallback to first result if nothing matches exactly
        if not best_match and pairs:
            best_match = pairs[0]
            if debug:
                print(f"⚠️ Using fallback: {best_match.get('baseToken', {}).get('symbol')}-{best_match.get('quoteToken', {}).get('symbol')}")

        return best_match

    except Exception as e:
        if debug:
            print(f"❌ Error fetching DexScreener data for {symbol}: {e}")
        return None


def get_pool_analytics(pool_data: Dict) -> Dict:
    """Extract important analytics from pool data"""
    analytics = {
        "price_usd": None,
        "price_change_24h": None,
        "volume_24h": None,
        "liquidity_usd": None,
        "fdv": None,
        "market_cap": None,
        "dex": None,
        "pair_address": None,
        "base_token": None,
        "quote_token": None,
        "txns_24h": None,
        "buy_tax": None,
        "sell_tax": None,
        # Extended fields likely present on pairs/search endpoints
        "url": None,
        "labels": None,
        "pair_created_at": None,
        "info_image_url": None,
        "info_websites": None,
        "info_socials": None,
        "boosts_active": None,
        "chain_id": None,
    }

    if not pool_data:
        return analytics

    try:
        analytics.update({
            "price_usd": pool_data.get("priceUsd"),
            "price_change_24h": pool_data.get("priceChange", {}).get("h24"),
            "volume_24h": pool_data.get("volume", {}).get("h24"),
            "liquidity_usd": pool_data.get("liquidity", {}).get("usd"),
            "fdv": pool_data.get("fdv"),
            "market_cap": pool_data.get("marketCap"),
            "dex": pool_data.get("dexId"),
            "pair_address": pool_data.get("pairAddress"),
            "base_token": pool_data.get("baseToken", {}).get("symbol"),
            "quote_token": pool_data.get("quoteToken", {}).get("symbol"),
            "txns_24h": pool_data.get("txns", {}).get("h24"),
            "buy_tax": pool_data.get("buyTax"),
            "sell_tax": pool_data.get("sellTax"),
            "url": pool_data.get("url"),
            "labels": pool_data.get("labels"),
            "pair_created_at": pool_data.get("pairCreatedAt"),
            "info_image_url": pool_data.get("info", {}).get("imageUrl"),
            "info_websites": [w.get("url") for w in (pool_data.get("info", {}).get("websites") or []) if isinstance(w, dict)],
            "info_socials": pool_data.get("info", {}).get("socials"),
            "boosts_active": (pool_data.get("boosts") or {}).get("active"),
            "chain_id": pool_data.get("chainId"),
        })
    except Exception:
        # Keep silent to avoid noisy UI; analytics keys default to None
        pass
    
    return analytics


# ------ Enhanced resolution helpers ------

# Map DefiLlama chain names to DexScreener chainId slugs
DEX_CHAIN_MAP: Dict[str, str] = {
    "ethereum": "ethereum",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon",
    "polygon zkevm": "polygon-zkevm",
    "binance": "bsc",
    "bsc": "bsc",
    "bnb": "bsc",
    "avalanche": "avalanche",
    "avax": "avalanche",
    "base": "base",
    "fantom": "fantom",
    "gnosis": "gnosis",
    "celo": "celo",
    "kava": "kava",
    "metis": "metis",
    "zksync": "zksync",
    "zksync era": "zksync",
    "linea": "linea",
    "scroll": "scroll",
    "blast": "blast",
    "solana": "solana",
    # Additional mappings for common variations
    "eth": "ethereum",
    "mainnet": "ethereum",
    "polygon zk": "polygon-zkevm",
    "polygon-zkevm": "polygon-zkevm",
    "binance smart chain": "bsc",
    "binance-smart-chain": "bsc",
    "avalanche c-chain": "avalanche",
    "avalanche-c-chain": "avalanche",
    "gnosis chain": "gnosis",
    "gnosis-chain": "gnosis",
    "zk sync": "zksync",
    "zk-sync": "zksync",
    "zk sync era": "zksync",
    "zk-sync-era": "zksync",
}


def map_chain_name_to_dex_id(chain_name: Optional[str]) -> Optional[str]:
    if not chain_name:
        return None
    key = str(chain_name).strip().lower()
    return DEX_CHAIN_MAP.get(key, key)


def search_dexscreener(query: str, debug: bool = False) -> List[Dict]:
    try:
        url = f"https://api.dexscreener.com/latest/dex/search?q={query}"
        if debug:
            print(f"🔍 DexScreener API Request: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pairs = data.get("pairs", [])
        if debug:
            print(f"📊 DexScreener Search Response: {len(pairs)} pairs found")
            if pairs:
                print(f"   First pair: {pairs[0].get('baseToken', {}).get('symbol')}-{pairs[0].get('quoteToken', {}).get('symbol')} on {pairs[0].get('chainId')}")
        return pairs
    except Exception as e:
        if debug:
            print(f"❌ search_dexscreener error for '{query}': {e}")
        return []


def fetch_pair_details(chain_id: str, pair_address: str, debug: bool = False) -> Optional[Dict]:
    """GET /latest/dex/pairs/{chainId}/{pairId}"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/pairs/{chain_id}/{pair_address}"
        if debug:
            print(f"🔍 DexScreener Pair Details Request: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pairs = data.get("pairs") or []
        if debug:
            print(f"📊 DexScreener Pair Details Response: {len(pairs)} pairs found")
            if pairs:
                pair = pairs[0]
                print(f"   Pair: {pair.get('baseToken', {}).get('symbol')}-{pair.get('quoteToken', {}).get('symbol')} on {pair.get('chainId')}")
                print(f"   Price: ${pair.get('priceUsd')}")
                print(f"   Volume 24h: ${pair.get('volume', {}).get('h24')}")
        return pairs[0] if pairs else None
    except Exception as e:
        if debug:
            print(f"❌ fetch_pair_details error: {e}")
        return None


def fetch_token_pairs(chain_id: str, token_address: str, debug: bool = False) -> List[Dict]:
    """GET /token-pairs/v1/{chainId}/{tokenAddress}"""
    try:
        url = f"https://api.dexscreener.com/token-pairs/v1/{chain_id}/{token_address}"
        if debug:
            print(f"🔍 DexScreener Token Pairs Request: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            if debug:
                print(f"📊 DexScreener Token Pairs Response: {len(data)} pairs found")
                if data:
                    print(f"   First pair: {data[0].get('baseToken', {}).get('symbol')}-{data[0].get('quoteToken', {}).get('symbol')} on {data[0].get('chainId')}")
            return data
        return []
    except Exception as e:
        if debug:
            print(f"❌ fetch_token_pairs error: {e}")
        return []


def merge_analytics(primary: Dict, override_pair: Optional[Dict]) -> Dict:
    """Merge analytics dict with an override pair object (prefer override fields when present)."""
    if not override_pair:
        return primary
    o = get_pool_analytics(override_pair)
    merged = primary.copy()
    for k, v in o.items():
        if v not in (None, [], {}):
            merged[k] = v
    return merged


def _normalize_address(addr: Optional[str]) -> Optional[str]:
    return addr.lower() if isinstance(addr, str) else None


def _pick_best_pair(pairs: List[Dict], chain_id: Optional[str] = None,
                    base_addr: Optional[str] = None, quote_addr: Optional[str] = None,
                    base_symbol: Optional[str] = None, quote_symbol: Optional[str] = None, debug: bool = False) -> Optional[Dict]:
    def score(pair: Dict) -> float:
        s = 0.0
        pair_chain = str(pair.get("chainId", "")).lower()
        
        # Chain match
        if chain_id and pair_chain == chain_id:
            s += 10.0
            if debug:
                print(f"   +10.0 for chain match: {pair_chain} == {chain_id}")
        
        # Address match
        b_addr = _normalize_address(pair.get("baseToken", {}).get("address"))
        q_addr = _normalize_address(pair.get("quoteToken", {}).get("address"))
        if base_addr and b_addr == base_addr:
            s += 5.0
            if debug:
                print(f"   +5.0 for base address match")
        if quote_addr and q_addr == quote_addr:
            s += 5.0
            if debug:
                print(f"   +5.0 for quote address match")
        
        # Symbol match (case-insensitive exact)
        b_sym = str(pair.get("baseToken", {}).get("symbol", "")).upper()
        q_sym = str(pair.get("quoteToken", {}).get("symbol", "")).upper()
        if base_symbol and b_sym == base_symbol.upper():
            s += 2.0
            if debug:
                print(f"   +2.0 for base symbol match: {b_sym} == {base_symbol.upper()}")
        if quote_symbol and q_sym == quote_symbol.upper():
            s += 2.0
            if debug:
                print(f"   +2.0 for quote symbol match: {q_sym} == {quote_symbol.upper()}")
        
        # Liquidity/volume add
        liq = pair.get("liquidity", {}).get("usd") or 0
        vol = pair.get("volume", {}).get("h24") or 0
        try:
            liq_score = min(float(liq) / 1e6, 10.0)  # up to +10
            vol_score = min(float(vol) / 1e6, 5.0)   # up to +5
            s += liq_score + vol_score
            if debug:
                print(f"   +{liq_score:.1f} for liquidity, +{vol_score:.1f} for volume")
        except Exception:
            pass
        
        if debug:
            print(f"   Total score: {s:.1f}")
        return s

    if not pairs:
        return None
    
    # Sort by score desc
    if chain_id:
        chain_id = chain_id.lower()
    if base_addr:
        base_addr = _normalize_address(base_addr)
    if quote_addr:
        quote_addr = _normalize_address(quote_addr)
    
    if debug:
        print(f"🔍 Scoring {len(pairs)} pairs for chain '{chain_id}', base '{base_symbol}', quote '{quote_symbol}'")
        for i, pair in enumerate(pairs[:3]):  # Show top 3
            print(f"  {i+1}. {pair.get('baseToken', {}).get('symbol')}-{pair.get('quoteToken', {}).get('symbol')} on {pair.get('chainId')}")
            score_val = score(pair)
            print(f"     Score: {score_val:.1f}")
    
    return sorted(pairs, key=lambda p: score(p), reverse=True)[0]


def find_pair_for_pool(pool: Dict, debug: bool = False) -> Optional[Dict]:
    """Resolve the best DexScreener pair for a DefiLlama pool using addresses, symbols, and chain.
    Returns the selected pair dict or None if not found.
    """
    if not isinstance(pool, dict):
        return None

    chain_llama = pool.get("chain")
    chain_id = map_chain_name_to_dex_id(chain_llama)
    
    if debug:
        print(f"🔍 Pool Chain Mapping: '{chain_llama}' -> '{chain_id}'")

    # Try using underlying token addresses if available
    underlying = pool.get("underlyingTokens") or pool.get("underlying") or []
    if isinstance(underlying, list) and len(underlying) >= 2:
        base_addr = _normalize_address(underlying[0])
        quote_addr = _normalize_address(underlying[1])
        if debug:
            print(f"🔍 Using underlying tokens: {base_addr} / {quote_addr}")
        # Search by each address, then pick best pair that contains both addresses
        pairs_a = search_dexscreener(underlying[0], debug=debug)
        pairs_b = search_dexscreener(underlying[1], debug=debug)
        candidates = []
        addr_set = {base_addr, quote_addr}
        for p in pairs_a + pairs_b:
            b = _normalize_address(p.get("baseToken", {}).get("address"))
            q = _normalize_address(p.get("quoteToken", {}).get("address"))
            if b in addr_set and q in addr_set:
                candidates.append(p)
        if candidates:
            best = _pick_best_pair(candidates, chain_id=chain_id, base_addr=base_addr, quote_addr=quote_addr, debug=debug)
            if debug and best:
                print(f"✅ Found pair via addresses: {best.get('baseToken', {}).get('symbol')}-{best.get('quoteToken', {}).get('symbol')} on {best.get('chainId')}")
            return best

    # Fallback: use symbol tokens
    symbol = str(pool.get("symbol", ""))
    token_a = token_b = None
    if symbol:
        tmp = symbol.replace(" ", "").upper()
        for sep in ["-", "/", "_", " "]:
            if sep in tmp:
                parts = [x for x in tmp.split(sep) if x]
                if len(parts) >= 2:
                    token_a, token_b = parts[0], parts[1]
                    break
        if token_a is None:
            token_a = tmp
        
        if debug:
            print(f"🔍 Using symbol tokens: {token_a} / {token_b}")

    # Try searching with both tokens together first for more specific results
    if token_a and token_b:
        search_query = f"{token_a}-{token_b}"
        if debug:
            print(f"🔍 Trying combined search: '{search_query}'")
        pairs = search_dexscreener(search_query, debug=debug)
        if pairs:
            best = _pick_best_pair(pairs, chain_id=chain_id, base_symbol=token_a, quote_symbol=token_b, debug=debug)
            if best:
                if debug:
                    print(f"✅ Found pair via combined search: {best.get('baseToken', {}).get('symbol')}-{best.get('quoteToken', {}).get('symbol')} on {best.get('chainId')}")
                return best
    
    # Try searching by both tokens and pick the best result
    best_result = None
    best_score = -1
    
    # Try first token
    if token_a:
        if debug:
            print(f"🔍 Trying first token search: '{token_a}'")
        pairs = search_dexscreener(token_a, debug=debug)
        if pairs:
            result = _pick_best_pair(pairs, chain_id=chain_id, base_symbol=token_a, quote_symbol=token_b, debug=debug)
            if result:
                # Calculate a simple score for comparison
                score = 0
                if result.get("chainId", "").lower() == chain_id:
                    score += 10
                if result.get("baseToken", {}).get("symbol", "").upper() == token_a.upper():
                    score += 5
                if result.get("quoteToken", {}).get("symbol", "").upper() == token_b.upper():
                    score += 5
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    if debug:
                        print(f"✅ Found pair via first token: {result.get('baseToken', {}).get('symbol')}-{result.get('quoteToken', {}).get('symbol')} on {result.get('chainId')} (score: {score})")
    
    # Try second token
    if token_b:
        if debug:
            print(f"🔍 Trying second token search: '{token_b}'")
        pairs = search_dexscreener(token_b, debug=debug)
        if pairs:
            result = _pick_best_pair(pairs, chain_id=chain_id, base_symbol=token_a, quote_symbol=token_b, debug=debug)
            if result:
                # Calculate a simple score for comparison
                score = 0
                if result.get("chainId", "").lower() == chain_id:
                    score += 10
                if result.get("baseToken", {}).get("symbol", "").upper() == token_a.upper():
                    score += 5
                if result.get("quoteToken", {}).get("symbol", "").upper() == token_b.upper():
                    score += 5
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    if debug:
                        print(f"✅ Found pair via second token: {result.get('baseToken', {}).get('symbol')}-{result.get('quoteToken', {}).get('symbol')} on {result.get('chainId')} (score: {score})")
    
    if best_result:
        if debug:
            print(f"🏆 Selected best result with score {best_score}")
        return best_result

    # Final fallback: broad search by project or full symbol
    for q in filter(None, [symbol, pool.get("project")]):
        if debug:
            print(f"🔍 Final fallback search: '{q}'")
        pairs = search_dexscreener(q, debug=debug)
        best = _pick_best_pair(pairs, chain_id=chain_id, debug=debug)
        if best:
            if debug:
                print(f"✅ Found pair via fallback: {best.get('baseToken', {}).get('symbol')}-{best.get('quoteToken', {}).get('symbol')} on {best.get('chainId')}")
            return best

    if debug:
        print("❌ No suitable pair found")
    return None


def build_dexscreener_url_from_pair(pair: Dict) -> Optional[str]:
    if not pair:
        return None
    chain_id = pair.get("chainId")
    pair_addr = pair.get("pairAddress")
    if not chain_id or not pair_addr:
        return None
    return f"https://dexscreener.com/{chain_id}/{pair_addr}"


def format_number(value, decimals=2):
    """Format large numbers with K, M, B suffixes"""
    if value is None:
        return "N/A"
    
    try:
        value = float(value)
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:.2f}"
    except:
        return "N/A"

def format_percentage(value):
    """Format percentage values"""
    if value is None:
        return "N/A"
    
    try:
        value = float(value)
        return f"{value:.2f}%"
    except:
        return "N/A" 