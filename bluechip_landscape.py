import requests
import logging
import csv
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Constants
PROTOCOL_API = "https://api.llama.fi/protocols"
YIELD_API = "https://yields.llama.fi/pools"

CATEGORIES = {"lending", "dexs", "yield"}
CHAINS = {"Ethereum", "Arbitrum", "Optimism", "Base", "Polygon", "Solana"}
STABLECOINS = {"USDC", "USDT", "DAI", "FRAX", "TUSD", "LUSD", "GUSD", "SUSD", "MIM", "ALUSD"}

ALWAYS_INCLUDE = {"kamino", "marginfi", "raydium", "gamma"}

# --- Fetch Protocol Metadata ---
def fetch_protocols():
    logging.info("Fetching all protocols...")
    try:
        response = requests.get(PROTOCOL_API)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Failed to fetch protocols: {e}")
        return []

def filter_protocols(protocols):
    logging.info("Filtering protocols by category and chain...")
    filtered = []
    for protocol in protocols:
        name = protocol.get("name", "").lower()
        slug = protocol.get("slug", "").lower()
        category = protocol.get("category", "").lower()
        protocol_chains = set(chain.capitalize() for chain in protocol.get("chains", []))
        if category not in CATEGORIES and slug not in ALWAYS_INCLUDE:
            continue
        if not protocol_chains & CHAINS and slug not in ALWAYS_INCLUDE:
            continue
        filtered.append({
            "name": name,
            "slug": slug,
            "category": category,
            "chains": list(protocol_chains & CHAINS),
            "tvl": protocol.get("tvl", 0)
        })
    logging.info(f"{len(filtered)} protocols matched.")
    return filtered

def classify_stablecoin(coin):
    coin = coin.upper()
    if coin in {"USDC", "USDT", "GUSD", "TUSD"}:
        return "Tier 1 - Fully Reserved (T-bill backed)"
    elif coin in {"DAI", "LUSD", "ALUSD"}:
        return "Tier 2 - Over-collateralized"
    elif coin in {"FRAX", "MIM", "SUSD"}:
        return "Tier 3 - Algorithmic/Looped"
    return "Unknown Quality"

def assess_risk(pool):
    meta = pool.get("poolMeta", "") or ""
    meta = meta.lower()
    if "lend" in meta or "borrow" in meta:
        return "Low Risk - Lending Market"
    elif "farm" in meta or "lp" in meta or "vault" in meta or "stake" in meta:
        return "High Risk - LP/Farming/Leveraged"
    return "Medium Risk - Unknown/Other"

def tag_pool(apr, risk):
    if apr is None:
        return "Unknown"
    if apr > 10 and "High Risk" in risk:
        return "High Risk / High Yield"
    elif apr >= 5 and "Low Risk" in risk:
        return "Attractive / Low Risk"
    elif apr >= 3 and "Medium Risk" in risk:
        return "Moderate Risk / Moderate Yield"
    else:
        return "Low Yield / Stable"

# --- Fetch Yield Pools ---
def fetch_yield_pools():
    try:
        response = requests.get(YIELD_API)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        logging.error(f"Failed to fetch yield data: {e}")
        return []

def evaluate_yield_stablecoin_pools(filtered_protocols):
    logging.info("Fetching and evaluating yield pools...")
    pools = fetch_yield_pools()
    results = []
    valid_protocols = {p['name'].lower() for p in filtered_protocols} | ALWAYS_INCLUDE

    for pool in pools:
        protocol = pool.get("project", "unknown").lower()
        if protocol not in valid_protocols:
            continue

        coin_name = pool.get("symbol", "").upper()
        if not any(stable in coin_name for stable in STABLECOINS):
            continue

        apr = pool.get("apy", None)
        if apr is None:
            continue

        stable = next((s for s in STABLECOINS if s in coin_name), "UNKNOWN")
        quality = classify_stablecoin(stable)
        risk = assess_risk(pool)
        tag = tag_pool(apr, risk)
        chain = pool.get("chain", "unknown")
        category = pool.get("poolMeta", "unknown")

        # Log each evaluated pool
        # logging.info(
        #     f"Protocol: {protocol:15} | Coin: {coin_name:20} | Chain: {chain:10} | Stablecoin: {stable:6} | "
        #     f"APR: {apr:.2f}% | Quality: {quality:40} | Risk: {risk:35} | Tag: {tag}"
        # )

        results.append({
            "protocol": protocol,
            "coin_name": coin_name,
            "chain": chain,
            "category": category,
            "stablecoin": stable,
            "apr_%": apr,
            "quality": quality,
            "risk": risk,
            "tag": tag
        })

    logging.info(f"Evaluated {len(results)} stablecoin pools from yield API.")
    return results

# --- Save to CSV ---
def save_to_csv(rows, filename):
    if not rows:
        logging.warning("No data to save.")
        return
    logging.info(f"Saving results to {filename}")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logging.info("Save successful.")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")

def run_bluechip_landscape():
    start = time.time()
    logging.info("Starting bluechip stablecoin yield analysis...")

    protocols = fetch_protocols()
    filtered_protocols = filter_protocols(protocols)

    if filtered_protocols:
        save_to_csv(filtered_protocols, "filtered_protocols.csv")
        print(f"Saved {len(filtered_protocols)} filtered protocols to filtered_protocols.csv")

    evaluated = evaluate_yield_stablecoin_pools(filtered_protocols)

    if evaluated:
        save_to_csv(evaluated, "evaluated_stablecoin_pools.csv")

        risk_order = {
            "Low Risk - Lending Market": 0,
            "Medium Risk - Unknown/Other": 1,
            "High Risk - LP/Farming/Leveraged": 2
        }

        evaluated_sorted = sorted(evaluated, key=lambda x: (risk_order.get(x['risk'], 99), -x['apr_%']))

        print("\nTop Stablecoin Pools by Risk (Low to High) and APR:\n")
        for pool in evaluated_sorted[:15]:
            print(f"{pool['protocol']:15} | {pool['coin_name']:20} | {pool['apr_%']:6.2f}% APR | "
                  f"{pool['quality']:40} | {pool['risk']:35} | Tag: {pool['tag']}")
    else:
        logging.warning("No stablecoin pools evaluated.")

    elapsed = time.time() - start
    logging.info(f"Completed in {elapsed:.2f} seconds")
    return filtered_protocols, evaluated

# --- Main ---
def main():
    run_bluechip_landscape()

if __name__ == "__main__":
    main()
