import os
from bluechip_landscape import run_bluechip_landscape
from youtube_scraper import run_youtube_scraper
from validate_watchlist_llama import run_watchlist_validation

def main():
    print("\n=== STEP 1: Mapping the Blue-Chip Landscape ===\n")
    filtered_protocols, evaluated_pools = run_bluechip_landscape()
    print(f"\n[Step 1] Filtered {len(filtered_protocols)} protocols, evaluated {len(evaluated_pools)} stablecoin pools.")

    print("\n=== STEP 2: Scoring Pools by Reward vs. Risk ===\n")
    if evaluated_pools:
        print("Top 5 scored pools:")
        for pool in sorted(evaluated_pools, key=lambda x: -x['apr_%'])[:5]:
            print(f"{pool['protocol']} | {pool['coin_name']} | {pool['apr_%']:.2f}% | {pool['quality']} | {pool['risk']} | {pool['tag']}")
    else:
        print("No pools evaluated.")

    print("\n=== STEP 3: Hunting for Fresh Ideas on Social Channels ===\n")
    yt_results, yt_analyses = run_youtube_scraper()
    print(f"\n[Step 3] Scraped {len(yt_results)} YouTube videos, {len(yt_analyses)} Gemini analyses.")
    # Optionally, extract new protocol/strategy names from analyses for validation

    print("\n=== STEP 4: Validating Every Newcomer ===\n")
    watchlist, validation_output = run_watchlist_validation()
    print(f"\n[Step 4] Watchlist contains {len(watchlist)} protocols. Validation output written to watchlist_llama_results.txt.")

    # Final summary
    print("\n=== PIPELINE COMPLETE ===\n")
    print(f"Protocols considered: {len(filtered_protocols)}")
    print(f"Pools scored: {len(evaluated_pools)}")
    # print(f"YouTube videos analyzed: {len(yt_results)}")
    # print(f"Watchlist protocols validated: {len(watchlist)}")
    print("\nSee output CSVs and text files for details.")

if __name__ == "__main__":
    main() 