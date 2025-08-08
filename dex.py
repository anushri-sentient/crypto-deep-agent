import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import os

# --- Configuration ---
PRO_API_BASE_URL = "https://pro-api.llama.fi"
COINS_API_BASE_URL = "https://coins.llama.fi"
STABLECOINS_API_BASE_URL = "https://stablecoins.llama.fi"
YIELDS_API_BASE_URL = "https://yields.llama.fi"
BRIDGES_API_BASE_URL = "https://bridges.llama.fi"
TVL_API_BASE_URL = "https://api.llama.fi" # For non-PRO TVL endpoints

# Set up Streamlit page
st.set_page_config(
    page_title="DeFi Staking Context Aggregator (Pro)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DeFi Staking Context Aggregator 📊")
st.write("Leveraging DefiLlama Pro API for in-depth insights into staking opportunities.")

# --- API Key Input (Sidebar) ---
st.sidebar.header("API Configuration")

# Try to get API key from secrets.toml first
api_key = os.getenv("LLAMAFI_API_KEY")
print("API Key from environment:", api_key)
st.sidebar.success("API Key loaded from `secrets.toml`!")


# --- DefiLlama API Helper Functions (with Caching) ---

# Caching strategy:
# - Short TTL (5-15 mins) for frequently changing data like current prices, core yield pools.
# - Medium TTL (1-4 hours) for less volatile data like chain TVL, protocol lists.
# - Long TTL (daily/weekly) for static or slowly changing data like hacks, emissions, raises.

@st.cache_data(ttl=60 * 15) # Cache for 15 minutes
def get_api_usage(key: str):
    """Fetches DefiLlama Pro API key usage."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/usage/APIKEY")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching API usage: {e}")
        return {"error": str(e)}

@st.cache_data(ttl=60 * 15) # Cache for 15 minutes
def fetch_yield_pools(key: str):
    """Retrieves the latest data for all yield pools."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/yields/poolsOld") # poolsOld for the pool_old field
        response.raise_for_status()
        data = response.json().get('data', [])
        df = pd.DataFrame(data)
        if not df.empty:
            df['tvlUsd'] = pd.to_numeric(df['tvlUsd'], errors='coerce')
            df['apy'] = pd.to_numeric(df['apy'], errors='coerce')
            df = df.dropna(subset=['tvlUsd', 'apy', 'project', 'chain', 'symbol'])
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching yield pools: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60 * 4) # Cache for 4 hours
def fetch_pool_chart(key: str, pool_id: str):
    """Fetches historical APY and TVL for a specific pool."""
    try:
        response = requests.get(f"{YIELDS_API_BASE_URL}/{key}/chart/{pool_id}")
        response.raise_for_status()
        data = response.json().get('data', [])
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        return df
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching chart for pool {pool_id}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60 * 15) # Cache for 15 minutes
def fetch_token_prices(token_strings: list):
    """Fetches current prices for a list of tokens."""
    if not token_strings:
        return {}
    coins_str = ",".join(token_strings)
    try:
        response = requests.get(f"{COINS_API_BASE_URL}/prices/current/{coins_str}")
        response.raise_for_status()
        return response.json().get('coins', {})
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching token prices: {e}")
        return {}

@st.cache_data(ttl=60 * 60) # Cache for 1 hour
def fetch_stablecoins():
    """Lists all stablecoins with their circulating amounts."""
    try:
        response = requests.get(f"{STABLECOINS_API_BASE_URL}/stablecoins")
        response.raise_for_status()
        return response.json().get('peggedAssets', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stablecoins: {e}")
        return []

@st.cache_data(ttl=60 * 60 * 24) # Cache for 24 hours
def fetch_hacks(key: str):
    """Fetches overview of all hacks."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/api/hacks")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching hacks data: {e}")
        return []

@st.cache_data(ttl=60 * 60 * 24) # Cache for 24 hours
def fetch_emissions(key: str):
    """Lists all tokens along with basic info for each (unlocks)."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/api/emissions")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching emissions data: {e}")
        return []

@st.cache_data(ttl=60 * 60) # Cache for 1 hour
def fetch_active_users(key: str):
    """Fetches active users data for chains and protocols."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/api/activeUsers")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching active users data: {e}")
        return {}

@st.cache_data(ttl=60 * 60) # Cache for 1 hour
def fetch_borrow_pools(key: str):
    """Fetches borrow costs APY of assets from lending markets."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/yields/poolsBorrow")
        response.raise_for_status()
        data = response.json().get('data', [])
        df = pd.DataFrame(data)
        if not df.empty:
            df['tvlUsd'] = pd.to_numeric(df['tvlUsd'], errors='coerce')
            df['apyBaseBorrow'] = pd.to_numeric(df['apyBaseBorrow'], errors='coerce')
            df = df.dropna(subset=['tvlUsd', 'apyBaseBorrow', 'project', 'chain', 'symbol'])
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching borrow pools: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60) # Cache for 1 hour
def fetch_lsd_rates(key: str):
    """Fetches APY rates of multiple LSDs."""
    try:
        response = requests.get(f"{YIELDS_API_BASE_URL}/{key}/lsdRates")
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching LSD rates: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60 * 4) # Cache for 4 hours
def fetch_category_performance(key: str, period: str = "30"):
    """Fetches chart of narratives based on category performance."""
    try:
        response = requests.get(f"{PRO_API_BASE_URL}/{key}/fdv/performance/{period}")
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], unit='s')
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching category performance: {e}")
        return pd.DataFrame()

# --- Main App Logic ---

# Display API Usage first
with st.sidebar:
    st.markdown("---")
    st.subheader("API Key Usage")
    usage_data = get_api_usage(api_key)
    if "error" not in usage_data:
        st.success(f"Credits Left: {usage_data.get('credits', 'N/A')} (Resets 1st of month)")
    else:
        st.error(f"Could not retrieve API usage: {usage_data['error']}. Check your API key.")


# --- Page Navigation ---
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard Overview",
        "DEX & Money Market Staking",
        "PT & YT Token Staking",
        "Stablecoin Peg Analysis",
        "Looping Strategy (Conceptual)",
        "Risk & Tokenomics Context"
    ]
)

if page == "Dashboard Overview":
    st.header("Dashboard Overview")
    st.write("A high-level view of the DeFi landscape and market trends.")

    # 1. Category Performance
    st.subheader("DeFi Category Performance (30 Days)")
    category_df = fetch_category_performance(api_key, period="30")
    if not category_df.empty:
        # Exclude 'date' column for plotting categories
        categories_to_plot = [col for col in category_df.columns if col != 'date']
        # Reshape for Plotly Express
        df_melted = category_df.melt(id_vars=['date'], value_vars=categories_to_plot,
                                     var_name='Category', value_name='Performance')
        fig = px.line(df_melted, x="date", y="Performance", color='Category',
                      title="Category Performance Over Time (Change in FDV)",
                      labels={"Performance": "FDV Change (%)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category performance data available.")


    # 2. General TVL Overview
    st.subheader("Overall DeFi TVL & Top Protocols")

    @st.cache_data(ttl=60 * 60) # Cache for 1 hour
    def fetch_protocols_overview():
        try:
            response = requests.get(f"{TVL_API_BASE_URL}/protocols")
            response.raise_for_status()
            df = pd.DataFrame(response.json())
            if not df.empty:
                df['tvl'] = pd.to_numeric(df['tvl'], errors='coerce')
                df = df.dropna(subset=['tvl'])
                df = df.sort_values('tvl', ascending=False)
            return df
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching protocols overview: {e}")
            return pd.DataFrame()

    protocols_overview_df = fetch_protocols_overview()
    if not protocols_overview_df.empty:
        st.dataframe(
            protocols_overview_df[['name', 'category', 'chains', 'tvl', 'change_1d', 'change_7d']]
            .head(10)
            .style.format({
                'tvl': "${:,.0f}",
                'change_1d': "{:,.2f}%",
                'change_7d': "{:,.2f}%"
            }),
            use_container_width=True
        )
    else:
        st.info("No protocols overview data available.")

elif page == "DEX & Money Market Staking":
    st.header("DEX & Money Market Staking Opportunities")
    st.write("Explore various yield farming and lending/borrowing pools.")

    with st.spinner("Fetching yield pools... This might take a moment if not cached."):
        pools_df = fetch_yield_pools(api_key)

    if not pools_df.empty:
        # Filter options
        st.sidebar.subheader("Filter Staking Pools")
        selected_chains = st.sidebar.multiselect(
            "Filter by Chain:",
            options=pools_df['chain'].unique().tolist(),
            default=None
        )
        selected_projects = st.sidebar.multiselect(
            "Filter by Project:",
            options=pools_df['project'].unique().tolist(),
            default=None
        )
        min_tvl = st.sidebar.slider("Minimum TVL (USD):", min_value=0, max_value=1000000000, value=1000000, step=100000)
        min_apy = st.sidebar.slider("Minimum APY (%):", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

        filtered_pools = pools_df[
            (pools_df['tvlUsd'] >= min_tvl) &
            (pools_df['apy'] >= min_apy)
        ]
        if selected_chains:
            filtered_pools = filtered_pools[filtered_pools['chain'].isin(selected_chains)]
        if selected_projects:
            filtered_pools = filtered_pools[filtered_pools['project'].isin(selected_projects)]

        st.info(f"Displaying {len(filtered_pools)} of {len(pools_df)} available pools.")

        # Display pools table
        st.subheader("Available Staking Pools")
        display_columns = ['project', 'chain', 'symbol', 'poolMeta', 'tvlUsd', 'apy', 'apyBase', 'apyReward', 'ilRisk', 'exposure']
        st.dataframe(
            filtered_pools[display_columns].style.format({
                'tvlUsd': "${:,.0f}",
                'apy': "{:,.2f}%",
                'apyBase': "{:,.2f}%",
                'apyReward': "{:,.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Detailed Pool View
        st.subheader("Detailed Pool Information")
        # Use a selectbox for a more controlled input, pre-filling with filtered pool names
        if not filtered_pools.empty:
            selected_pool_name = st.selectbox(
                "Select a pool for detailed historical data:",
                filtered_pools['project'] + " - " + filtered_pools['symbol'] + " (" + filtered_pools['chain'] + ")",
                index=0 if not filtered_pools.empty else None
            )
            if selected_pool_name:
                selected_row = filtered_pools[
                    (filtered_pools['project'] + " - " + filtered_pools['symbol'] + " (" + filtered_pools['chain'] + ")") == selected_pool_name
                ].iloc[0]
                pool_id = selected_row['pool']

                st.write(f"### Historical Data for {selected_pool_name}")
                st.markdown(f"**URL:** [{selected_row['url']}]({selected_row['url']})")

                chart_df = fetch_pool_chart(api_key, pool_id)

                if not chart_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_apy = px.line(chart_df, x='timestamp', y='apy', title='Historical APY')
                        st.plotly_chart(fig_apy, use_container_width=True)
                    with col2:
                        fig_tvl = px.line(chart_df, x='timestamp', y='tvlUsd', title='Historical TVL (USD)')
                        st.plotly_chart(fig_tvl, use_container_width=True)
                else:
                    st.info("No historical chart data available for this pool.")

                # Contextual info for the selected protocol
                st.write("### Related Protocol Context")
                protocol_slug = selected_row['project'].replace(" ", "-").lower() # Simple slug conversion for some APIs
                col_hacks, col_users = st.columns(2)

                with col_hacks:
                    st.subheader("Hacks History")
                    hacks_data = fetch_hacks(api_key)
                    protocol_hacks = [h for h in hacks_data if protocol_slug.lower() in h.get('name', '').lower()]
                    if protocol_hacks:
                        for hack in protocol_hacks:
                            st.error(f"🚨 **{hack['name']}** - **{hack['amount']:,} USD** lost on {datetime.fromtimestamp(hack['date']).strftime('%Y-%m-%d')} ({hack['classification']})")
                    else:
                        st.success("No known hacks reported for this protocol in DefiLlama's database.")

                with col_users:
                    st.subheader("Active User Metrics (Latest)")
                    active_users_data = fetch_active_users(api_key)
                    # Try to find the protocol by name (simplistic, could be improved with mapping)
                    found_user_data = None
                    for pid, p_data in active_users_data.items():
                        if p_data.get('name', '').lower() == selected_row['project'].lower():
                            found_user_data = p_data
                            break
                    if found_user_data:
                        st.write(f"**Users:** {found_user_data.get('users', {}).get('value', 'N/A'):,.0f}")
                        st.write(f"**Transactions:** {found_user_data.get('txs', {}).get('value', 'N/A')}")
                        st.write(f"**Gas (USD):** ${found_user_data.get('gasUsd', {}).get('value', 'N/A'):,.2f}")
                        st.caption(f"As of: {datetime.fromtimestamp(found_user_data.get('users', {}).get('end', 0)).strftime('%Y-%m-%d %H:%M UTC')}")
                    else:
                        st.info("No active user data available for this protocol.")
        else:
            st.info("No pools match your filters. Adjust the criteria or ensure data is loaded.")
    else:
        st.info("No yield pools data available. Please check your API key or try again later.")

elif page == "PT & YT Token Staking":
    st.header("Principal Token (PT) & Yield Token (YT) Staking")
    st.write("These tokens represent segregated principal and yield components of a yield-bearing asset, primarily from platforms like Pendle.")

    with st.spinner("Fetching yield pools to identify PT/YT tokens..."):
        pools_df = fetch_yield_pools(api_key)

    pt_yt_pools = pools_df[
        pools_df['poolMeta'].astype(str).str.contains("PT|YT", na=False, case=False)
    ]

    if not pt_yt_pools.empty:
        st.info(f"Found {len(pt_yt_pools)} potential PT/YT pools.")
        st.dataframe(
            pt_yt_pools[['project', 'chain', 'symbol', 'poolMeta', 'tvlUsd', 'apy', 'apyBase', 'apyReward', 'url']]
            .style.format({
                'tvlUsd': "${:,.0f}",
                'apy': "{:,.2f}%",
                'apyBase': "{:,.2f}%",
                'apyReward': "{:,.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Understanding PT/YT Tokens")
        st.markdown("""
        *   **Principal Token (PT):** Represents the underlying asset that earns yield. When you hold a PT, you're staking the principal amount, which can be redeemed for the underlying asset at maturity.
        *   **Yield Token (YT):** Represents the future yield of the underlying asset. Holding a YT allows you to earn the yield on a specified principal amount without holding the principal itself.

        **Key Consideration:** PT/YT tokens have a **maturity date**. Their value and yield behavior change significantly as they approach maturity. DefiLlama's `apy` for these tokens often reflects the implied yield to maturity.
        """)
        st.info("For a complete understanding of PT/YT, refer to the documentation of platforms like Pendle Finance.")
    else:
        st.info("No PT/YT token pools found in the DefiLlama data. This might be due to current market conditions, or how `poolMeta` is structured.")


elif page == "Stablecoin Peg Analysis":
    st.header("Stablecoin Peg Analysis")
    st.write("Understand the stability and market capitalization of various stablecoins.")

    with st.spinner("Fetching stablecoin data..."):
        stablecoins_data = fetch_stablecoins()
        if stablecoins_data:
            stablecoins_df = pd.DataFrame(stablecoins_data)
            stablecoins_df['totalCirculatingUSD'] = stablecoins_df['circulating'].apply(lambda x: x.get('peggedUSD', 0) if isinstance(x, dict) else 0)
            stablecoins_df['price'] = pd.to_numeric(stablecoins_df['price'], errors='coerce')

            st.subheader("Current Stablecoin Overview")
            display_cols = ['name', 'symbol', 'pegType', 'pegMechanism', 'totalCirculatingUSD', 'chains', 'price']
            st.dataframe(
                stablecoins_df[display_cols].style.format({
                    'totalCirculatingUSD': "${:,.0f}",
                    'price': "{:,.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Stablecoin Dominance per Chain")
            selected_chain_for_dominance = st.selectbox(
                "Select Chain:",
                options=sorted(list(set(chain for sc in stablecoins_data for chain in sc.get('chains', []))))
            )
            if selected_chain_for_dominance:
                @st.cache_data(ttl=60 * 15)
                def fetch_stablecoin_dominance(key: str, chain: str):
                    try:
                        response = requests.get(f"{PRO_API_BASE_URL}/{key}/stablecoins/stablecoindominance/{chain}")
                        response.raise_for_status()
                        return response.json()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching stablecoin dominance for {chain}: {e}")
                        return []

                dominance_data = fetch_stablecoin_dominance(api_key, selected_chain_for_dominance)
                if dominance_data:
                    dominance_df = pd.DataFrame(dominance_data)
                    dominance_df['date'] = pd.to_datetime(dominance_df['date'], unit='s')
                    dominance_df['totalCirculatingUSD_peggedUSD'] = dominance_df['totalCirculatingUSD'].apply(lambda x: x.get('peggedUSD', 0))
                    
                    st.write(f"**Largest Stablecoin on {selected_chain_for_dominance}:**")
                    if 'greatestMcap' in dominance_df.columns and not dominance_df.empty:
                        latest_dominance = dominance_df.iloc[-1]
                        st.info(f"Symbol: **{latest_dominance['greatestMcap']['symbol']}** (Gecko ID: {latest_dominance['greatestMcap']['gecko_id']}), Market Cap: **${latest_dominance['greatestMcap']['mcap']:,.0f}**")

                    fig_dominance = px.line(dominance_df, x='date', y='totalCirculatingUSD_peggedUSD', title=f'Total Pegged USD Circulating on {selected_chain_for_dominance}')
                    st.plotly_chart(fig_dominance, use_container_width=True)

                else:
                    st.info(f"No stablecoin dominance data available for {selected_chain_for_dominance}.")
            else:
                st.info("Select a chain to view its stablecoin dominance.")
        else:
            st.info("No stablecoin data available.")


elif page == "Looping Strategy (Conceptual)":
    st.header("Looping Strategy (Conceptual Overview)")
    st.write("Looping involves recursively borrowing against supplied collateral to amplify yield. This section outlines the components, but a full calculation requires more specific protocol LTV/liquidation data not available in DefiLlama's generic APIs.")

    st.warning("⚠️ **Disclaimer:** Full looping APY calculation requires precise LTV, liquidation thresholds, and often specific protocol smart contract interactions. This section provides conceptual data for common lending/borrowing assets. Always do your own research (DYOR) and understand the risks.")

    with st.spinner("Fetching lending and borrowing pools..."):
        lend_pools = fetch_yield_pools(api_key)
        borrow_pools = fetch_borrow_pools(api_key)

    if not lend_pools.empty and not borrow_pools.empty:
        common_assets_lend = lend_pools[['symbol', 'chain', 'project', 'apyBase', 'tvlUsd']].copy()
        common_assets_borrow = borrow_pools[['symbol', 'chain', 'project', 'apyBaseBorrow', 'tvlUsd']].copy()

        # Merge based on symbol and chain for simplicity, a real app would use contract addresses
        merged_df = pd.merge(
            common_assets_lend,
            common_assets_borrow,
            on=['symbol', 'chain', 'project'],
            suffixes=('_lend', '_borrow')
        )

        if not merged_df.empty:
            merged_df['apy_spread'] = merged_df['apyBase_lend'] - merged_df['apyBase_borrow']

            st.subheader("Lending & Borrowing Rates Comparison")
            st.write("Assets where both lending and borrowing are available on the same protocol.")
            display_cols = [
                'project', 'chain', 'symbol',
                'apyBase_lend', 'tvlUsd_lend',
                'apyBase_borrow', 'tvlUsd_borrow',
                'apy_spread'
            ]
            st.dataframe(
                merged_df[display_cols].style.format({
                    'apyBase_lend': "{:,.2f}%",
                    'tvlUsd_lend': "${:,.0f}",
                    'apyBase_borrow': "{:,.2f}%",
                    'tvlUsd_borrow': "${:,.0f}",
                    'apy_spread': "{:,.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )
            st.markdown("""
            **How to Interpret:**
            *   `apyBase_lend`: The base APY you earn by supplying the asset.
            *   `apyBase_borrow`: The base APY you pay to borrow the asset.
            *   `apy_spread`: The difference between lending and borrowing rates. A positive spread means you earn more lending than you pay borrowing.
            *   **Looping:** Involves supplying asset A, borrowing asset A (or another asset) against it, and then re-supplying the borrowed asset to earn more yield. This amplifies the `apy_spread`.
            """)
            st.markdown("To calculate actual looping APY, you would need to factor in the **Loan-to-Value (LTV)** and **liquidation thresholds** of the specific lending protocol, which are not directly available from these DefiLlama APIs.")
        else:
            st.info("No common assets found for both lending and borrowing across protocols.")
    else:
        st.info("Could not fetch both lending and borrowing pool data.")

elif page == "Risk & Tokenomics Context":
    st.header("Risk & Tokenomics Context")
    st.write("Understand potential risks and token supply dynamics for better-informed decisions.")

    # 1. Hacks Overview
    st.subheader("DeFi Hacks & Exploits")
    with st.spinner("Fetching hacks data..."):
        hacks_df = pd.DataFrame(fetch_hacks(api_key))
    if not hacks_df.empty:
        hacks_df['date'] = pd.to_datetime(hacks_df['date'], unit='s')
        hacks_df = hacks_df.sort_values('date', ascending=False)
        st.dataframe(
            hacks_df[['date', 'name', 'amount', 'classification', 'technique', 'chain', 'returnedFunds', 'source']]
            .style.format({'amount': "${:,.0f}"}),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Data sourced from DefiLlama's Hacks Dashboard.")
    else:
        st.info("No hacks data available.")

    # 2. Token Unlocks / Emissions
    st.subheader("Upcoming Token Unlocks (Emissions)")
    st.write("Significant token unlocks can increase supply, potentially impacting token prices and the value of yield rewards denominated in those tokens.")
    with st.spinner("Fetching emissions data..."):
        emissions_data = fetch_emissions(api_key)
        emissions_df = pd.DataFrame(emissions_data)

    if not emissions_df.empty:
        emissions_df['nextEventDate'] = emissions_df['nextEvent'].apply(lambda x: datetime.fromtimestamp(x['date']) if isinstance(x, dict) and 'date' in x else None)
        emissions_df['nextEventUnlock'] = emissions_df['nextEvent'].apply(lambda x: x.get('toUnlock', 0) if isinstance(x, dict) else 0)
        emissions_df['mcap'] = pd.to_numeric(emissions_df['mcap'], errors='coerce')
        emissions_df = emissions_df.dropna(subset=['mcap', 'nextEventDate', 'nextEventUnlock'])

        current_time = datetime.now()
        upcoming_unlocks = emissions_df[emissions_df['nextEventDate'] > current_time].sort_values('nextEventDate').head(20)

        if not upcoming_unlocks.empty:
            st.markdown("#### Top 20 Upcoming Unlocks (by Date)")
            st.dataframe(
                upcoming_unlocks[['name', 'symbol', 'nextEventDate', 'nextEventUnlock', 'mcap', 'circSupply', 'maxSupply']]
                .style.format({
                    'nextEventUnlock': "{:,.0f}",
                    'mcap': "${:,.0f}",
                    'circSupply': "{:,.0f}",
                    'maxSupply': "{:,.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )
            st.caption("Note: 'nextEventUnlock' represents the total amount unlocking, not necessarily only the circulating supply.")
        else:
            st.info("No upcoming token unlock events found in the next period.")
    else:
        st.info("No emissions data available.")

    # 3. Liquid Staking Derivative (LSD) Rates
    st.subheader("Liquid Staking Derivative (LSD) Rates")
    st.write("Yields for staked ETH derivatives like stETH, rETH, etc.")
    with st.spinner("Fetching LSD rates..."):
        lsd_df = fetch_lsd_rates(api_key)
    if not lsd_df.empty:
        st.dataframe(
            lsd_df[['name', 'symbol', 'expectedRate', 'marketRate', 'ethPeg', 'fee']]
            .style.format({
                'expectedRate': "{:,.2f}%",
                'marketRate': "{:,.2f}%",
                'ethPeg': "{:,.2f}%",
                'fee': "{:,.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("`ethPeg` indicates deviation from ETH peg.")
    else:
        st.info("No Liquid Staking Derivative (LSD) rates available.")

st.sidebar.markdown("---")
st.sidebar.caption("Data provided by DefiLlama Pro API")
st.sidebar.caption("App built with Streamlit")