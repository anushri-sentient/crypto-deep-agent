import streamlit as st
import requests
import pandas as pd
import altair as alt
import datetime
import json
from openai import OpenAI

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Llama.fi Protocol Data Viewer")

st.title("📊 Llama.fi Protocol Data Viewer with AI-Powered Insights")

st.markdown("""
This application allows you to fetch and visualize detailed data for specific DeFi protocols directly from the Llama.fi Pro API.
It includes general protocol information, current and historical Total Value Locked (TVL), token distribution, available audit information,
and **AI-generated insights** on the TVL trends using OpenAI.

**You will need:**
1. A Pro API key from [Llama.fi](https://www.llama.fi/docs/api)
2. An OpenAI API key from [OpenAI](https://platform.openai.com/api-keys)

**How to Use:**
1. Enter your API keys below.
2. Enter the Protocol ID (e.g., `aave`, `uniswap-v3`, `makerdao`). You can find protocol IDs on the Llama.fi website by looking at the URL for a protocol's page.
3. Click "Fetch Protocol Data" to load the information and generate AI insights.
""")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    api_key = st.text_input("🔑 Enter your Llama.fi Pro API Key", type="password")
with col2:
    openai_api_key = st.text_input("🤖 Enter your OpenAI API Key", type="password")

protocol_input = st.text_input("🔗 Enter Protocol ID (e.g., 'aave', 'uniswap-v3')", "aave")

def generate_openai_insights(tvl_data, protocol_name, openai_api_key):
    """Generate insights using OpenAI based on TVL data"""
    try:
        client = OpenAI(api_key=openai_api_key)
        
        # Prepare data summary for OpenAI
        if tvl_data.empty:
            return "No TVL data available for analysis."
        
        # Create a summary of the data
        data_summary = {
            "protocol_name": protocol_name,
            "data_points": len(tvl_data),
            "date_range": {
                "start": tvl_data['date'].min().strftime('%Y-%m-%d'),
                "end": tvl_data['date'].max().strftime('%Y-%m-%d')
            },
            "tvl_statistics": {
                "current_tvl": float(tvl_data.iloc[-1]['totalLiquidityUSD']),
                "peak_tvl": float(tvl_data['totalLiquidityUSD'].max()),
                "min_tvl": float(tvl_data['totalLiquidityUSD'].min()),
                "average_tvl": float(tvl_data['totalLiquidityUSD'].mean())
            },
            "chains": list(tvl_data['Chain'].unique()) if 'Chain' in tvl_data.columns else [],
            "volatility": float(tvl_data['totalLiquidityUSD'].std())
        }
        
        # Recent trend analysis (last 30 days)
        recent_data = tvl_data[tvl_data['date'] >= (datetime.datetime.now() - datetime.timedelta(days=30))]
        if len(recent_data) > 1:
            recent_change = ((recent_data.iloc[-1]['totalLiquidityUSD'] - recent_data.iloc[0]['totalLiquidityUSD']) 
                           / recent_data.iloc[0]['totalLiquidityUSD'] * 100)
            data_summary["recent_trend_30d"] = float(recent_change)
        
        prompt = f"""
        As a DeFi analyst, analyze the following TVL (Total Value Locked) data for the protocol {protocol_name} and provide comprehensive insights:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Please provide:
        1. **Executive Summary**: A brief overview of the protocol's TVL performance
        2. **Key Trends**: Major patterns in TVL over time
        3. **Notable Events**: Identify potential significant events based on TVL changes (major increases/decreases)
        4. **Risk Assessment**: Evaluate volatility and stability
        5. **Market Position**: Commentary on TVL levels relative to DeFi market
        6. **Future Outlook**: Potential implications of current trends

        Format your response in clear sections with markdown formatting. Be specific with numbers and dates when possible.
        Focus on actionable insights that would be valuable to investors, researchers, or protocol stakeholders.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert DeFi analyst with deep knowledge of blockchain protocols, TVL metrics, and market dynamics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

# --- Fetch Data Button ---
if st.button("🚀 Fetch Protocol Data & Generate AI Insights"):
    if not api_key:
        st.warning("Please enter your Llama.fi Pro API Key to proceed.")
    elif not openai_api_key:
        st.warning("Please enter your OpenAI API Key to generate AI insights.")
    elif not protocol_input:
        st.warning("Please enter a Protocol ID.")
    else:
        # Construct the API URL
        api_url = f"https://pro-api.llama.fi/{api_key}/api/protocol/{protocol_input.lower()}"

        st.info(f"⏳ Fetching data for protocol: **{protocol_input}**...")

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            if not data:
                st.error(f"🚫 No data found for protocol ID: '{protocol_input}'. Please check the ID or your API key.")
                st.json(response.json())
            else:
                st.success(f"✅ Successfully fetched data for {data.get('name', protocol_input)}!")

                # --- Display Basic Information ---
                st.header(f"✨ {data.get('name', 'N/A')} ({data.get('symbol', 'N/A')}) Overview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Name", data.get('name', 'N/A'))
                with col2:
                    st.metric("Symbol", data.get('symbol', 'N/A'))
                with col3:
                    st.metric("Category", data.get('category', 'N/A'))

                st.write(f"**ID:** `{data.get('id', 'N/A')}`")
                st.write(f"**Chains Supported:** {', '.join(data.get('chains', ['N/A']))}")
                if data.get('url'):
                    st.write(f"**Website:** {data['url']}")
                if data.get('twitter'):
                    st.write(f"**Twitter:** [@{data['twitter']}](https://twitter.com/{data['twitter']})")
                if data.get('description'):
                    st.write(f"**Description:** {data['description']}")
                if data.get('misrepresentedTokens'):
                    st.warning("🚨 **Warning:** This protocol has `misrepresentedTokens` flag set to `True`. Please verify data externally.")

                # --- Display Current Chain TVLs ---
                st.subheader("💰 Current Total Value Locked (TVL) by Chain")
                current_chain_tvls = data.get('currentChainTvls')
                if current_chain_tvls:
                    df_current_tvls = pd.DataFrame(
                        list(current_chain_tvls.items()),
                        columns=['Chain', 'Current TVL (USD)']
                    )
                    df_current_tvls = df_current_tvls.sort_values(by='Current TVL (USD)', ascending=False)
                    
                    total_current_tvl = df_current_tvls['Current TVL (USD)'].sum()
                    st.markdown(f"**Total Current TVL Across All Chains:** {format_large_number(total_current_tvl)}")

                    df_current_tvls['Current TVL (USD)'] = df_current_tvls['Current TVL (USD)'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(df_current_tvls, use_container_width=True, hide_index=True)
                else:
                    st.info("ℹ️ No current chain TVL data available for this protocol.")

                # --- Audit Report Information ---
                st.markdown("---")
                st.subheader("🛡️ Audit Information")

                audits_count = data.get('audits')
                if audits_count:
                    st.write(f"**Number of Audits Reported:** `{audits_count}`")
                else:
                    st.info("No explicit audit count found in the data.")

                audit_note = data.get('audit_note')
                if audit_note:
                    st.info(f"**Audit Note:** {audit_note}")
                else:
                    st.info("No specific audit note provided in the data.")

                audit_links = data.get('audit_links')
                if audit_links:
                    st.markdown("**Audit Links:**")
                    for i, link in enumerate(audit_links):
                        st.markdown(f"- [Audit Report {i+1}]({link})")
                else:
                    st.info("No audit links found in the data.")

                # --- Display Historical Chain TVLs and Generate AI Insights ---
                st.subheader("🕰️ Historical Data by Chain")
                chain_tvls_data = data.get('chainTvls')

                if chain_tvls_data:
                    all_tvl_df = pd.DataFrame()
                    latest_tokens_data = {}

                    for chain, chain_details in chain_tvls_data.items():
                        # Process TVL history
                        tvl_history = chain_details.get('tvl')
                        if tvl_history:
                            df_tvl = pd.DataFrame(tvl_history)
                            df_tvl['date'] = pd.to_datetime(df_tvl['date'], unit='s')
                            df_tvl['Chain'] = chain
                            all_tvl_df = pd.concat([all_tvl_df, df_tvl])

                        # Process Tokens history
                        tokens_history = chain_details.get('tokens')
                        if tokens_history:
                            latest_token_entry = sorted(tokens_history, key=lambda x: x['date'], reverse=True)
                            if latest_token_entry:
                                latest_tokens_data[chain] = latest_token_entry[0].get('tokens', {})

                    # --- Plot Historical TVL ---
                    if not all_tvl_df.empty:
                        st.markdown("---")
                        st.markdown("#### Chart: Historical Total Value Locked (TVL) Trends")
                        
                        # Create an Altair multi-line chart
                        chart = alt.Chart(all_tvl_df).mark_line().encode(
                            x=alt.X('date:T', title='Date'),
                            y=alt.Y('totalLiquidityUSD:Q', title='Total Liquidity (USD)', axis=alt.Axis(format='$.0s')),
                            color='Chain:N',
                            tooltip=[
                                alt.Tooltip('date:T', title='Date'),
                                alt.Tooltip('totalLiquidityUSD:Q', title='TVL', format='$,.2f'),
                                'Chain:N'
                            ]
                        ).properties(
                            title="Total Value Locked (TVL) Over Time by Chain"
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)

                        # --- AI-Generated Insights ---
                        st.markdown("---")
                        st.subheader("🤖 AI-Powered TVL Analysis & Insights")
                        
                        with st.spinner("🧠 Generating AI insights from TVL data..."):
                            ai_insights = generate_openai_insights(
                                all_tvl_df, 
                                data.get('name', protocol_input), 
                                openai_api_key
                            )
                        
                        st.markdown(ai_insights)

                        # --- Enhanced Statistical Summary ---
                        st.markdown("---")
                        st.subheader("📈 Statistical Summary")
                        
                        all_tvl_df_sorted = all_tvl_df.sort_values(by='date').reset_index(drop=True)
                        
                        if len(all_tvl_df_sorted) > 1:
                            # Create metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                current_tvl = all_tvl_df_sorted.iloc[-1]['totalLiquidityUSD']
                                st.metric("Current TVL", format_large_number(current_tvl))
                            
                            with col2:
                                peak_tvl = all_tvl_df_sorted['totalLiquidityUSD'].max()
                                st.metric("Peak TVL", format_large_number(peak_tvl))
                            
                            with col3:
                                avg_tvl = all_tvl_df_sorted['totalLiquidityUSD'].mean()
                                st.metric("Average TVL", format_large_number(avg_tvl))
                            
                            with col4:
                                volatility = all_tvl_df_sorted['totalLiquidityUSD'].std()
                                st.metric("Volatility (σ)", format_large_number(volatility))

                    else:
                        st.info("ℹ️ No historical TVL data available for generating insights.")

                    # --- Display Latest Token Distribution ---
                    st.markdown("---")
                    st.markdown("#### Latest Token Distribution by Chain")
                    if latest_tokens_data:
                        for chain, tokens in latest_tokens_data.items():
                            if tokens:
                                with st.expander(f"Show Latest Tokens for **{chain}**"):
                                    df_tokens = pd.DataFrame(
                                        list(tokens.items()),
                                        columns=['Token', 'Amount']
                                    )
                                    df_tokens = df_tokens.sort_values(by='Amount', ascending=False)
                                    st.dataframe(df_tokens, use_container_width=True, hide_index=True)
                            else:
                                st.write(f"No latest token data for {chain}.")
                    else:
                        st.info("ℹ️ No historical token data available for any chain.")

                else:
                    st.info("ℹ️ No detailed historical chain TVL or token data available for this protocol.")

        # --- Error Handling ---
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                st.error("❌ Authentication Error: Your Llama.fi API key is invalid or unauthorized.")
            elif response.status_code == 404:
                st.error(f"❌ Protocol Not Found: Could not find data for '{protocol_input}'.")
            else:
                st.error(f"❌ HTTP Error: Status Code {response.status_code}")
                st.json(response.json())
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Error: Could not connect to the Llama.fi API.")
        except requests.exceptions.Timeout:
            st.error("❌ Timeout Error: The request timed out.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request Error: {e}")
        except ValueError:
            st.error("❌ Data Processing Error: Could not decode JSON response.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("• Protocol TVL visualization")
st.sidebar.markdown("• AI-powered insights")
st.sidebar.markdown("• Historical trend analysis")
st.sidebar.markdown("• Multi-chain support")
st.sidebar.markdown("• Audit information")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit, Llama.fi API, and OpenAI.")