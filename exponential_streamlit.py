import streamlit as st
import pandas as pd
from exponential_scraper import scrape_and_match_exponential_to_defillama

st.title("Exponential.fi Pools: Risk A & B, Sorted by APY")

num_pools = st.selectbox("How many top pools to display?", [10, 25, 50, 100], index=2)

with st.spinner("Scraping Exponential.fi pools and matching to DeFi Llama..."):
    df = scrape_and_match_exponential_to_defillama()

if df is not None and not df.empty:
    st.write(f"Showing top {num_pools} pools by APY (with DeFi Llama match):")
    st.dataframe(df.head(num_pools)[['Pool', 'Chain', 'TVL', 'Risk', 'Yield', 'APY', 'DeFiLlama Project', 'DeFiLlama TVL', 'DeFiLlama APY']])
else:
    st.warning("No pools found or failed to scrape data.") 