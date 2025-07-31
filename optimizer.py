# optimizer.py

import streamlit as st
import pandas as pd
import logging
import openai
import re
import plotly.express as px
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# --- Helper functions duplicated here to avoid circular imports ---
# NOTE: These are also in report.py. This is a compromise for the 3-file structure.
def classify_risk(pool):
    apy = pool.get("apy", 0)
    tvl = pool.get("tvlUsd", 0)
    if apy > 20 or tvl < 100_000: return "High"
    # Simplified version for brevity
    return "Medium" if 5 < apy <= 20 else "Low"

def score_pool(pool):
    tvl = pool.get("tvlUsd", 0)
    apy = pool.get("apy", 0)
    risk_level = classify_risk(pool)
    if risk_level == "Low": return tvl * 0.0001 + apy * 10
    if risk_level == "Medium": return tvl * 0.00005 + apy * 20
    return tvl * 0.00001 + apy * 30

# --- Portfolio Generation ---
@st.cache_data(ttl=3600, show_spinner=True)
def generate_llm_portfolio_recommendation(token_symbol, total_value, eligible_pools, risk_preference):
    """Generates a portfolio recommendation using an LLM."""
    if not OPENAI_API_KEY: return None, "OpenAI API key not configured."

    pools_for_llm = sorted(eligible_pools, key=score_pool, reverse=True)[:25]
    pools_str = "\n".join([f"- ID: {p['pool']} | Project: {p['project']} | APY: {p['apy']:.2f}% | TVL: ${p['tvlUsd']/1e6:.1f}M | Risk: {classify_risk(p)}" for p in pools_for_llm])
    
    prompt = f"""
    You are an expert DeFi portfolio manager. Construct a profitable and risk-appropriate yield farming portfolio.

    **User Details:**
    - Amount: ${total_value:,.0f} in {token_symbol.upper()}
    - Risk Preference: {risk_preference}

    **Available Pools:**
    {pools_str}

    **Instructions:**
    1.  Select the best pools from the list using their exact 'ID'.
    2.  Diversify across at least 3 blockchains if possible.
    3.  Provide a "Portfolio Explanation" and then a "Portfolio Allocation" markdown table. The total allocation MUST sum to 100%.

    **Portfolio Explanation:**
    [Your strategic approach here...]

    **Portfolio Allocation:**
    | Pool ID | Project | Symbol | Chain | APY (%) | Risk Level | Allocation (%) |
    |---|---|---|---|---|---|---|
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=1500,
        )
        llm_output = response.choices[0].message.content
        explanation, table_part = llm_output.split("Portfolio Allocation:", 1)
        
        portfolio_allocations = []
        pool_id_map = {p.get("pool"): p for p in eligible_pools}
        for row in table_part.strip().split('\n'):
            if '|' in row and '---' not in row and 'Pool ID' not in row:
                cols = [c.strip() for c in row.strip('|').split('|')]
                if len(cols) == 7:
                    pool_id, alloc_pct_str = cols[0], cols[6]
                    original_pool = pool_id_map.get(pool_id)
                    if original_pool:
                        portfolio_allocations.append({
                            "pool": original_pool,
                            "allocation_percent": float(alloc_pct_str.replace('%', '')),
                            "risk": classify_risk(original_pool),
                            "apy": original_pool.get("apy", 0)
                        })
        return portfolio_allocations, explanation.strip()
    except Exception as e:
        logging.error(f"Failed to generate LLM portfolio: {e}")
        return None, "An unexpected error occurred during portfolio generation."


def create_portfolio_optimizer(token_symbol, pools, token_amount, token_price):
    """Create the portfolio optimization UI, powered by LLM."""
    st.subheader("🎯 LLM-Powered Portfolio Constructor")
    total_value = token_amount * token_price
    if total_value == 0:
        st.warning("Please enter a valid amount and ensure token price is available.")
        return

    risk_preference = st.selectbox("Risk Preference", ["Conservative", "Balanced", "Aggressive"])

    if st.button("🚀 Construct Portfolio"):
        portfolio_allocations, explanation = generate_llm_portfolio_recommendation(
            token_symbol, total_value, pools, risk_preference
        )

        if not portfolio_allocations:
            st.error(f"Failed to generate portfolio: {explanation}")
            return

        # --- Display Portfolio Summary ---
        total_return = sum(item["pool"]["apy"] / 100 * (item["allocation_percent"] / 100) * total_value for item in portfolio_allocations)
        weighted_apy = (total_return / total_value) * 100 if total_value > 0 else 0
        
        st.markdown("### 🧠 LLM's Strategy Explanation")
        st.markdown(explanation)
        
        st.markdown("### 💎 Recommended Portfolio Allocation")
        for item in portfolio_allocations:
            pool = item['pool']
            st.markdown(
                f"- **{pool['project']} - {pool['symbol']}** ({item['allocation_percent']:.1f}%): "
                f"Invest `${total_value * item['allocation_percent']/100:,.0f}` at {pool['apy']:.2f}% APY on {pool['chain']}"
            )

        # --- Display Charts ---
        if len(portfolio_allocations) > 1:
            df = pd.DataFrame([{
                "Project": p["pool"]["project"],
                "APY": p["pool"]["apy"],
                "Allocation": p["allocation_percent"],
                "Risk": p["risk"]
            } for p in portfolio_allocations])
            
            fig = px.bar(df, x="Project", y="Allocation", color="Risk", title="Portfolio Allocation by Project")
            st.plotly_chart(fig, use_container_width=True)