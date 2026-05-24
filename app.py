import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import yfinance as yf
import re

st.set_page_config(page_title="Vic's Options Dashboard", layout="wide")
st.markdown("""
    <style>
    * { font-size: 20px !important; }
    h1 { font-size: 48px !important; }
    h2 { font-size: 36px !important; }
    h3 { font-size: 28px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Vic Gnozzo's  --  5-24-26 151pm My Options Trades Visualizer")
st.caption(f"📅 Today: **{datetime.now().strftime('%B %d, %Y')}**")

# Full Black-Scholes with Greeks
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return {'price': intrinsic, 'delta': 1 if option_type == 'call' else -1, 
                'gamma': 0, 'vega': 0, 'theta': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

def get_underlying_info(symbol_group):
    sym = str(symbol_group).upper()
    if 'XSP' in sym:
        return '^XSP', 747
    return '^GSPC', 7473

# Upload
uploaded_file = st.file_uploader("Upload your Fidelity Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Loaded {len(df)} trades!")
    st.dataframe(df)

    today = datetime.now().date()
    selected_date = st.date_input("Select Date for Market Price", today)

    df['Symbol_Group'] = df['Symbol'].astype(str).str.upper().str[:4]
    symbol_groups = sorted(df['Symbol_Group'].unique())

    results = []
    group_settings = {}

    # ================== PER SYMBOL SECTION ==================
    for group in symbol_groups:
        _, default_price = get_underlying_info(group)
        
        st.subheader(f"🔹 {group} Position")
        
        # Controls right above the chart
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            spot = st.slider(f"{group} Price", 500 if 'XSP' in group else 5000, 
                           800 if 'XSP' in group else 8000, default_price, step=5, key=f"spot_{group}")
        with col2:
            dte = st.slider(f"{group} DTE", 1, 120, 13, step=1, key=f"dte_{group}")
        with col3:
            vol = st.slider(f"{group} IV (%)", 8, 50, 18, step=1, key=f"vol_{group}") / 100.0
        with col4:
            rate = st.slider(f"{group} Risk Free (%)", 0.0, 6.0, 4.2, step=0.1, key=f"rate_{group}") / 100.0
        
        group_settings[group] = {'spot': spot, 'dte': dte, 'vol': vol, 'rate': rate}

        # Calculate for this group
        group_df = df[df['Symbol_Group'] == group]
        group_results = []
        
        for _, row in group_df.iterrows():
            qty = float(row.get('Quantity', 1))
            trade_price = float(row.get('Trade Price', row.get('Price', 0)))
            desc = str(row.get('Description', ''))
            
            strike_match = re.search(r'(\d{4,5})', desc)
            strike = int(strike_match.group(1)) if strike_match else 5800
            
            T = dte / 365.0
            greeks = black_scholes_greeks(spot, strike, T, rate, vol, 'call' if 'Call' in desc.upper() else 'put')
            
            theo_price = greeks['price']
            current_pnl = (theo_price - trade_price) * qty * 100
            
            group_results.append({
                'Account': str(row.get('Account', 'Unknown')),
                'Symbol_Group': group,
                'Description': desc[:50],
                'Strike': strike,
                'Qty': qty,
                'Trade_Price': trade_price,
                'Theo_Price': round(theo_price, 2),
                'PnL_$': round(current_pnl, 2),
                'Delta': round(greeks['delta'], 3),
                'Gamma': round(greeks['gamma'], 4),
                'Vega': round(greeks['vega'], 3),
                'Theta': round(greeks['theta'], 3)
            })
        
        group_result_df = pd.DataFrame(group_results)
        results.extend(group_results)
        
        # Show Greeks + Current P&L for this group
        st.dataframe(group_result_df.style.format({
            "PnL_$": "${:.0f}", "Theo_Price": "{:.2f}", 
            "Delta": "{:.3f}", "Gamma": "{:.4f}", "Vega": "{:.3f}", "Theta": "{:.3f}"
        }), use_container_width=True)

        # Payoff Diagram with current P&L line
        st.write(f"**{group} Payoff Diagram**")
        price_range = np.linspace(spot - 150 if 'XSP' in group else spot - 450, 
                                  spot + 150 if 'XSP' in group else spot + 450, 300)
        total_payoff = np.zeros_like(price_range, dtype=float)
        
        for _, row in group_result_df.iterrows():
            K = row['Strike']
            qty = row['Qty']
            entry = row['Trade_Price']
            is_call = 'Call' in str(row['Description']).upper()
            payoff = np.maximum(price_range - K, 0) if is_call else np.maximum(K - price_range, 0)
            total_payoff += (payoff * qty * 100) - (entry * qty * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range, y=total_payoff, mode='lines', 
                                name='Payoff at Expiration', line=dict(color='lime', width=4)))
        
        # Horizontal line for current theoretical P&L
        current_group_pnl = group_result_df['PnL_$'].sum()
        fig.add_hline(y=current_group_pnl, line_dash="dash", line_color="orange", 
                      annotation_text=f"Current P&L: ${current_group_pnl:,.0f}")
        
        fig.add_vline(x=spot, line_dash="dash", line_color="red", annotation_text=f"Current: {spot}")
        
        fig.update_layout(
            title=f"{group} Payoff (DTE: {dte} days)",
            xaxis_title="Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            height=520,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Final Summary
    result_df = pd.DataFrame(results)
    st.subheader("📊 Net P&L by Account")
    account_pnl = result_df.groupby('Account')['PnL_$'].sum().reset_index()
    account_pnl = account_pnl.rename(columns={'PnL_$': 'Net_PnL'})
    st.dataframe(account_pnl.style.format({"Net_PnL": "${:.0f}"}), use_container_width=True)