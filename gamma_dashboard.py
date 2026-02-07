# =========================================================
# ZETHETA GAMMA SQUEEZE DASHBOARD – UNIVERSAL VERSION
# Works with ALL Streamlit versions
# =========================================================

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


# =========================================================
# PAGE
# =========================================================

st.set_page_config(layout="wide")
st.title("🚀 Gamma Squeeze Intelligence Dashboard")


# =========================================================
# SIDEBAR
# =========================================================

ticker_symbol = st.sidebar.text_input("Ticker", "AAPL")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h","1d"])
expiry_index = st.sidebar.slider("Expiry index", 0, 5, 0)
refresh = st.sidebar.slider("Auto Refresh (sec)", 5, 60, 15)

# ✅ UNIVERSAL REFRESH
st_autorefresh(interval=refresh * 1000, key="refresh")


ticker = yf.Ticker(ticker_symbol)


# =========================================================
# BLACK SCHOLES
# =========================================================

def gamma(S,K,T,r,sigma):
    if T<=0 or sigma<=0:
        return 0
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.pdf(d1)/(S*sigma*np.sqrt(T))


def delta(S,K,T,r,sigma,call=True):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(d1)-1


# =========================================================
# LOAD DATA
# =========================================================

price_df = ticker.history(period="2d", interval=interval)

if price_df.empty:
    st.stop()

price = price_df["Close"].iloc[-1]

expiries = ticker.options
expiry = expiries[min(expiry_index, len(expiries)-1)]
chain = ticker.option_chain(expiry)

calls = chain.calls.copy()
puts = chain.puts.copy()

T = 7/365


# =========================================================
# ADD GREEKS
# =========================================================

def add(df, call=True):
    g, d = [], []
    for _, r in df.iterrows():
        iv = r["impliedVolatility"] or 0.3
        g.append(gamma(price,r["strike"],T,0.01,iv))
        d.append(delta(price,r["strike"],T,0.01,iv,call))
    df["gamma"]=g
    df["delta"]=d
    return df

calls = add(calls,True)
puts  = add(puts,False)

calls["GEX"]=calls["gamma"]*calls["openInterest"]*100*price**2
puts["GEX"]=puts["gamma"]*puts["openInterest"]*100*price**2

calls["FLOW"]=calls["delta"]*calls["openInterest"]*100
puts["FLOW"]=puts["delta"]*puts["openInterest"]*100


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "Overview",
    "OI / Volume",
    "Gamma Exposure",
    "Hedge Flow",
    "Squeeze Score",
    "Backtest"
])


# =========================================================
# OVERVIEW
# =========================================================

with tabs[0]:

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Price",round(price,2))
    c2.metric("High",round(price_df["High"].max(),2))
    c3.metric("Low",round(price_df["Low"].min(),2))
    c4.metric("Volume",int(price_df["Volume"].sum()))

    fig=go.Figure()
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df["Open"],
        high=price_df["High"],
        low=price_df["Low"],
        close=price_df["Close"]
    ))

    st.plotly_chart(fig, width="stretch")


# =========================================================
# OI
# =========================================================

with tabs[1]:

    call_oi=calls.groupby("strike")["openInterest"].sum()
    put_oi=puts.groupby("strike")["openInterest"].sum()

    st.bar_chart(pd.DataFrame({"Call":call_oi,"Put":put_oi}))


# =========================================================
# GEX
# =========================================================

with tabs[2]:

    gex=calls.groupby("strike")["GEX"].sum()-puts.groupby("strike")["GEX"].sum()
    st.bar_chart(gex)


# =========================================================
# FLOW
# =========================================================

with tabs[3]:

    flow=calls.groupby("strike")["FLOW"].sum()+puts.groupby("strike")["FLOW"].sum()
    st.line_chart(flow.cumsum())


# =========================================================
# SCORE
# =========================================================

with tabs[4]:

    oi_ratio=calls["openInterest"].sum()/(puts["openInterest"].sum()+1)
    gex_total=abs(gex.sum())
    volume_spike=price_df["Volume"].iloc[-1]/price_df["Volume"].mean()

    score=min((0.4*oi_ratio+0.4*gex_total/1e9+0.2*volume_spike)/10,1)

    st.metric("Squeeze Probability",f"{round(score*100,2)}%")
    st.progress(score)


# =========================================================
# BACKTEST
# =========================================================

with tabs[5]:

    hist=ticker.history(period="1y")
    hist["ret"]=hist["Close"].pct_change()
    hist["signal"]=(hist["Volume"]>hist["Volume"].rolling(20).mean()*2).astype(int)
    hist["strat"]=hist["ret"]*hist["signal"]

    equity=(1+hist["strat"]).cumprod()
    st.line_chart(equity)
