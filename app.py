"""
Kronos Prediction Web App
INPUT: Ticker symbols via Streamlit UI -> yfinance fetches OHLCVA data
OUTPUT: Plotly charts (1-year % return vs ACWI, candlestick + volume), statistical metrics grid
Last updated: 2026-04-15
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time
import ta
from datetime import datetime, timedelta
import yfinance as yf


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading Kronos model (~30s on first run)...")
def load_model():
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=1024)
    return predictor


# ============================================================
# DATA FETCHING (with retry/fallback)
# ============================================================
def fetch_yfinance_data(ticker: str, period: str = "2y"):
    """Fetch OHLCVA data from yfinance with retry logic.
    Fallback chain: period=2y -> period=1y -> final attempt.
    """
    max_retries = 2
    retry_count = 0
    retry_period = period
    while retry_count < max_retries:
        try:
            df = yf.Ticker(ticker).history(period=retry_period)
            if df.empty:
                raise ValueError("No data returned")
            df.index = df.index.tz_localize(None)
            df = df.rename(columns={
                'Volume': 'volume', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close'
            })
            df['amount'] = df['volume'] * df['open']
            # Ensure all required columns exist
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col not in df.columns:
                    df[col] = 0.0
            return df[['open', 'high', 'low', 'close', 'volume', 'amount']]
        except Exception:
            retry_count += 1
            if retry_count < max_retries and retry_period == period:
                retry_period = "1y"
            if retry_count >= max_retries:
                raise
            time.sleep(3)
    return None


def get_business_dates(start_date, n_days):
    """Generate next n business days from start_date."""
    dates = []
    current = start_date if isinstance(start_date, pd.Timestamp) else pd.Timestamp(start_date)
    while len(dates) < n_days:
        current = current + pd.Timedelta(days=1)
        if current.dayofweek < 5:
            dates.append(current)
    return pd.DatetimeIndex(dates)


# ============================================================
# PREDICTION ENGINE
# ============================================================
@st.cache_resource(show_spinner="Running ACWI benchmark prediction (~30s, cached globally)...")
def predict_acwi():
    """Run Kronos prediction on ACWI once, cache globally."""
    predictor = load_model()
    df = fetch_yfinance_data("ACWI", "2y").tail(40)
    if len(df) < 40:
        return None
    lookback = df.tail(40)
    y_timestamp = get_business_dates(lookback.index[-1] + pd.Timedelta(days=1), 20)
    pred_df = predictor.predict(
        df=lookback, x_timestamp=lookback.index, y_timestamp=y_timestamp,
        pred_len=20, T=0.8, top_p=0.9, sample_count=10
    )
    return pred_df


def predict_ticker(predictor, ticker, lookback_days=40, pred_len=20):
    """Run prediction for a single ticker."""
    df = fetch_yfinance_data(ticker, "2y")
    if df is None:
        raise ValueError(f"No data found for {ticker}")
    if len(df) < lookback_days:
        raise ValueError(f"{ticker}: only {len(df)} days available, need {lookback_days}+")
    hist_df = df.tail(lookback_days)
    y_timestamp = get_business_dates(hist_df.index[-1] + pd.Timedelta(days=1), pred_len)
    pred_df = predictor.predict(
        df=hist_df, x_timestamp=hist_df.index, y_timestamp=y_timestamp,
        pred_len=pred_len, T=0.8, top_p=0.9, sample_count=10
    )
    return pred_df, hist_df


# ============================================================
# TECHNICAL STATS
# ============================================================
def compute_stats(df_hist: pd.DataFrame, pred_df: pd.DataFrame, df_full_year: pd.DataFrame = None):
    """Compute technical indicators from historical + predicted data."""
    # Combine historical + predicted for indicators that need full context
    df = pd.concat([df_hist[['open', 'high', 'low', 'close', 'volume']],
                    pred_df[['open', 'high', 'low', 'close', 'volume']]])
    closes = df['close']

    # Use full year data for 52-week stats if available
    source_for_52w = df_full_year if df_full_year is not None else df_hist

    stats = {}
    # Key prices
    stats['Current Price'] = f"${df_hist['close'].iloc[-1]:.2f}"
    stats['Day Change %'] = (
        f"{((df_hist['close'].iloc[-1] - df_hist['close'].iloc[-2]) / df_hist['close'].iloc[-2] * 100):+.2f}%"
        if len(df_hist) >= 2 else "N/A"
    )

    # 52-week high/low
    if len(source_for_52w) >= 252:
        year_high = source_for_52w['high'].tail(252).max()
        year_low = source_for_52w['low'].tail(252).min()
    else:
        year_high = source_for_52w['high'].max()
        year_low = source_for_52w['low'].min()
    stats['52-Week High'] = f"${year_high:.2f}"
    stats['52-Week Low'] = f"${year_low:.2f}"

    # SMAs (on full combined data)
    stats['SMA 20'] = f"${closes.tail(20).mean():.2f}"
    stats['SMA 50'] = f"${closes.tail(50).mean():.2f}" if len(closes) >= 50 else "N/A"
    stats['SMA 200'] = f"${closes.mean():.2f}" if len(closes) >= 200 else f"N/A ({len(closes)} days)"

    # EMAs
    stats['EMA 12'] = f"${closes.ewm(span=12, adjust=False).mean().iloc[-1]:.2f}"
    stats['EMA 26'] = f"${closes.ewm(span=26, adjust=False).mean().iloc[-1]:.2f}"

    # RSI
    rsi_series = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    stats['RSI (14)'] = f"{rsi_series.iloc[-1]:.1f}"
    if rsi_series.iloc[-1] > 70:
        stats['RSI Signal'] = "Overbought (>70)"
    elif rsi_series.iloc[-1] < 30:
        stats['RSI Signal'] = "Oversold (<30)"
    else:
        stats['RSI Signal'] = "Neutral"

    # MACD
    macd = ta.trend.MACD(closes, window_slow=26, window_fast=12, window_sign=9)
    stats['MACD'] = f"{macd.macd().iloc[-1]:.2f}"
    stats['MACD Signal'] = f"{macd.macd_signal().iloc[-1]:.2f}"
    stats['MACD Histogram'] = f"{macd.macd_diff().iloc[-1]:.2f}"

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(closes, window=20, window_dev=2)
    stats['BB Upper'] = f"${bb.bollinger_hband().iloc[-1]:.2f}"
    stats['BB Middle'] = f"${bb.bollinger_mavg().iloc[-1]:.2f}"
    stats['BB Lower'] = f"${bb.bollinger_lband().iloc[-1]:.2f}"

    # ATR
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    stats['ATR (14)'] = f"{atr.average_true_range().iloc[-1]:.2f}"

    # Volume
    stats['Volume Today'] = f"{df_hist['volume'].iloc[-1]:,.0f}"
    stats['Avg Volume (20d)'] = f"{df['volume'].tail(20).mean():,.0f}"

    # Predicted return (entry = last day open, exit = last predicted close)
    pred_close = pred_df['close'].iloc[-1]
    entry = df_hist['open'].iloc[-1]
    pred_return = (pred_close / entry - 1) * 100
    stats['Predicted Return (20d)'] = f"{pred_return:+.2f}%"

    return stats


# ============================================================
# PLOTLY CHARTS
# ============================================================
def build_return_chart(df_hist_full, pred_df, acwi_hist, acwi_pred, ticker_name):
    """1-Year % Return chart — clean TradingView style.
    Single continuous line with predicted continuation. Minimal axes."""

    # Historical returns normalized to 0% at start
    asset_ret = df_hist_full['close'] / df_hist_full['close'].iloc[0] - 1
    acwi_ret = acwi_hist['close'] / acwi_hist['close'].iloc[0] - 1

    # Predicted returns on same 1Y scale: (pred_close / year_start) - 1
    base_asset = df_hist_full['close'].iloc[0]
    base_acwi = acwi_hist['close'].iloc[0]
    pred_ret = pred_df['close'] / base_asset - 1
    acwi_pred_ret = acwi_pred['close'] / base_acwi - 1

    # Build continuous data: append prediction to last historical point
    today = asset_ret.index[-1]
    acwi_today = acwi_ret.iloc[-1]

    asset_y = np.concatenate([[asset_ret.values[-1]], pred_ret.values]) * 100
    asset_x = pd.DatetimeIndex([today]).append(pred_ret.index)
    acwi_y = np.concatenate([[acwi_today], acwi_pred_ret.values]) * 100
    acwi_x = pd.DatetimeIndex([today]).append(acwi_pred_ret.index)

    fig = go.Figure()

    # Historical — muted gray/blue
    fig.add_trace(go.Scatter(
        x=asset_ret.index, y=asset_ret.values * 100, name=ticker_name,
        mode='lines', line=dict(color='#22c55e', width=2.5),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=acwi_ret.index, y=acwi_ret.values * 100, name='ACWI',
        mode='lines', line=dict(color='rgba(128,128,128,0.6)', width=1.5),
        showlegend=False
    ))

    # Predicted continuation — brighter green for asset
    fig.add_trace(go.Scatter(
        x=asset_x, y=asset_y, name=f'{ticker_name} Predicted',
        mode='lines', line=dict(color='#16a34a', width=2.5, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=acwi_x, y=acwi_y, name='ACWI Predicted',
        mode='lines', line=dict(color='rgba(128,128,128,0.4)', width=1.5, dash='dot'),
        showlegend=False
    ))

    # 0% reference line
    y_min = min(asset_ret.min(), acwi_ret.min()) * 100
    y_max = max(
        max(asset_ret.max(), acwi_ret.max()) * 100,
        max(asset_y.max(), acwi_y.max())
    )
    fig.add_shape(type='line', x0=asset_ret.index[0], y0=0,
                  x1=asset_x[-1], y1=0,
                  line=dict(color='rgba(128,128,128,0.25)', dash='dash', width=1))

    # Today marker
    fig.add_vline(x=today, line_dash='dot', line_color='rgba(128,128,128,0.3)', line_width=1)
    fig.add_annotation(
        x=today, y=0, text='Today',
        showarrow=False, font=dict(size=10, color='rgb(150,150,150)'),
        xanchor='center', yanchor='bottom', yshift=5
    )

    # Clean layout
    fig.update_layout(
        title=dict(text=f"{ticker_name} vs ACWI", font=dict(size=16)),
        plot_bgcolor='white', margin=dict(l=50, r=30, t=50, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title='')
    fig.update_yaxes(showgrid=False, zeroline=False, title='Return %')

    return fig


def build_candlestick_chart(df_hist, pred_df, ticker_name):
    """Candlestick chart with volume subplot. Predicted candles at half opacity."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.75, 0.25]
    )

    # Historical candles
    fig.add_trace(go.Candlestick(
        x=df_hist.index, open=df_hist['open'], high=df_hist['high'],
        low=df_hist['low'], close=df_hist['close'],
        name="Historical", increasing_line_color='green', decreasing_line_color='red'
    ), row=1, col=1)

    # Predicted candles (lighter)
    fig.add_trace(go.Candlestick(
        x=pred_df.index, open=pred_df['open'], high=pred_df['high'],
        low=pred_df['low'], close=pred_df['close'],
        name="Predicted", opacity=0.5,
        increasing_line_color='#1f77b4', decreasing_line_color='#ff7f0e'
    ), row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=df_hist.index, y=df_hist['volume'], name='Volume',
        marker_color='gray', opacity=0.4
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=pred_df.index, y=pred_df['volume'], name='Pred Volume',
        marker_color='#1f77b4', opacity=0.3
    ), row=2, col=1)

    # Separator line
    today = df_hist.index[-1]
    fig.add_vline(x=today, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)

    fig.update_layout(
        title=f"{ticker_name} — Candlestick + Volume",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(
        page_title="Kronos Stock Predictor", page_icon="📈", layout="wide"
    )

    st.title("Kronos Stock Prediction")
    st.caption("AI-powered prediction with ACWI benchmark comparison")

    # Disclaimer
    st.info("⚠️ Predictions generated using a general time-series model. Not financial advice.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        lookback = st.slider("Lookback Days", min_value=20, max_value=100, value=40, step=5)
        pred_len = st.slider("Prediction Days", min_value=5, max_value=30, value=20, step=5)
        st.divider()
        if st.button("Clear ACWI Cache", help="Re-run ACWI prediction on next query"):
            # Clear only the ACWI prediction cache, preserve the loaded model
            st.cache_resource.clear()
            st.success("Cache cleared. ACWI will be re-predicted next run.")

    # Input
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL").strip().upper()

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return

        try:
            # Fetch full year data for stats (does this first so we can show it if model fails)
            with st.spinner(f"Fetching {ticker} data..."):
                df_full_year = fetch_yfinance_data(ticker, "1y")
                if df_full_year is None or df_full_year.empty:
                    st.error(f"No data found for '{ticker}'. Check the ticker symbol.")
                    return

            # Load model
            with st.spinner("Loading Kronos model..."):
                predictor = load_model()

            # Fetch ACWI benchmark data (cached globally)
            with st.spinner("Fetching ACWI benchmark..."):
                try:
                    acwi_pred = predict_acwi()
                    acwi_hist_full = fetch_yfinance_data("ACWI", "1y")
                except Exception:
                    acwi_pred = None
                    acwi_hist_full = None

            # Run prediction
            with st.spinner(f"Predicting {ticker} ({lookback}d lookback → {pred_len}d forecast)..."):
                hist_df = df_full_year.tail(lookback)
                if len(hist_df) < lookback:
                    st.error(f"Insufficient data: {len(df_full_year)} days available, need {lookback}+")
                    return

                y_timestamp = get_business_dates(
                    hist_df.index[-1] + pd.Timedelta(days=1), pred_len
                )
                pred_df = predictor.predict(
                    df=hist_df, x_timestamp=hist_df.index, y_timestamp=y_timestamp,
                    pred_len=pred_len, T=0.8, top_p=0.9, sample_count=10
                )

            # Compute stats
            stats = compute_stats(hist_df, pred_df, df_full_year)

            # Display results
            st.divider()
            st.subheader(f"Prediction for {ticker}")

            # Chart 1: 1-Year % Return
            st.plotly_chart(
                build_return_chart(df_full_year, pred_df, acwi_hist_full, acwi_pred, ticker),
                use_container_width=True
            )

            # Chart 2: Candlestick
            st.plotly_chart(
                build_candlestick_chart(hist_df, pred_df, ticker),
                use_container_width=True
            )

            # Stats grid
            st.subheader("Technical Statistics")
            cols = st.columns(4)
            for i, (key, val) in enumerate(stats.items()):
                cols[i % 4].metric(key, val)

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
