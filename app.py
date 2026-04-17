"""
Kronos Prediction Web App
INPUT: Ticker symbols via Streamlit UI -> yfinance fetches OHLCVA data
OUTPUT: Plotly charts (1-year % return vs benchmark ETF, candlestick OHLC), statistical metrics grid
Last updated: 2026-04-15
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time
import ta
from datetime import datetime, timedelta
import torch
import yfinance as yf

# Benchmark ETFs for return comparison (yfinance symbols)
BENCHMARK_CHOICES = ("ACWI", "SPY", "CQQQ")
BENCHMARK_LABELS = {
    "ACWI": "ACWI (MSCI ACWI)",
    "SPY": "SPY (S&P 500)",
    "CQQQ": "CQQQ (China tech)",
}

# Kronos sampling — temperature and fixed RNG for reproducible forecasts
KRONOS_TEMPERATURE = 0.6
KRONOS_TOP_P = 0.9
KRONOS_SAMPLE_COUNT = 10
KRONOS_RANDOM_SEED = 42


def set_inference_seed(seed: int = KRONOS_RANDOM_SEED) -> None:
    """Reset RNGs before predict so torch.multinomial sampling is reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass


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
@st.cache_resource(show_spinner="Running benchmark prediction (~30s, cached per benchmark)...")
def predict_benchmark(benchmark_ticker: str, lookback_days: int, pred_len: int):
    """Run Kronos on the chosen benchmark; cache key includes ticker and horizons."""
    predictor = load_model()
    df = fetch_yfinance_data(benchmark_ticker, "2y")
    if df is None or len(df) < lookback_days:
        return None
    lookback = df.tail(lookback_days)
    y_timestamp = get_business_dates(
        lookback.index[-1] + pd.Timedelta(days=1), pred_len
    )
    set_inference_seed(KRONOS_RANDOM_SEED)
    pred_df = predictor.predict(
        df=lookback, x_timestamp=lookback.index, y_timestamp=y_timestamp,
        pred_len=pred_len, T=KRONOS_TEMPERATURE, top_p=KRONOS_TOP_P,
        sample_count=KRONOS_SAMPLE_COUNT,
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
    set_inference_seed(KRONOS_RANDOM_SEED)
    pred_df = predictor.predict(
        df=hist_df, x_timestamp=hist_df.index, y_timestamp=y_timestamp,
        pred_len=pred_len, T=KRONOS_TEMPERATURE, top_p=KRONOS_TOP_P,
        sample_count=KRONOS_SAMPLE_COUNT,
    )
    return pred_df, hist_df


# ============================================================
# TECHNICAL STATS
# ============================================================
def compute_stats(df_hist: pd.DataFrame, pred_df: pd.DataFrame, df_full_year: pd.DataFrame = None):
    """Compute technical indicators, returned as a grouped dict for Bento Grid.

    Returns dict of { "Section": { "Label": {"value": str, "hero": bool}, ... }, ... }
    """
    df = pd.concat([df_hist[['open', 'high', 'low', 'close', 'volume']],
                    pred_df[['open', 'high', 'low', 'close', 'volume']]])
    closes = df['close']
    source_for_52w = df_full_year if df_full_year is not None else df_hist

    # 52-week range
    if len(source_for_52w) >= 252:
        year_high = source_for_52w['high'].tail(252).max()
        year_low = source_for_52w['low'].tail(252).min()
    else:
        year_high = source_for_52w['high'].max()
        year_low = source_for_52w['low'].min()

    day_chg = (
        f"{((df_hist['close'].iloc[-1] - df_hist['close'].iloc[-2]) / df_hist['close'].iloc[-2] * 100):+.2f}%"
        if len(df_hist) >= 2 else "N/A"
    )

    # RSI
    rsi_val = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
    if rsi_val > 70:
        rsi_signal = "Overbought (>70)"
    elif rsi_val < 30:
        rsi_signal = "Oversold (<30)"
    else:
        rsi_signal = "Neutral"

    # MACD
    macd = ta.trend.MACD(closes, window_slow=26, window_fast=12, window_sign=9)

    # Bollinger
    bb = ta.volatility.BollingerBands(closes, window=20, window_dev=2)

    # ATR
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)

    # Predicted return
    pred_return = (pred_df['close'].iloc[-1] / df_hist['open'].iloc[-1] - 1) * 100

    from collections import OrderedDict
    grouped = OrderedDict()

    grouped["Price"] = OrderedDict([
        ("Current Price", {"value": f"${df_hist['close'].iloc[-1]:.2f}", "hero": True}),
        ("Day Change", {"value": day_chg, "hero": False}),
        ("52-Week High", {"value": f"${year_high:.2f}", "hero": False}),
        ("52-Week Low", {"value": f"${year_low:.2f}", "hero": False}),
    ])

    grouped["Trend"] = OrderedDict([
        ("SMA 20", {"value": f"${closes.tail(20).mean():.2f}", "hero": False}),
        ("SMA 50", {"value": f"${closes.tail(50).mean():.2f}" if len(closes) >= 50 else "N/A", "hero": False}),
        ("SMA 200", {"value": f"${closes.mean():.2f}" if len(closes) >= 200 else f"N/A ({len(closes)}d)", "hero": False}),
        ("EMA 12", {"value": f"${closes.ewm(span=12, adjust=False).mean().iloc[-1]:.2f}", "hero": False}),
        ("EMA 26", {"value": f"${closes.ewm(span=26, adjust=False).mean().iloc[-1]:.2f}", "hero": False}),
    ])

    grouped["Momentum"] = OrderedDict([
        ("RSI (14)", {"value": f"{rsi_val:.1f}", "hero": False}),
        ("RSI Signal", {"value": rsi_signal, "hero": False}),
        ("MACD", {"value": f"{macd.macd().iloc[-1]:.2f}", "hero": False}),
        ("MACD Signal", {"value": f"{macd.macd_signal().iloc[-1]:.2f}", "hero": False}),
    ])

    grouped["Volatility"] = OrderedDict([
        ("BB Upper", {"value": f"${bb.bollinger_hband().iloc[-1]:.2f}", "hero": False}),
        ("BB Middle", {"value": f"${bb.bollinger_mavg().iloc[-1]:.2f}", "hero": False}),
        ("BB Lower", {"value": f"${bb.bollinger_lband().iloc[-1]:.2f}", "hero": False}),
        ("ATR (14)", {"value": f"{atr.average_true_range().iloc[-1]:.2f}", "hero": False}),
    ])

    grouped["Volume"] = OrderedDict([
        ("Volume Today", {"value": f"{df_hist['volume'].iloc[-1]:,.0f}", "hero": False}),
        ("Avg Volume (20d)", {"value": f"{df['volume'].tail(20).mean():,.0f}", "hero": False}),
    ])

    grouped["Forecast"] = OrderedDict([
        ("Predicted Return", {"value": f"{pred_return:+.2f}%", "hero": True}),
    ])

    return grouped


# ============================================================
# PLOTLY CHARTS
# ============================================================
def build_return_chart(
    df_hist_full, pred_df, bench_hist, bench_pred, ticker_name, benchmark_label: str
):
    """1-Year % Return chart — clean TradingView style.
    Single continuous line with predicted continuation. Minimal axes."""

    # Historical returns normalized to 0% at start
    asset_ret = df_hist_full['close'] / df_hist_full['close'].iloc[0] - 1
    bench_ret = bench_hist['close'] / bench_hist['close'].iloc[0] - 1

    # Predicted returns on same 1Y scale: (pred_close / year_start) - 1
    base_asset = df_hist_full['close'].iloc[0]
    base_bench = bench_hist['close'].iloc[0]
    pred_ret = pred_df['close'] / base_asset - 1
    bench_pred_ret = bench_pred['close'] / base_bench - 1

    # Build continuous data: append prediction to last historical point
    today = asset_ret.index[-1]
    bench_today = bench_ret.iloc[-1]

    asset_y = np.concatenate([[asset_ret.values[-1]], pred_ret.values]) * 100
    asset_x = pd.DatetimeIndex([today]).append(pred_ret.index)
    bench_y = np.concatenate([[bench_today], bench_pred_ret.values]) * 100
    bench_x = pd.DatetimeIndex([today]).append(bench_pred_ret.index)

    from style import (PLOTLY_BG, PLOTLY_FONT, PLOTLY_TITLE_FONT,
                       PLOTLY_ASSET_COLOR, PLOTLY_BENCH_COLOR, PLOTLY_GRID)
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=asset_ret.index, y=asset_ret.values * 100, name=ticker_name,
        mode='lines', line=dict(color=PLOTLY_ASSET_COLOR, width=2.5),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=bench_ret.index, y=bench_ret.values * 100, name=benchmark_label,
        mode='lines', line=dict(color=PLOTLY_BENCH_COLOR, width=1.5),
        showlegend=False,
    ))

    # Predicted continuation
    fig.add_trace(go.Scatter(
        x=asset_x, y=asset_y, name=f'{ticker_name} Predicted',
        mode='lines', line=dict(color=PLOTLY_ASSET_COLOR, width=2.5, dash='dash'),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=bench_x, y=bench_y, name=f'{benchmark_label} Predicted',
        mode='lines', line=dict(color=PLOTLY_BENCH_COLOR, width=1.5, dash='dot'),
        showlegend=False,
    ))

    # 0% reference
    fig.add_shape(
        type='line', x0=asset_ret.index[0], y0=0, x1=asset_x[-1], y1=0,
        line=dict(color='rgba(0,0,0,0.08)', dash='dash', width=1),
    )

    # Demarcation between historical and predicted
    fig.add_vrect(
        x0=today,
        x1=asset_x[-1],
        fillcolor='rgba(59,130,246,0.14)',
        line_width=0,
        layer='below',
    )
    fig.add_annotation(
        x=today, y=0.02, yref='paper', text='Today',
        showarrow=False, font=dict(size=10, family=PLOTLY_FONT, color='#94A3B8'),
        xanchor='center', yanchor='bottom', yshift=5,
    )

    fig.update_layout(
        title=dict(
            text=f"{ticker_name} vs {benchmark_label}",
            font=dict(size=18, family=PLOTLY_TITLE_FONT, color='#1E293B'),
        ),
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_BG,
        margin=dict(l=50, r=30, t=55, b=40),
        height=420,
        font=dict(family=PLOTLY_FONT, color='#64748B'),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=PLOTLY_GRID, zeroline=False, title='',
        linecolor='rgba(0,0,0,0.06)',
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=PLOTLY_GRID, zeroline=False, title='Return %',
        linecolor='rgba(0,0,0,0.06)',
    )

    return fig


def build_candlestick_chart(df_hist, pred_df, ticker_name):
    """Single-panel OHLC candlestick: historical + predicted (no volume)."""
    from style import (PLOTLY_BG, PLOTLY_FONT, PLOTLY_TITLE_FONT, PLOTLY_GRID,
                       POSITIVE, NEGATIVE)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_hist.index,
        open=df_hist['open'], high=df_hist['high'],
        low=df_hist['low'], close=df_hist['close'],
        name="Historical",
        increasing_line_color="#15803d", decreasing_line_color="#b91c1c",
        increasing_fillcolor=POSITIVE, decreasing_fillcolor=NEGATIVE,
    ))

    fig.add_trace(go.Candlestick(
        x=pred_df.index,
        open=pred_df['open'], high=pred_df['high'],
        low=pred_df['low'], close=pred_df['close'],
        name="Predicted",
        increasing_line_color="#15803d", decreasing_line_color="#b91c1c",
        increasing_fillcolor="rgba(22,163,74,0.5)",
        decreasing_fillcolor="rgba(220,38,38,0.5)",
        line=dict(width=1),
        opacity=0.6,
    ))

    forecast_start = pred_df.index[0]
    chart_end = pred_df.index[-1]

    # Demarcation between historical and predicted
    fig.add_vrect(
        x0=forecast_start,
        x1=chart_end,
        fillcolor="rgba(59,130,246,0.14)",
        line_width=0,
        layer="below",
    )
    fig.update_layout(
        title=dict(
            text=f"{ticker_name} — Candlestick",
            font=dict(size=18, family=PLOTLY_TITLE_FONT, color='#1E293B'),
        ),
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_BG,
        height=520,
        xaxis_rangeslider_visible=False,
        font=dict(family=PLOTLY_FONT, color='#64748B'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(
        title_text="Price", showgrid=True, gridcolor=PLOTLY_GRID,
        linecolor='rgba(0,0,0,0.06)',
    )
    fig.update_xaxes(
        title_text="Date", showgrid=False,
        linecolor='rgba(0,0,0,0.06)',
    )

    return fig


# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    from style import inject_css, render_stats_grid

    st.set_page_config(
        page_title="Kronos Stock Predictor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()

    # Title
    st.title("Kronos")
    st.caption("AI-powered stock prediction with selectable benchmark")

    # Glassmorphic settings bar
    st.markdown('<div class="glass-bar">', unsafe_allow_html=True)
    col_b, col_l, col_p, col_t, col_btn = st.columns([1.4, 0.8, 0.8, 1.2, 0.6])
    with col_b:
        benchmark = st.selectbox(
            "Benchmark",
            options=list(BENCHMARK_CHOICES),
            index=0,
            format_func=lambda t: BENCHMARK_LABELS.get(t, t),
        )
    with col_l:
        lookback = st.slider("Lookback", min_value=20, max_value=100, value=40, step=5)
    with col_p:
        pred_len = st.slider("Forecast", min_value=5, max_value=30, value=20, step=5)
    with col_t:
        ticker = st.text_input("Ticker", value="AAPL", label_visibility="visible").strip().upper()
    with col_btn:
        st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Run Prediction", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer (muted line)
    st.markdown(
        '<p class="disclaimer-text">Predictions generated by a general time-series model. Not financial advice.</p>',
        unsafe_allow_html=True,
    )

    # Cache clear
    col_cache, _ = st.columns([0.2, 0.8])
    with col_cache:
        if st.button("Clear cache"):
            st.cache_resource.clear()
            st.success("Cleared.")

    if run_clicked:
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return

        try:
            with st.spinner(f"Fetching {ticker} data..."):
                df_full_year = fetch_yfinance_data(ticker, "1y")
                if df_full_year is None or df_full_year.empty:
                    st.error(f"No data found for \'{ticker}\'. Check the ticker symbol.")
                    return

            with st.spinner("Loading Kronos model..."):
                predictor = load_model()

            bench_label = BENCHMARK_LABELS.get(benchmark, benchmark)
            with st.spinner(f"Predicting {benchmark} benchmark..."):
                try:
                    bench_pred = predict_benchmark(benchmark, lookback, pred_len)
                    bench_hist_full = fetch_yfinance_data(benchmark, "1y")
                except Exception:
                    bench_pred = None
                    bench_hist_full = None

            with st.spinner(f"Predicting {ticker} ({lookback}d \u2192 {pred_len}d)..."):
                hist_df = df_full_year.tail(lookback)
                if len(hist_df) < lookback:
                    st.error(f"Insufficient data: {len(df_full_year)} days, need {lookback}+")
                    return

                y_timestamp = get_business_dates(
                    hist_df.index[-1] + pd.Timedelta(days=1), pred_len
                )
                set_inference_seed(KRONOS_RANDOM_SEED)
                pred_df = predictor.predict(
                    df=hist_df, x_timestamp=hist_df.index, y_timestamp=y_timestamp,
                    pred_len=pred_len, T=KRONOS_TEMPERATURE, top_p=KRONOS_TOP_P,
                    sample_count=KRONOS_SAMPLE_COUNT,
                )

            grouped_stats = compute_stats(hist_df, pred_df, df_full_year)

            # Results
            st.markdown("---")

            if bench_hist_full is not None and bench_pred is not None and not bench_hist_full.empty:
                st.plotly_chart(
                    build_return_chart(
                        df_full_year, pred_df,
                        bench_hist_full, bench_pred,
                        ticker, bench_label,
                    ),
                    use_container_width=True,
                )
            else:
                st.warning(f"Could not load benchmark {benchmark}.")

            st.plotly_chart(
                build_candlestick_chart(hist_df, pred_df, ticker),
                use_container_width=True,
            )

            render_stats_grid(grouped_stats)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
