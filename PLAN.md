---

name: Kronos Web App
overview: Clone the GitHub repo into a clean self-contained project, remove unnecessary files, add the Kronos model and build a Streamlit web app for stock prediction with technical stats and a selectable ETF benchmark (ACWI, SPY, or CQQQ).
todos:

- id: clone-repo
content: Clone GitHub repo to local path and populate with model/ directory
status: pending
- id: write-requirements
content: "Create requirements.txt with all dependencies (streamlit, yfinance, pandas, numpy, plotly, torch, einops, huggingface_hub, safetensors, ta, gunicorn)"
status: pending
- id: create-streamlit-config
content: Create .streamlit/config.toml (light theme, server port 8501)
status: pending
- id: write-app-core
content: "Write app.py: model loading (@st.cache_resource), data fetching with yfinance retry/fallback, prediction engine, cached benchmark predictions (ACWI / SPY / CQQQ)"
status: pending
- id: write-stats
content: "Write app.py: compute_stats function with graceful fallbacks (SMA 200 N/A if < 200 days)"
status: pending
- id: write-charts
content: "Write app.py: Plotly charts (1-year % return vs selected benchmark + predictions, candlestick OHLC)"
status: pending
- id: write-ui
content: "Write app.py: Streamlit UI layout with main-column settings, error handling, metrics grid"
status: pending
- id: create-deployment
content: Create Dockerfile and railway.toml for Railway deployment
status: pending
- id: update-readme
content: Update README.md with setup and deployment instructions
status: pending
- id: test-locally
content: Test the app locally with a sample ticker (e.g., AAPL)
status: pending
isProject: false

---

# Kronos Prediction Web App — Implementation Plan

## Architecture Overview

The app is entirely self-contained in the `Kronos-Web` repo. It imports `model/` locally — no external path references.

**Model loading note:** The Dockerfile pre-bakes model weights into the image, but Kronos internally downloads from HuggingFace if files are missing locally. We'll need to ensure `Kronos.from_pretrained()` loads from a local path (e.g., `./model_cache/`) rather than fetching at runtime. The Dockerfile will handle this by downloading weights during image build.

## Step 1: Clone & Populate the Repo

1. Clone `https://github.com/ArjunDivecha/Kronos-Web` to `/Users/arjundivecha/Dropbox/AAA Backup/A Complete/Kronos Web`
2. Copy `shiyu-coder-Kronos/model/__init__.py`, `kronos.py`, `module.py` into the cloned repo's `model/` directory
3. Nothing to delete — the GitHub repo is empty/new

Final structure:

```
Kronos Web/
  app.py                    # Streamlit app (~500-600 lines)
  requirements.txt          # All Python dependencies
  Dockerfile                # For Railway deployment
  railway.toml              # Railway deployment config
  .streamlit/
    config.toml             # Streamlit config (light theme)
  model/                    # Kronos core model (local copy)
    __init__.py
    kronos.py
    module.py
  README.md                 # Setup + deployment instructions
```

## Step 2: requirements.txt & .streamlit/config.toml

Basic deps: streamlit, yfinance, pandas, numpy, plotly, torch, einops, huggingface_hub, safetensors, ta, gunicorn (for Railway production serving)

**yfinance backup strategy** (minimal usage, but defensive):

1. Primary: `yfinance.Ticker(ticker).history(period="2y")`
2. Fallback 1: Retry with period="1y" if 2y fails (still enough for stats)
3. Fallback 2: `yfinance.download([ticker])` as alternative API call
4. All calls wrapped with retry logic (max 2 retries, 5s delay)
5. If all fail: display error with cached ticker data if available

**Streamlit production startup**: Use `gunicorn` with `streamlit.server` or a `Procfile` running `streamlit run app.py` via Railway's Docker entrypoint.

Light theme (Streamlit default) with server port 8501.

## Step 3: app.py — Full App

Import: `from model import Kronos, KronosTokenizer, KronosPredictor` (local package, no sys.path hacks)

**Model loading**: `@st.cache_resource` to load once per server process (~410MB, 20-30s first load).

**Data fetching**: `yfinance.Ticker(ticker).history(period="2y")`, slice last 252 rows (1 year) and last 40 rows (lookback). `amount = volume * open`. `y_timestamp` = next 20 business dates.

**Prediction**: `predictor.predict(..., T=0.6, top_p=0.9, sample_count=10)` with RNG seeded to 42 before each run for reproducibility.

**Benchmark caching**: `@st.cache_resource` caches each benchmark ETF’s Kronos prediction keyed by ticker plus lookback and prediction length. The **Clear benchmark cache** button clears cached resources (including benchmark runs); the model loader may reload on next use.

**Technical stats**: SMA 20/50/200 (SMA 200: if insufficient data, display "N/A — need 200+ trading days"), EMA 12/26, RSI 14, MACD/Signal/Hist, Bollinger Bands (20, ±2σ), ATR 14, Avg Volume 20d, Forecasted Return %, 52-Week High/Low, Day Change %, Current Price.

**Prediction visualization**: With `sample_count=10`, compute mean + prediction bands (min/max range shown as shaded area) so users can see prediction confidence.

**Chart 1**: 1-Year % Return — asset and selected benchmark (ACWI, SPY, or CQQQ via dropdown) normalized to 0% at the start of the 1y window, with predicted returns on the same scale as dashed continuations. Vertical "Today" separator.

**Chart 2**: Candlestick — lookback historical + predicted horizon; green/red by OHLC direction; predicted fills slightly transparent. No volume panel.

**UI layout**: Stacked vertical — Settings (benchmark dropdown, lookback/prediction sliders, cache clear) in the main column, then ticker input and charts, then stats grid.

**Error handling**:

- Invalid tickers → friendly error message with example valid tickers
- yfinance timeouts → retry logic with 2 retries, 5s delay; fallback to cached data if available
- < 40 days data → reject with clear message
- Model loading failures → show error with troubleshooting steps
- Railway RAM limits → warning banner if memory exceeds threshold

**Prediction disclaimer**: Banner at top: "Predictions generated using a general time-series model and are not financial advice."

## Step 4: Deployment (Dockerfile + Railway)

Dockerfile pre-downloads model weights from HuggingFace into the image (avoids 60s cold-start download at runtime). Railway.toml configures Docker build.

**Platform feasibility:**

- Railway: YES (recommended, $5-10/mo for 2GB RAM)
- Vercel Pro: Possible but awkward (Python ML workloads, 15min timeout needed)
- Cloudflare Workers: NOT feasible (128MB limit, model is 410MB)

## Step 5: README with setup + deployment instructions

