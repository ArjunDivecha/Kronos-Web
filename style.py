"""
Kronos Dashboard -- CSS theme injection and HTML rendering utilities.
Provides inject_css() for global styling and render_stats_grid() for Bento card layout.
"""
import streamlit as st

# ── Palette ──────────────────────────────────────────────────
BG_PAGE = "#FAF7F2"
BG_CARD = "#FFFFFF"
ACCENT = "#4F46E5"
ACCENT_HOVER = "#4338CA"
TEXT_PRIMARY = "#1E293B"
TEXT_SECONDARY = "#64748B"
TEXT_MUTED = "#94A3B8"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
BORDER = "#E2E8F0"
SHADOW_SM = "0 1px 4px rgba(0,0,0,0.06)"
SHADOW_MD = "0 4px 12px rgba(0,0,0,0.08)"

# ── Plotly constants (reused by app.py chart builders) ───────
PLOTLY_BG = BG_PAGE
PLOTLY_FONT = "Inter, sans-serif"
PLOTLY_TITLE_FONT = "DM Serif Display, serif"
PLOTLY_ASSET_COLOR = ACCENT
PLOTLY_BENCH_COLOR = "#94A3B8"
PLOTLY_GRID = "rgba(0,0,0,0.04)"


def inject_css() -> None:
    """Inject Google Fonts + global CSS overrides into the Streamlit page."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

        <style>
        /* ── Page background & base font ─────────────────── */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: %(bg)s !important;
            font-family: 'Inter', sans-serif !important;
            color: %(text)s;
        }
        .stApp > header { background-color: transparent !important; }

        /* ── Hide hamburger menu, footer, deploy button ──── */
        #MainMenu, footer, [data-testid="stStatusWidget"],
        button[data-testid="manage-app-button"],
        [data-testid="stDeployButton"] { display: none !important; }

        /* ── Title (h1) ──────────────────────────────────── */
        h1 {
            font-family: 'DM Serif Display', serif !important;
            font-weight: 400 !important;
            font-size: 2.6rem !important;
            letter-spacing: -0.02em !important;
            color: %(text)s !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* ── Section headers (h2, h3) ────────────────────── */
        h2, h3 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: %(text)s !important;
        }

        /* ── Caption / subtitle ──────────────────────────── */
        .stCaption, [data-testid="stCaptionContainer"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 1rem !important;
            color: %(muted)s !important;
            margin-top: -4px !important;
        }

        /* ── Primary button ──────────────────────────────── */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, %(accent)s 0%%, %(accent_h)s 100%%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 12px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.6rem 1.8rem !important;
            box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
            transition: all 0.2s ease-out !important;
            max-width: 260px !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
        }

        /* ── Secondary / default buttons ─────────────────── */
        .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
            border-radius: 10px !important;
            border: 1px solid %(border)s !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.85rem !important;
            color: %(sec)s !important;
            background: %(bg)s !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {
            border-color: %(accent)s !important;
            color: %(accent)s !important;
        }

        /* ── Text input ──────────────────────────────────── */
        .stTextInput > div > div > input {
            border-radius: 12px !important;
            border: 1px solid %(border)s !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
            padding: 0.55rem 0.9rem !important;
            background: #fff !important;
            transition: border-color 0.15s ease !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: %(accent)s !important;
            box-shadow: 0 0 0 2px rgba(79,70,229,0.12) !important;
        }

        /* ── Select box ──────────────────────────────────── */
        [data-testid="stSelectbox"] > div > div {
            border-radius: 12px !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* ── Slider accent ───────────────────────────────── */
        .stSlider [data-testid="stThumbValue"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem !important;
        }

        /* ── Info / alert banners ─────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: 12px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
        }

        /* ── Divider ─────────────────────────────────────── */
        hr {
            border-color: %(border)s !important;
            opacity: 0.5 !important;
        }

        /* ── Glassmorphic container ──────────────────────── */
        .glass-bar {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(226, 232, 240, 0.55);
            border-radius: 16px;
            padding: 20px 28px 16px 28px;
            margin-bottom: 18px;
        }

        /* ── Metric overrides (fallback for non-bento) ──── */
        [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif !important;
            text-transform: uppercase !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.06em !important;
            color: %(sec)s !important;
        }

        /* ── Bento grid ──────────────────────────────────── */
        .bento-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-top: 6px;
        }
        @media (max-width: 768px) {
            .bento-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .bento-section-label {
            grid-column: 1 / -1;
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: %(muted)s;
            padding: 12px 0 2px 4px;
            border-bottom: 1px solid %(border)s;
            margin-bottom: 2px;
        }
        .bento-card {
            background: %(card)s;
            border: 1px solid %(border)s;
            border-radius: 14px;
            padding: 18px 16px;
            box-shadow: %(shadow_sm)s;
            transition: transform 0.18s ease-out, box-shadow 0.18s ease-out;
        }
        .bento-card:hover {
            transform: translateY(-2px);
            box-shadow: %(shadow_md)s;
        }
        .bento-card.hero { grid-column: span 2; }
        .bento-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: %(sec)s;
            margin-bottom: 6px;
        }
        .bento-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.35rem;
            font-weight: 600;
            color: %(text)s;
            line-height: 1.2;
        }
        .bento-value.positive { color: %(pos)s; }
        .bento-value.negative { color: %(neg)s; }

        /* ── Disclaimer muted line ───────────────────────── */
        .disclaimer-text {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: %(muted)s;
            text-align: center;
            padding: 4px 0 10px 0;
        }
        </style>
        """ % {
            "bg": BG_PAGE,
            "card": BG_CARD,
            "text": TEXT_PRIMARY,
            "sec": TEXT_SECONDARY,
            "muted": TEXT_MUTED,
            "accent": ACCENT,
            "accent_h": ACCENT_HOVER,
            "border": BORDER,
            "shadow_sm": SHADOW_SM,
            "shadow_md": SHADOW_MD,
            "pos": POSITIVE,
            "neg": NEGATIVE,
        },
        unsafe_allow_html=True,
    )


def _value_class(value_str: str) -> str:
    """Return a CSS class based on whether the value is positive, negative, or neutral."""
    if not value_str or value_str == "N/A":
        return ""
    stripped = value_str.replace("$", "").replace("%", "").replace(",", "").strip()
    if stripped.startswith("+"):
        return " positive"
    if stripped.startswith("-"):
        return " negative"
    return ""


def render_stats_grid(grouped: dict) -> None:
    """Render a Bento Grid of stat cards from a grouped stats dictionary.

    grouped = {
        "Section Name": {
            "Label": {"value": "$123.45", "hero": False},
            ...
        },
        ...
    }
    """
    html_parts = ['<div class="bento-grid">']

    for section, items in grouped.items():
        html_parts.append(
            f'<div class="bento-section-label">{section}</div>'
        )
        for label, info in items.items():
            val = info["value"]
            hero = "hero" if info.get("hero") else ""
            vcls = _value_class(val)
            html_parts.append(
                f'<div class="bento-card {hero}">'
                f'  <div class="bento-label">{label}</div>'
                f'  <div class="bento-value{vcls}">{val}</div>'
                f'</div>'
            )

    html_parts.append("</div>")
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)
