"""
GV2-EDGE V9.0 — Professional Trading Dashboard
===============================================

Dashboard temps réel pour le système de détection anticipative
des top gainers small caps US.

Stack: Streamlit + Plotly + Custom CSS
Auto-refresh: streamlit-autorefresh (non-bloquant)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

# Auto-refresh (non-blocking)
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="GV2-EDGE V9.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS
# ============================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --bg-hover: #252b3b;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --border: #374151;
    }

    .stApp { background: linear-gradient(135deg, #0a0e17 0%, #0f172a 100%); font-family: 'Outfit', sans-serif; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    h1, h2, h3 { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; color: var(--text-primary) !important; }
    h1 {
        font-size: 2.2rem !important;
        background: linear-gradient(90deg, #06b6d4, #10b981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }

    /* Cards */
    .card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 10px; padding: 1rem; margin: 0.4rem 0;
        transition: border-color 0.2s ease;
    }
    .card:hover { border-color: var(--accent-cyan); }

    .card-buy-strong { background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,182,212,0.08) 100%); border-left: 3px solid #10b981; }
    .card-buy        { background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(139,92,246,0.08) 100%); border-left: 3px solid #3b82f6; }
    .card-watch      { background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(239,68,68,0.08)   100%); border-left: 3px solid #f59e0b; }
    .card-early      { background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(245,158,11,0.08)  100%); border-left: 3px solid #8b5cf6; }

    /* KPI metrics */
    .kpi { text-align: center; padding: 1.2rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; }
    .kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: var(--text-primary); }
    .kpi-label { font-size: 0.78rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }
    .kpi-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; margin-top: 0.3rem; }
    .green { color: #10b981; } .red { color: #ef4444; } .yellow { color: #f59e0b; } .cyan { color: #06b6d4; } .muted { color: #6b7280; }

    /* Ticker symbol */
    .tick { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.1rem; color: #06b6d4; }
    .badge {
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
        font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    }
    .badge-strong { background: rgba(16,185,129,0.2); color: #10b981; }
    .badge-buy    { background: rgba(59,130,246,0.2);  color: #3b82f6; }
    .badge-watch  { background: rgba(245,158,11,0.2);  color: #f59e0b; }
    .badge-early  { background: rgba(139,92,246,0.2);  color: #8b5cf6; }

    /* Status pulse */
    .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; animation: pulse 1.5s infinite; margin-right: 5px; }
    @keyframes pulse { 0%,100% { opacity:1; box-shadow: 0 0 4px #10b981; } 50% { opacity:0.5; box-shadow: none; } }

    /* Log viewer */
    .log-container {
        background: #0d1117; border: 1px solid var(--border); border-radius: 8px;
        padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
        max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;
    }
    .log-error  { color: #ef4444; }
    .log-warn   { color: #f59e0b; }
    .log-info   { color: #9ca3af; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: var(--bg-secondary); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: var(--bg-card); padding: 0.4rem; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; color: var(--text-secondary); font-weight: 500; }
    .stTabs [aria-selected="true"] { background: var(--bg-hover); color: var(--text-primary); }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    .score-high { color: #10b981; } .score-med { color: #f59e0b; } .score-low { color: #ef4444; }
    .divider { border-top: 1px solid var(--border); margin: 0.5rem 0; }

    /* Gap cards */
    .gap-up   { border-left: 3px solid #10b981; }
    .gap-down { border-left: 3px solid #ef4444; }
</style>
""", unsafe_allow_html=True)


# ============================
# PATHS
# ============================

DATA_DIR   = Path("data")
SIGNALS_DB = DATA_DIR / "signals_history.db"
LOGS_DIR   = DATA_DIR / "logs"
AUDIT_DIR  = DATA_DIR / "audit_reports"

# ============================
# HELPERS
# ============================

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fmt_time(ts: str) -> str:
    """Format ISO timestamp to HH:MM ET-friendly display."""
    try:
        return ts[11:16]  # HH:MM from ISO
    except Exception:
        return ""


def score_color(v: float) -> str:
    if v >= 0.65:
        return "#10b981"
    if v >= 0.50:
        return "#f59e0b"
    return "#ef4444"


def signal_badge(stype: str) -> str:
    mapping = {
        "BUY_STRONG": ("badge-strong", "🚨 BUY_STRONG"),
        "BUY":        ("badge-buy",    "✅ BUY"),
        "WATCH":      ("badge-watch",  "👀 WATCH"),
        "EARLY_SIGNAL": ("badge-early","⚡ EARLY"),
    }
    cls, label = mapping.get(stype, ("badge-watch", stype))
    return f'<span class="badge {cls}">{label}</span>'


# ============================
# DATA LOADERS
# ============================

def load_signals(hours_back: int = 24) -> pd.DataFrame:
    if not SIGNALS_DB.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(SIGNALS_DB), check_same_thread=False)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        df = pd.read_sql_query(
            f"SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp DESC",
            conn, params=(cutoff,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_monster_components(row) -> dict:
    """Extract Monster Score V4 components from a DB row (Series)."""
    return {
        "Event":       min(1.0, float(row.get("event_impact", 0) or 0)),
        "Volume":      min(1.0, float(row.get("volume_spike", 0) or 0)),
        "Pattern":     min(1.0, float(row.get("pattern_score", 0) or 0)),
        "PM Trans":    min(1.0, float(row.get("pm_transition_score", 0) or 0)),
        "Momentum":    min(1.0, float(row.get("momentum", 0) or 0)),
        "Options":     0.0,  # not yet in DB schema
        "Accel":       0.0,
        "Social":      0.0,
        "Squeeze":     0.0,
    }


def load_extended_gaps() -> list:
    """Load extended hours gap data from cache."""
    for fname in ("extended_hours_cache.json", "ah_scan.json", "pm_scan.json"):
        data = load_json(DATA_DIR / fname)
        if data:
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # flatten dict {ticker: {...}}
                out = []
                for t, v in data.items():
                    if isinstance(v, dict):
                        v["ticker"] = t
                        out.append(v)
                return out
    return []


def get_system_status() -> dict:
    status = {"ibkr": False, "grok": False, "finnhub": False, "telegram": False}
    try:
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        status["ibkr"] = bool(ibkr and ibkr.connected)
    except Exception:
        pass
    try:
        from config import GROK_API_KEY, FINNHUB_API_KEY, TELEGRAM_SIGNALS_TOKEN
        status["grok"]     = bool(GROK_API_KEY and not GROK_API_KEY.startswith("YOUR_"))
        status["finnhub"]  = bool(FINNHUB_API_KEY and not FINNHUB_API_KEY.startswith("YOUR_"))
        status["telegram"] = bool(TELEGRAM_SIGNALS_TOKEN and not TELEGRAM_SIGNALS_TOKEN.startswith("YOUR_"))
    except Exception:
        try:
            from config import GROK_API_KEY, FINNHUB_API_KEY, TELEGRAM_BOT_TOKEN
            status["grok"]    = bool(GROK_API_KEY and not GROK_API_KEY.startswith("YOUR_"))
            status["finnhub"] = bool(FINNHUB_API_KEY and not FINNHUB_API_KEY.startswith("YOUR_"))
            status["telegram"] = bool(TELEGRAM_BOT_TOKEN and not TELEGRAM_BOT_TOKEN.startswith("YOUR_"))
        except Exception:
            pass
    return status


def get_ibkr_info() -> dict | None:
    try:
        from config import USE_IBKR_DATA
        if not USE_IBKR_DATA:
            return None
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        if ibkr is None:
            return None
        return ibkr.get_connection_stats()
    except Exception:
        return None


def get_session() -> str:
    """Compute current market session from ET time (no pytz dependency)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback: UTC-4 approximation (EDT most of the year)
        now_et = datetime.now(timezone.utc) - timedelta(hours=4)

    if now_et.weekday() >= 5:  # Sat/Sun
        return "CLOSED"

    t = now_et.hour * 60 + now_et.minute
    if   4 * 60 <= t <  9 * 60 + 30:  return "PREMARKET"
    elif 9 * 60 + 30 <= t < 16 * 60:  return "RTH"
    elif 16 * 60 <= t < 20 * 60:      return "AFTER_HOURS"
    else:                              return "CLOSED"


def load_log_tail(log_file: str, n: int = 100) -> list[str]:
    """Return last N lines of a log file."""
    path = LOGS_DIR / log_file
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = deque(f, maxlen=n)
        return list(lines)
    except Exception:
        return []


def load_latest_audit() -> dict | None:
    if not AUDIT_DIR.exists():
        return None
    files = list(AUDIT_DIR.glob("*.json"))
    if not files:
        return None
    return load_json(sorted(files)[-1])


# ============================
# CHART BUILDERS
# ============================

_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1f2e",
    font=dict(color="#9ca3af"),
    margin=dict(l=40, r=20, t=30, b=30),
)


def chart_score_radar(components: dict):
    labels = list(components.keys())
    values = [components[k] for k in labels]
    values.append(values[0])
    labels.append(labels[0])
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        fillcolor="rgba(6,182,212,0.25)",
        line=dict(color="#06b6d4", width=2),
    ))
    fig.update_layout(
        **_PLOTLY_BASE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#6b7280", size=9), gridcolor="#374151"),
            angularaxis=dict(tickfont=dict(color="#9ca3af", size=10), gridcolor="#374151"),
            bgcolor="#1a1f2e",
        ),
        showlegend=False, height=280,
    )
    return fig


def chart_timeline(df: pd.DataFrame):
    color_map = {"BUY_STRONG": "#10b981", "BUY": "#3b82f6", "EARLY_SIGNAL": "#8b5cf6", "WATCH": "#f59e0b"}
    fig = px.scatter(
        df, x="timestamp", y="monster_score", color="signal_type",
        size="monster_score", hover_data=["ticker", "signal_type", "monster_score"],
        color_discrete_map=color_map,
    )
    fig.update_layout(
        **_PLOTLY_BASE,
        xaxis=dict(gridcolor="#374151", tickfont=dict(color="#9ca3af")),
        yaxis=dict(gridcolor="#374151", tickfont=dict(color="#9ca3af"), title="Monster Score"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=260,
    )
    return fig


def chart_hit_rate_gauge(hit_rate: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hit_rate * 100,
        number={"suffix": "%", "font": {"color": "#f9fafb", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#6b7280"}},
            "bar": {"color": "#06b6d4"},
            "bgcolor": "#1a1f2e", "bordercolor": "#374151",
            "steps": [
                {"range": [0, 40],  "color": "rgba(239,68,68,0.25)"},
                {"range": [40, 65], "color": "rgba(245,158,11,0.25)"},
                {"range": [65, 100],"color": "rgba(16,185,129,0.25)"},
            ],
            "threshold": {"line": {"color": "#10b981", "width": 3}, "thickness": 0.75, "value": 65},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f9fafb"}, height=230, margin=dict(l=20,r=20,t=40,b=10))
    return fig


def chart_score_distribution(df: pd.DataFrame):
    fig = px.histogram(df, x="monster_score", nbins=20, color_discrete_sequence=["#06b6d4"])
    fig.update_layout(
        **_PLOTLY_BASE,
        xaxis=dict(gridcolor="#374151", title="Score"),
        yaxis=dict(gridcolor="#374151", title="Count"),
        height=260,
    )
    return fig


def chart_radar_bars(radars: dict):
    names  = list(radars.keys())
    scores = [radars[n].get("score", 0) for n in names]
    colors = ["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b"]
    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker_color=colors[:len(names)],
        text=[f"{s:.2f}" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        **_PLOTLY_BASE,
        yaxis=dict(range=[0, 1], gridcolor="#374151"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=260,
    )
    return fig


def chart_events_pie(event_counts: pd.Series):
    fig = px.pie(
        values=event_counts.values, names=event_counts.index,
        color_discrete_sequence=["#06b6d4","#3b82f6","#8b5cf6","#f59e0b","#10b981","#ef4444"],
        hole=0.35,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af"), height=260, margin=dict(l=10,r=10,t=20,b=10))
    fig.update_traces(textfont=dict(color="white"))
    return fig


# ============================
# SIDEBAR
# ============================

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    # Auto-refresh
    auto_refresh = st.toggle("🔄 Auto Refresh", value=True)
    refresh_sec  = st.select_slider("Interval", options=[15, 30, 60, 120], value=30) if auto_refresh else 30

    st.markdown("---")

    # Filters
    st.markdown("### 🔍 Filters")
    hours_back = st.selectbox(
        "Time Range",
        [6, 12, 24, 48, 168], index=2,
        format_func=lambda x: f"Last {x}h" if x < 168 else "Last 7 days"
    )
    signal_filter = st.multiselect(
        "Signal Types",
        ["BUY_STRONG", "BUY", "WATCH", "EARLY_SIGNAL"],
        default=["BUY_STRONG", "BUY", "EARLY_SIGNAL"],
    )
    min_score = st.slider("Min Monster Score", 0.0, 1.0, 0.40, 0.05)

    st.markdown("---")

    # Quick actions
    st.markdown("### 🚀 Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Audit", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    from daily_audit import run_daily_audit
                    run_daily_audit(send_telegram=False)
                    st.success("Done!")
                except Exception as e:
                    st.error(str(e))
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # System status
    st.markdown("### 🛡️ System")
    status = get_system_status()
    for k, ok in status.items():
        st.markdown(f"{'🟢' if ok else '🔴'} **{k.upper()}**")

    # IBKR details
    ibkr_info = get_ibkr_info()
    if ibkr_info:
        st.markdown("---")
        st.markdown("### 🔌 IBKR")
        state = ibkr_info.get("state", "UNKNOWN")
        icons  = {"CONNECTED": "🟢", "RECONNECTING": "🟡", "DISCONNECTED": "🔴", "CONNECTING": "🟡", "FAILED": "🔴"}
        st.markdown(f"{icons.get(state, '⚪')} **{state}**")
        if ibkr_info.get("connected"):
            uptime = ibkr_info.get("uptime_seconds", 0)
            ustr   = f"{uptime/3600:.1f}h" if uptime >= 3600 else f"{uptime/60:.0f}min" if uptime >= 60 else f"{uptime:.0f}s"
            lat    = ibkr_info.get("heartbeat_latency_ms", 0)
            st.caption(f"Uptime: {ustr}  |  Latency: {lat:.0f}ms")

    # V9 Modules
    st.markdown("---")
    st.markdown("### 🧠 V9 Modules")
    try:
        from config import (
            ENABLE_MULTI_RADAR, ENABLE_ACCELERATION_ENGINE, ENABLE_SMALLCAP_RADAR,
            ENABLE_PRE_HALT_ENGINE, ENABLE_RISK_GUARD, ENABLE_MARKET_MEMORY,
            ENABLE_CATALYST_V3, ENABLE_PRE_SPIKE_RADAR,
        )
        mods = {
            "Signal Producer (L1)": True,
            "Order Computer (L2)": True,
            "Execution Gate (L3)": True,
            "Multi-Radar V9": ENABLE_MULTI_RADAR,
            "Acceleration V8": ENABLE_ACCELERATION_ENGINE,
            "SmallCap Radar": ENABLE_SMALLCAP_RADAR,
            "Pre-Halt Engine": ENABLE_PRE_HALT_ENGINE,
            "Risk Guard V8": ENABLE_RISK_GUARD,
            "Market Memory": ENABLE_MARKET_MEMORY,
            "Catalyst V3": ENABLE_CATALYST_V3,
            "Pre-Spike Radar": ENABLE_PRE_SPIKE_RADAR,
        }
    except Exception:
        mods = {"Signal Producer": True, "Order Computer": True, "Execution Gate": True}
    for name, active in mods.items():
        st.markdown(f"{'🟢' if active else '⚫'} {name}")

    st.markdown("---")
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    try:
        from zoneinfo import ZoneInfo as _ZI
        _now_et = datetime.now(_ZI("America/New_York"))
    except Exception:
        _now_et = datetime.now(timezone.utc) - timedelta(hours=4)
    st.caption(f"v9.0  •  {_now_et.strftime('%H:%M:%S')} ET")


# ============================
# AUTO-REFRESH (top-level, non-blocking)
# ============================

if auto_refresh:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_sec * 1000, key="main_refresh")
    else:
        st.markdown(
            f'<meta http-equiv="refresh" content="{refresh_sec}">',
            unsafe_allow_html=True,
        )


# ============================
# HEADER
# ============================

col_title, col_sess, col_time = st.columns([4, 1, 1])

with col_title:
    st.markdown("# 🎯 GV2-EDGE V9.0")
    st.markdown("**Multi-Radar Detection Architecture** — Small Caps US  |  3-Layer Pipeline")

with col_sess:
    session = get_session()
    sess_map = {
        "PREMARKET":   ("🌅", "Pre-Market",  "#f59e0b"),
        "RTH":         ("📈", "RTH Open",    "#10b981"),
        "AFTER_HOURS": ("🌙", "After-Hours", "#8b5cf6"),
        "CLOSED":      ("💤", "Closed",      "#6b7280"),
    }
    icon, label, color = sess_map.get(session, ("❓", session, "#6b7280"))
    st.markdown(f"""
    <div style="text-align:center;padding:0.8rem;">
        <div style="font-size:1.6rem;">{icon}</div>
        <div style="color:{color};font-weight:600;font-size:0.9rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_time:
    try:
        from zoneinfo import ZoneInfo as _ZI
        now_et = datetime.now(_ZI("America/New_York"))
    except Exception:
        now_et = datetime.now(timezone.utc) - timedelta(hours=4)
    st.markdown(f"""
    <div style="text-align:right;padding:0.8rem;color:#9ca3af;font-size:0.82rem;">
        <div><span class="live-dot"></span><b style="color:#10b981;">LIVE</b></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:#f9fafb;">{now_et.strftime("%H:%M:%S")}</div>
        <div>{now_et.strftime("%Y-%m-%d")} <span style="color:#06b6d4;">ET</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================
# LOAD DATA (once, shared across tabs)
# ============================

signals_df  = load_signals(hours_back)
audit_data  = load_latest_audit()
gaps_data   = load_extended_gaps()

total_sig   = len(signals_df)
buy_strong  = len(signals_df[signals_df["signal_type"] == "BUY_STRONG"]) if not signals_df.empty else 0
hit_rate    = audit_data.get("hit_rate", 0) if audit_data else 0
avg_lead    = audit_data.get("avg_lead_time_hours", 0) if audit_data else 0

universe_size = 0
for uf in (DATA_DIR / "universe.csv", DATA_DIR / "universe_v3.csv"):
    if uf.exists():
        try:
            universe_size = len(pd.read_csv(uf))
            break
        except Exception:
            pass

# ============================
# KPI ROW
# ============================

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, value, label, delta="", delta_class="muted"):
    col.markdown(f"""
    <div class="kpi">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-delta {delta_class}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

score_cls = "green" if hit_rate > 0.65 else "yellow" if hit_rate > 0.40 else "red"

kpi(k1, total_sig,             "Total Signals",  f"Last {hours_back}h")
kpi(k2, buy_strong,            "BUY_STRONG",     "🔥 Hot",           "green")
kpi(k3, f"{hit_rate*100:.1f}%","Hit Rate",       "Target ≥ 65%",      score_cls)
kpi(k4, f"{avg_lead:.1f}h",    "Avg Lead Time",  "Before spike")
kpi(k5, f"{universe_size:,}",  "Universe",       "Tickers tracked",  "cyan")

st.markdown("<br>", unsafe_allow_html=True)


# ============================
# TABS
# ============================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live Signals",
    "📊 Analytics",
    "📅 Events",
    "🛰️ Multi-Radar V9",
    "📋 Live Logs",
    "🔍 Audit",
])


# ─────────────────────────────
# TAB 1 — LIVE SIGNALS
# ─────────────────────────────

with tab1:
    st.markdown("### 🔥 Active Trading Signals")

    # Extended hours gaps (prominent)
    if gaps_data:
        st.markdown("#### 📈 Extended Hours — Top Movers")
        sorted_gaps = sorted(gaps_data, key=lambda x: abs(x.get("gap_pct", x.get("gap", 0))), reverse=True)[:10]
        gcols = st.columns(min(len(sorted_gaps), 5))
        for i, g in enumerate(sorted_gaps[:5]):
            ticker  = g.get("ticker", "?")
            gap_pct = g.get("gap_pct", g.get("gap", 0))
            vol     = g.get("volume", g.get("vol", 0))
            price   = g.get("price", g.get("last", 0))
            color   = "#10b981" if gap_pct > 0 else "#ef4444"
            sign    = "+" if gap_pct > 0 else ""
            gcols[i].markdown(f"""
            <div class="card {'gap-up' if gap_pct > 0 else 'gap-down'}">
                <div class="tick">{ticker}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:{color};font-weight:700;">
                    {sign}{gap_pct:.1f}%
                </div>
                <div class="muted" style="font-size:0.75rem;">
                    Vol: {int(vol):,} {"| $"+str(round(price,2)) if price else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    if signals_df.empty:
        st.info("No signals in the selected time range. The engine is scanning...")
    else:
        fdf = signals_df.copy()
        if signal_filter:
            fdf = fdf[fdf["signal_type"].isin(signal_filter)]
        fdf = fdf[fdf["monster_score"] >= min_score]

        if fdf.empty:
            st.warning("No signals match the current filters.")
        else:
            # 3-column split by type
            col_s, col_b, col_w = st.columns(3)

            def render_signal_card(row, card_cls):
                ticker    = row.get("ticker", "?")
                stype     = row.get("signal_type", "WATCH")
                score     = float(row.get("monster_score", 0) or 0)
                ts        = fmt_time(str(row.get("timestamp", "")))
                entry     = row.get("entry_price")
                stop      = row.get("stop_loss")
                shares    = row.get("shares")
                ev_imp    = float(row.get("event_impact", 0) or 0)
                vol_spk   = float(row.get("volume_spike", 0) or 0)

                entry_str  = f"${entry:.2f}" if entry else "—"
                stop_str   = f"${stop:.2f}"  if stop  else "—"
                shares_str = str(int(shares)) if shares else "—"

                return f"""
                <div class="card {card_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="tick">{ticker}</span>
                        {signal_badge(stype)}
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:0.6rem;">
                        <div style="text-align:center;">
                            <div style="color:#9ca3af;font-size:0.7rem;">SCORE</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{score_color(score)};">{score:.2f}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#9ca3af;font-size:0.7rem;">ENTRY</div>
                            <div style="font-family:'JetBrains Mono',monospace;color:#f9fafb;">{entry_str}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#9ca3af;font-size:0.7rem;">STOP</div>
                            <div style="font-family:'JetBrains Mono',monospace;color:#ef4444;">{stop_str}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#9ca3af;font-size:0.7rem;">SHARES</div>
                            <div style="font-family:'JetBrains Mono',monospace;color:#f9fafb;">{shares_str}</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:0.5rem;margin-top:0.5rem;">
                        <span style="font-size:0.7rem;color:#9ca3af;">Ev:{ev_imp:.2f}</span>
                        <span style="font-size:0.7rem;color:#9ca3af;">Vol:{vol_spk:.2f}</span>
                        <span style="font-size:0.7rem;color:#6b7280;margin-left:auto;">{ts}</span>
                    </div>
                </div>"""

            with col_s:
                st.markdown("#### 🚨 BUY_STRONG")
                df_s = fdf[fdf["signal_type"] == "BUY_STRONG"]
                if not df_s.empty:
                    for _, r in df_s.head(6).iterrows():
                        st.markdown(render_signal_card(r, "card-buy-strong"), unsafe_allow_html=True)
                else:
                    st.caption("No BUY_STRONG")

            with col_b:
                st.markdown("#### ✅ BUY")
                df_b = fdf[fdf["signal_type"] == "BUY"]
                if not df_b.empty:
                    for _, r in df_b.head(6).iterrows():
                        st.markdown(render_signal_card(r, "card-buy"), unsafe_allow_html=True)
                else:
                    st.caption("No BUY")

            with col_w:
                st.markdown("#### 👀 WATCH / EARLY")
                df_w = fdf[fdf["signal_type"].isin(["WATCH", "EARLY_SIGNAL"])]
                if not df_w.empty:
                    for _, r in df_w.head(6).iterrows():
                        cls = "card-early" if r.get("signal_type") == "EARLY_SIGNAL" else "card-watch"
                        st.markdown(render_signal_card(r, cls), unsafe_allow_html=True)
                else:
                    st.caption("No WATCH/EARLY")

            st.markdown("<br>", unsafe_allow_html=True)

            # Timeline
            if "timestamp" in fdf.columns and not fdf.empty:
                st.markdown("#### 📈 Signals Timeline")
                st.plotly_chart(chart_timeline(fdf), use_container_width=True)

            # Full table
            with st.expander("📋 Full Signals Table"):
                cols = [c for c in ["timestamp","ticker","signal_type","monster_score","entry_price","stop_loss","shares","event_impact","volume_spike"] if c in fdf.columns]
                st.dataframe(fdf[cols].head(100), use_container_width=True, hide_index=True)


# ─────────────────────────────
# TAB 2 — ANALYTICS
# ─────────────────────────────

with tab2:
    st.markdown("### 📊 Score Analytics")

    col_radar, col_dist = st.columns(2)

    with col_radar:
        st.markdown("#### Monster Score V4 Breakdown")
        if not signals_df.empty:
            buy_rows = signals_df[signals_df["signal_type"].isin(["BUY_STRONG", "BUY"])]
            if not buy_rows.empty:
                comp = load_monster_components(buy_rows.iloc[0])
                st.plotly_chart(chart_score_radar(comp), use_container_width=True)
                st.caption("Last BUY/BUY_STRONG signal — Options/Accel/Social/Squeeze pending extended DB schema")
            else:
                st.info("Aucun signal BUY en base — radar disponible après le premier cycle RTH/PM")
        else:
            st.info("No signals yet")

    with col_dist:
        st.markdown("#### Score Distribution")
        if not signals_df.empty and "monster_score" in signals_df.columns:
            st.plotly_chart(chart_score_distribution(signals_df), use_container_width=True)
        else:
            st.info("No data")

    st.markdown("---")

    # Signals by type bar
    st.markdown("#### Signal Type Breakdown")
    if not signals_df.empty:
        type_counts = signals_df["signal_type"].value_counts().reset_index()
        type_counts.columns = ["Signal", "Count"]
        color_map = {"BUY_STRONG": "#10b981", "BUY": "#3b82f6", "WATCH": "#f59e0b", "EARLY_SIGNAL": "#8b5cf6"}
        fig_types = px.bar(
            type_counts, x="Signal", y="Count",
            color="Signal", color_discrete_map=color_map,
            text="Count",
        )
        fig_types.update_layout(
            **_PLOTLY_BASE, showlegend=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#374151"),
            height=250,
        )
        fig_types.update_traces(textposition="outside")
        st.plotly_chart(fig_types, use_container_width=True)
    else:
        st.info("No signals to plot")

    st.markdown("---")

    # Boost reference
    st.markdown("#### 🚀 V9 Intelligence Boosts (theoretical max)")
    boost_data = pd.DataFrame({
        "Boost": ["Beat Rate", "Extended Hours", "Acceleration V8", "Insider (V8)", "Short Squeeze"],
        "Max":   [0.15, 0.22, 0.15, 0.15, 0.20],
        "Source": ["Historical earnings", "Gap + AH/PM vol", "ACCUMULATING/LAUNCHING", "SEC Form 4", "Short float"],
    })
    st.dataframe(boost_data, use_container_width=True, hide_index=True)

    # Proposed orders in last 24h
    st.markdown("---")
    st.markdown("#### 📋 Proposed Orders (last 24h)")
    if not signals_df.empty:
        order_df = signals_df[signals_df["signal_type"].isin(["BUY_STRONG", "BUY"])].copy()
        order_df = order_df[["timestamp", "ticker", "signal_type", "monster_score", "entry_price", "stop_loss", "shares"]].head(20)
        order_df.columns = ["Time", "Ticker", "Signal", "Score", "Entry $", "Stop $", "Shares"]
        st.dataframe(order_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ IBKR Gateway = READ ONLY — ces ordres ne sont PAS exécutés")
    else:
        st.info("No BUY signals recorded yet")


# ─────────────────────────────
# TAB 3 — EVENTS
# ─────────────────────────────

with tab3:
    st.markdown("### 📅 Detected Events & Catalysts")

    events_cache = load_json(DATA_DIR / "events_cache.json")

    col_ev, col_side = st.columns([2, 1])

    with col_ev:
        if events_cache:
            df_ev = pd.DataFrame(events_cache)
            if not df_ev.empty and "type" in df_ev.columns:
                ec = df_ev["type"].value_counts()
                st.markdown("#### Event Types")
                st.plotly_chart(chart_events_pie(ec), use_container_width=True)

                st.markdown("#### Recent Events")
                show_cols = [c for c in ["ticker","type","boosted_impact","date","is_bearish"] if c in df_ev.columns]
                st.dataframe(df_ev[show_cols].head(25), use_container_width=True, hide_index=True)
        else:
            st.info("No events cached yet — will populate after first scan cycle")

    with col_side:
        st.markdown("#### Upcoming Catalysts")
        fda_data = load_json(DATA_DIR / "fda_calendar.json")
        watch_data = load_json(DATA_DIR / "watchlist.json")
        upcoming = []
        if fda_data and isinstance(fda_data, list):
            upcoming = fda_data[:8]
        elif watch_data and isinstance(watch_data, list):
            upcoming = watch_data[:8]

        if upcoming:
            for item in upcoming:
                ticker = item.get("ticker", item.get("symbol", "?"))
                event  = item.get("event_type", item.get("type", item.get("catalyst", "Event")))
                date   = item.get("date", item.get("event_date", ""))
                st.markdown(f"""
                <div class="card">
                    <span class="tick">{ticker}</span>
                    <span style="color:#f59e0b;margin-left:0.4rem;font-size:0.82rem;">{event}</span>
                    <div style="font-size:0.72rem;color:#6b7280;">{date}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;color:#6b7280;font-size:0.82rem;">
                data/fda_calendar.json non généré<br>
                Lance batch_scheduler pour peupler
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────
# TAB 4 — MULTI-RADAR V9
# ─────────────────────────────

with tab4:
    st.markdown("### 🛰️ Multi-Radar Engine V9")
    st.markdown("""
    <div class="card" style="color:#9ca3af;font-size:0.85rem;">
        4 radars indépendants (<b style="color:#06b6d4">asyncio.gather</b>) →
        Confluence Matrix 2D (Flow × Catalyst) →
        Signal final + modifiers (Smart Money, Sentiment)
    </div>
    """, unsafe_allow_html=True)

    # Session weights table
    st.markdown("#### Session Adapter — Poids par sous-session")
    sw = pd.DataFrame({
        "Sous-session": ["AFTER_HOURS","PRE_MARKET","RTH_OPEN","RTH_MIDDAY","RTH_CLOSE","CLOSED"],
        "Flow":         [15, 30, 35, 40, 30,  5],
        "Catalyst":     [45, 30, 20, 20, 30, 50],
        "Smart Money":  [10, 15, 30, 25, 25,  5],
        "Sentiment":    [30, 25, 15, 15, 15, 40],
    })

    # Highlight current session
    current_row = {"AFTER_HOURS": 0, "PREMARKET": 1, "RTH": 2, "CLOSED": 5}.get(session, -1)
    st.dataframe(sw, use_container_width=True, hide_index=True)

    col_m, col_mod = st.columns(2)
    with col_m:
        st.markdown("#### Confluence Matrix")
        st.markdown("""
| Flow \\ Catalyst | HIGH ≥0.6 | MEDIUM 0.3-0.6 | LOW <0.3 |
|:---------------|:---------:|:--------------:|:--------:|
| **HIGH**       | BUY_STRONG | BUY           | WATCH    |
| **MEDIUM**     | BUY        | WATCH         | EARLY    |
| **LOW**        | WATCH      | EARLY         | NO_SIGNAL|
        """)
    with col_mod:
        st.markdown("#### Modifiers")
        st.markdown("""
- **Smart Money HIGH** → upgrade +1 niveau
- **Sentiment HIGH** + 2+ radars actifs → upgrade +1
- **4/4 UNANIMOUS** → +0.15 bonus, min BUY si score > 0.50
- **3/4 STRONG** → +0.10 bonus
- **2/4 MODERATE** → +0.05 bonus
        """)

    # Live results from DB
    st.markdown("#### Dernier résultat Multi-Radar (depuis DB)")

    def load_last_radar():
        if not SIGNALS_DB.exists():
            return None
        try:
            conn = sqlite3.connect(str(SIGNALS_DB), check_same_thread=False)
            row = conn.execute("""
                SELECT ticker, metadata FROM signals
                WHERE signal_type IN ('BUY_STRONG','BUY')
                  AND metadata IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            conn.close()
            if row:
                meta = json.loads(row[1]) if row[1] else {}
                radar = meta.get("multi_radar_result")
                if isinstance(radar, dict) and "radars" in radar:
                    return row[0], radar
        except Exception:
            pass
        return None

    rdata = load_last_radar()
    if rdata:
        ticker_r, rd = rdata
        st.markdown(
            f"**Ticker:** `{ticker_r}` &nbsp;|&nbsp; "
            f"**Signal:** `{rd.get('signal_type','?')}` &nbsp;|&nbsp; "
            f"**Score:** `{rd.get('final_score',0):.2f}` &nbsp;|&nbsp; "
            f"**Agreement:** `{rd.get('agreement','?')}`"
        )
        radars = rd.get("radars", {})
        if radars:
            rows = []
            for name, info in radars.items():
                rows.append({
                    "Radar": name,
                    "Score": f"{info.get('score',0):.2f}",
                    "Confidence": f"{info.get('confidence',0):.0%}",
                    "State": info.get("state", "—"),
                    "Signals": ", ".join(info.get("signals",[])[:3]),
                    "Scan ms": f"{info.get('scan_time_ms',0):.1f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.plotly_chart(chart_radar_bars(radars), use_container_width=True)
    else:
        st.info(
            "Aucun résultat Multi-Radar en base. "
            "Le moteur V9 doit effectuer au moins un cycle complet avec ENABLE_MULTI_RADAR=True."
        )


# ─────────────────────────────
# TAB 5 — LIVE LOGS
# ─────────────────────────────

with tab5:
    st.markdown("### 📋 Live System Logs")

    # Build list of available log files
    log_files = []
    if LOGS_DIR.exists():
        log_files = sorted(f.name for f in LOGS_DIR.glob("*.log"))

    col_logctrl, col_logview = st.columns([1, 3])

    with col_logctrl:
        st.markdown("#### 🗂️ Select Log")
        PRIORITY_LOGS = ["main.log", "signal_producer.log", "multi_radar.log",
                         "execution_gate.log", "ibkr_connector.log", "telegram_alerts.log"]

        if log_files:
            # Put priority logs first
            ordered = [l for l in PRIORITY_LOGS if l in log_files] + \
                      [l for l in log_files if l not in PRIORITY_LOGS]
            selected_log = st.selectbox("Log file", ordered, index=0)
        else:
            selected_log = None
            st.warning("No log files found at data/logs/")

        n_lines = st.select_slider("Lines to show", [50, 100, 200, 500], value=100)

        level_filter = st.multiselect(
            "Filter by level",
            ["INFO", "WARNING", "ERROR", "DEBUG"],
            default=["INFO", "WARNING", "ERROR"],
        )

        keyword = st.text_input("Keyword filter", placeholder="e.g. BUY_STRONG, RNA, ERROR")

        st.markdown("---")

        # Log file sizes
        st.markdown("#### 📁 File Sizes")
        if log_files:
            for lf in log_files[:10]:
                try:
                    sz = (LOGS_DIR / lf).stat().st_size
                    sz_str = f"{sz/1024:.1f}KB" if sz < 1_048_576 else f"{sz/1_048_576:.1f}MB"
                    is_sel = "**" if lf == selected_log else ""
                    st.caption(f"{is_sel}{lf}{is_sel} — {sz_str}")
                except Exception:
                    pass

    with col_logview:
        if selected_log:
            lines = load_log_tail(selected_log, n_lines)

            # Apply filters
            if level_filter:
                lines = [l for l in lines if any(lvl in l for lvl in level_filter)]
            if keyword:
                kw_lower = keyword.lower()
                lines = [l for l in lines if kw_lower in l.lower()]

            st.markdown(f"#### 📄 `{selected_log}` — last {len(lines)} lines")

            if lines:
                # Color-code by level
                html_lines = []
                for line in lines:
                    line_esc = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    if "| ERROR |" in line or "ERROR" in line[:50]:
                        html_lines.append(f'<span class="log-error">{line_esc}</span>')
                    elif "| WARNING |" in line or "WARNING" in line[:50]:
                        html_lines.append(f'<span class="log-warn">{line_esc}</span>')
                    else:
                        html_lines.append(f'<span class="log-info">{line_esc}</span>')

                log_html = "".join(html_lines)
                st.markdown(
                    f'<div class="log-container">{log_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No lines match the current filters")
        else:
            st.info("Select a log file on the left")

    # Quick multi-log summary
    st.markdown("---")
    st.markdown("#### ⚡ Recent ERRORs across all logs")
    if LOGS_DIR.exists() and log_files:
        error_lines = []
        for lf in log_files:
            for line in load_log_tail(lf, 200):
                if "| ERROR |" in line or ("ERROR" in line[:50] and "|" in line):
                    error_lines.append(f"[{lf}] {line.strip()}")
        if error_lines:
            # Last 20
            for line in error_lines[-20:]:
                st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#ef4444;padding:0.1rem 0;">{line}</div>', unsafe_allow_html=True)
        else:
            st.success("No recent errors across all log files ✅")
    else:
        st.info("Log directory not available (check data/logs/)")


# ─────────────────────────────
# TAB 6 — AUDIT
# ─────────────────────────────

with tab6:
    st.markdown("### 🔍 Performance Audit")

    if audit_data:
        col_g, col_m = st.columns([1, 2])

        with col_g:
            st.markdown("#### Hit Rate")
            st.plotly_chart(chart_hit_rate_gauge(audit_data.get("hit_rate", 0)), use_container_width=True)

        with col_m:
            st.markdown("#### Metrics")
            metrics = [
                ("Total Signals",   audit_data.get("total_signals", 0),                     "📡"),
                ("True Positives",  audit_data.get("true_positives", 0),                    "🟢"),
                ("False Positives", audit_data.get("false_positives", 0),                   "🔴"),
                ("Missed Movers",   audit_data.get("missed_movers", 0),                     "⚠️"),
                ("Early Catch",     f"{audit_data.get('early_catch_rate',0)*100:.1f}%",      "⏰"),
                ("Avg Lead Time",   f"{audit_data.get('avg_lead_time_hours',0):.1f}h",       "📊"),
            ]
            ca, cb, cc = st.columns(3)
            for i, (label, val, ico) in enumerate(metrics):
                [ca, cb, cc][i % 3].markdown(f"""
                <div class="card" style="text-align:center;">
                    <div style="color:#9ca3af;font-size:0.75rem;">{ico} {label}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#f9fafb;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📋 Full Audit JSON"):
            st.json(audit_data)

        if "missed_details" in audit_data and audit_data["missed_details"]:
            st.markdown("#### ❌ Missed Movers")
            st.dataframe(pd.DataFrame(audit_data["missed_details"]), use_container_width=True, hide_index=True)

    else:
        st.info("No audit report yet.")
        if st.button("🚀 Run Daily Audit Now"):
            with st.spinner("Running..."):
                try:
                    from daily_audit import run_daily_audit
                    run_daily_audit(send_telegram=False)
                    st.success("Done! Refresh to see results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Audit failed: {e}")


# ============================
# FOOTER
# ============================

st.markdown("---")
fc1, fc2, fc3 = st.columns(3)
fc1.caption("🎯 GV2-EDGE V9.0 — Multi-Radar Detection Architecture")
try:
    from zoneinfo import ZoneInfo as _ZI
    _ft = datetime.now(_ZI("America/New_York"))
except Exception:
    _ft = datetime.now(timezone.utc) - timedelta(hours=4)
fc2.caption(f"Updated: {_ft.strftime('%Y-%m-%d %H:%M:%S')} ET")
fc3.caption("Small Caps US — Anticipation > Reaction 🚀")
