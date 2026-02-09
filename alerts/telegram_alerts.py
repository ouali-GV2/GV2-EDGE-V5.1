"""
GV2-EDGE V6.0 — Telegram Alerts System
=======================================

Alertes enrichies avec:
- EVENT_TYPE emojis (18 types, 5 tiers)
- Catalyst Score V3 integration
- NLP Enrichi sentiment badges
- Pre-Spike Radar alerts
- Repeat Gainer badges

Architecture V6 Anticipation Multi-Layer
"""

import requests
from typing import Dict, Any, Optional, List

from utils.logger import get_logger
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger("TELEGRAM_ALERTS")

TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


# ============================
# V6 EVENT TYPE TAXONOMY
# ============================

EVENT_TYPE_EMOJI = {
    # TIER 1 - CRITICAL IMPACT (0.90-1.00)
    "FDA_APPROVAL": "\U0001F3C6",       # Trophy
    "PDUFA_DECISION": "\U0001F4C5",     # Calendar
    "BUYOUT_CONFIRMED": "\U0001F4B0",   # Money bag

    # TIER 2 - HIGH IMPACT (0.75-0.89)
    "FDA_TRIAL_POSITIVE": "\U00002705", # Green check
    "BREAKTHROUGH_DESIGNATION": "\U0001F31F", # Star
    "FDA_FAST_TRACK": "\U0001F680",     # Rocket
    "MERGER_ACQUISITION": "\U0001F91D", # Handshake
    "EARNINGS_BEAT_BIG": "\U0001F4C8",  # Chart up
    "MAJOR_CONTRACT": "\U0001F4DD",     # Contract

    # TIER 3 - MODERATE IMPACT (0.60-0.74)
    "GUIDANCE_RAISE": "\U0001F4CA",     # Bar chart
    "EARNINGS_BEAT": "\U0001F4B5",      # Dollar
    "PARTNERSHIP": "\U0001F517",        # Link
    "PRICE_TARGET_RAISE": "\U0001F3AF", # Target

    # TIER 4 - LOW-MODERATE IMPACT (0.45-0.59)
    "ANALYST_UPGRADE": "\U0001F4DD",    # Note
    "SHORT_SQUEEZE_SIGNAL": "\U0001F4A5", # Collision
    "UNUSUAL_VOLUME_NEWS": "\U0001F4CA", # Chart

    # TIER 5 - SPECULATIVE (0.30-0.44)
    "BUYOUT_RUMOR": "\U0001F914",       # Thinking
    "SOCIAL_MEDIA_SURGE": "\U0001F4F1", # Phone
    "BREAKING_POSITIVE": "\U0001F4F0",  # Newspaper

    # Legacy mappings
    "earnings": "\U0001F4B5",
    "fda": "\U0001F3E5",
    "default": "\U0001F4E2"             # Megaphone
}

TIER_LABELS = {
    1: "\U0001F534 TIER 1 - CRITICAL",    # Red circle
    2: "\U0001F7E0 TIER 2 - HIGH",        # Orange circle
    3: "\U0001F7E1 TIER 3 - MODERATE",    # Yellow circle
    4: "\U0001F7E2 TIER 4 - LOW-MOD",     # Green circle
    5: "\U0001F535 TIER 5 - SPECULATIVE"  # Blue circle
}

# Impact to tier mapping
IMPACT_TO_TIER = {
    (0.90, 1.00): 1,
    (0.75, 0.89): 2,
    (0.60, 0.74): 3,
    (0.45, 0.59): 4,
    (0.30, 0.44): 5
}


# ============================
# Helper Functions
# ============================

def get_event_emoji(event_type: str) -> str:
    """Get emoji for event type"""
    return EVENT_TYPE_EMOJI.get(event_type, EVENT_TYPE_EMOJI["default"])


def get_tier_from_impact(impact: float) -> int:
    """Get tier number from impact score"""
    if impact >= 0.90:
        return 1
    elif impact >= 0.75:
        return 2
    elif impact >= 0.60:
        return 3
    elif impact >= 0.45:
        return 4
    else:
        return 5


def get_tier_label(tier: int) -> str:
    """Get tier label with emoji"""
    return TIER_LABELS.get(tier, TIER_LABELS[5])


def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for NLP sentiment"""
    sentiment_map = {
        "VERY_BULLISH": "\U0001F680\U0001F680",    # Double rocket
        "BULLISH": "\U0001F680",                   # Rocket
        "SLIGHTLY_BULLISH": "\U0001F4C8",          # Chart up
        "NEUTRAL": "\U00002796",                   # Minus
        "SLIGHTLY_BEARISH": "\U0001F4C9",          # Chart down
        "BEARISH": "\U0001F4A9",                   # Poop
        "VERY_BEARISH": "\U0001F4A9\U0001F4A9"     # Double poop
    }
    return sentiment_map.get(sentiment, "\U00002753")  # Question mark


def get_signal_emoji(signal_type: str) -> str:
    """Get emoji for signal type"""
    signal_map = {
        "BUY_STRONG": "\U0001F525\U0001F680",     # Fire + Rocket
        "BUY": "\U00002705",                       # Green check
        "WATCH_EARLY": "\U0001F440",              # Eyes
        "WATCH": "\U0001F50D",                     # Magnifying glass
        "AVOID": "\U0001F6AB"                      # No entry
    }
    return signal_map.get(signal_type, "\U0001F4E2")


# ============================
# Send message
# ============================

def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send message to Telegram

    Args:
        text: Message text
        parse_mode: "Markdown" or "HTML"

    Returns:
        Success status
    """
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode
        }

        r = requests.post(TELEGRAM_URL, json=payload, timeout=5)

        if r.status_code != 200:
            logger.warning(f"Telegram error: {r.text}")
            return False

        return True

    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ============================
# V6 Signal Alert (Enhanced)
# ============================

def send_signal_alert(
    signal: Dict[str, Any],
    position: Optional[Dict[str, Any]] = None,
    v6_data: Optional[Dict[str, Any]] = None
):
    """
    Send enhanced V6 signal alert

    Args:
        signal: Signal data (ticker, signal, monster_score, confidence)
        position: Optional position data (entry, stop, shares, risk_amount)
        v6_data: Optional V6 enrichment data:
            - event_type: EVENT_TYPE from unified taxonomy
            - event_impact: Impact score 0-1
            - catalyst_score: Catalyst Score V3 value
            - nlp_sentiment: NLP Enrichi sentiment direction
            - pre_spike_signals: Pre-Spike Radar signals count
            - repeat_gainer: Boolean if repeat gainer
            - repeat_gainer_spikes: Number of past spikes
    """
    ticker = signal.get('ticker', 'UNKNOWN')
    signal_type = signal.get('signal', signal.get('signal_type', 'BUY'))
    monster_score = signal.get('monster_score', 0)
    confidence = signal.get('confidence', 0)

    # Signal emoji
    signal_emoji = get_signal_emoji(signal_type)

    # Build header
    msg = f"""
{signal_emoji} *GV2-EDGE V6.0 SIGNAL*

\U0001F4CA Ticker: `{ticker}`
\U000026A1 Signal: *{signal_type}*
\U0001F3AF Monster Score: `{monster_score:.2f}`
\U0001F4C8 Confidence: `{confidence:.2f}`
"""

    # V6 enrichments
    if v6_data:
        event_type = v6_data.get('event_type')
        event_impact = v6_data.get('event_impact', 0)
        catalyst_score = v6_data.get('catalyst_score')
        nlp_sentiment = v6_data.get('nlp_sentiment')
        pre_spike_signals = v6_data.get('pre_spike_signals', 0)
        repeat_gainer = v6_data.get('repeat_gainer', False)
        repeat_gainer_spikes = v6_data.get('repeat_gainer_spikes', 0)

        msg += "\n*--- V6 Intelligence ---*\n"

        # Event type with tier
        if event_type:
            event_emoji = get_event_emoji(event_type)
            tier = get_tier_from_impact(event_impact)
            tier_label = get_tier_label(tier)
            msg += f"{event_emoji} Event: `{event_type}`\n"
            msg += f"\U0001F3C5 {tier_label} (impact: {event_impact:.2f})\n"

        # Catalyst Score V3
        if catalyst_score and catalyst_score > 0:
            msg += f"\U0001F9EA Catalyst Score V3: `{catalyst_score:.2f}`\n"

        # NLP Sentiment
        if nlp_sentiment:
            sentiment_emoji = get_sentiment_emoji(nlp_sentiment)
            msg += f"{sentiment_emoji} NLP Sentiment: `{nlp_sentiment}`\n"

        # Pre-Spike Radar
        if pre_spike_signals > 0:
            radar_level = "\U0001F534" if pre_spike_signals >= 3 else "\U0001F7E1"
            msg += f"{radar_level} Pre-Spike Radar: `{pre_spike_signals}/4` signals\n"

        # Repeat Gainer badge
        if repeat_gainer:
            msg += f"\U0001F501 REPEAT GAINER ({repeat_gainer_spikes} past spikes)\n"

    # Position info
    if position:
        msg += f"""
*--- Position ---*
\U0001F4B0 Entry: `{position.get('entry', 'N/A')}`
\U0001F6D1 Stop: `{position.get('stop', 'N/A')}`
\U0001F4E6 Shares: `{position.get('shares', 'N/A')}`
\U00002696 Risk: `${position.get('risk_amount', 'N/A')}`
\U0001F551 Session: `{position.get('session', 'N/A')}`
"""

    send_message(msg)


# ============================
# Pre-Spike Radar Alert
# ============================

def send_pre_spike_alert(
    ticker: str,
    signals_count: int,
    signals_detail: Dict[str, bool],
    acceleration_score: float,
    monster_score: float
):
    """
    Send Pre-Spike Radar alert when multiple signals detected

    Args:
        ticker: Stock ticker
        signals_count: Number of signals (1-4)
        signals_detail: Dict of signal name -> active status
        acceleration_score: Overall acceleration score
        monster_score: Current monster score
    """
    # Alert level based on signals
    if signals_count >= 4:
        alert_level = "\U0001F534\U0001F534\U0001F534 IMMINENT"  # Red x3
    elif signals_count >= 3:
        alert_level = "\U0001F534\U0001F534 HIGH"              # Red x2
    elif signals_count >= 2:
        alert_level = "\U0001F7E0 MODERATE"                    # Orange
    else:
        alert_level = "\U0001F7E1 WATCH"                       # Yellow

    msg = f"""
\U0001F4E1 *PRE-SPIKE RADAR ALERT*

\U0001F4CA Ticker: `{ticker}`
{alert_level} ({signals_count}/4 signals)

*Active Signals:*
"""

    # Signal details with checkmarks
    signal_names = {
        'volume_acceleration': '\U0001F4C8 Volume Acceleration',
        'bid_ask_tightening': '\U0001F4B1 Bid-Ask Tightening',
        'price_compression': '\U0001F5DC Price Compression',
        'dark_pool_activity': '\U0001F5A5 Dark Pool Activity'
    }

    for signal_key, display_name in signal_names.items():
        is_active = signals_detail.get(signal_key, False)
        status = "\U00002705" if is_active else "\U0000274C"
        msg += f"{status} {display_name}\n"

    msg += f"""
\U0001F680 Acceleration Score: `{acceleration_score:.2f}`
\U0001F3AF Monster Score: `{monster_score:.2f}`

\U000023F0 *ACTION: Monitor closely for entry*
"""

    send_message(msg)


# ============================
# Catalyst Alert
# ============================

def send_catalyst_alert(
    ticker: str,
    catalyst_type: str,
    catalyst_score: float,
    source: str,
    headline: str,
    tier: int,
    confluence_count: int = 1
):
    """
    Send Catalyst Score V3 alert for significant catalysts

    Args:
        ticker: Stock ticker
        catalyst_type: Type from unified taxonomy
        catalyst_score: Calculated catalyst score
        source: News source
        headline: News headline
        tier: Catalyst tier (1-5)
        confluence_count: Number of concurrent catalysts
    """
    event_emoji = get_event_emoji(catalyst_type)
    tier_label = get_tier_label(tier)

    # Urgency based on tier
    urgency_map = {
        1: "\U0001F6A8 CRITICAL CATALYST",
        2: "\U0001F525 HOT CATALYST",
        3: "\U0001F4E2 CATALYST DETECTED",
        4: "\U0001F4E1 MINOR CATALYST",
        5: "\U0001F50D SPECULATIVE CATALYST"
    }
    urgency = urgency_map.get(tier, urgency_map[5])

    msg = f"""
{event_emoji} *{urgency}*

\U0001F4CA Ticker: `{ticker}`
\U0001F3F7 Type: `{catalyst_type}`
{tier_label}

\U0001F9EA Catalyst Score: `{catalyst_score:.2f}`
"""

    if confluence_count > 1:
        msg += f"\U0001F4A5 CONFLUENCE: `{confluence_count}` concurrent catalysts!\n"

    msg += f"""
\U0001F4F0 Source: `{source}`
\U0001F4DD _{headline[:100]}..._
"""

    send_message(msg)


# ============================
# NLP Sentiment Alert
# ============================

def send_nlp_sentiment_alert(
    ticker: str,
    sentiment_direction: str,
    sentiment_score: float,
    news_count: int,
    dominant_category: str,
    urgency_level: str
):
    """
    Send NLP Enrichi sentiment summary alert

    Args:
        ticker: Stock ticker
        sentiment_direction: Sentiment direction from NLP Enrichi
        sentiment_score: Aggregated sentiment score
        news_count: Number of news analyzed
        dominant_category: Most common news category
        urgency_level: NEWS urgency level
    """
    sentiment_emoji = get_sentiment_emoji(sentiment_direction)

    # Urgency emoji
    urgency_emojis = {
        "BREAKING": "\U0001F6A8",
        "HIGH": "\U0001F534",
        "MODERATE": "\U0001F7E0",
        "NORMAL": "\U0001F7E2",
        "LOW": "\U0001F535"
    }
    urgency_emoji = urgency_emojis.get(urgency_level, "\U0001F7E2")

    msg = f"""
{sentiment_emoji} *NLP SENTIMENT ALERT*

\U0001F4CA Ticker: `{ticker}`
\U0001F9E0 Sentiment: *{sentiment_direction}*
\U0001F4CA Score: `{sentiment_score:.2f}`

{urgency_emoji} Urgency: `{urgency_level}`
\U0001F4F0 News Analyzed: `{news_count}`
\U0001F3F7 Category: `{dominant_category}`
"""

    send_message(msg)


# ============================
# Repeat Gainer Alert
# ============================

def send_repeat_gainer_alert(
    ticker: str,
    past_spikes: int,
    avg_spike_pct: float,
    last_spike_date: str,
    volatility_score: float,
    monster_score: float
):
    """
    Send Repeat Gainer Memory alert

    Args:
        ticker: Stock ticker
        past_spikes: Number of historical spikes
        avg_spike_pct: Average spike percentage
        last_spike_date: Date of last spike
        volatility_score: Current volatility score from memory
        monster_score: Current monster score
    """
    # Badge level based on past spikes
    if past_spikes >= 5:
        badge = "\U0001F451 SERIAL RUNNER"      # Crown
    elif past_spikes >= 3:
        badge = "\U0001F525 HOT REPEAT"         # Fire
    else:
        badge = "\U0001F501 KNOWN MOVER"        # Repeat

    msg = f"""
\U0001F501 *REPEAT GAINER DETECTED*

\U0001F4CA Ticker: `{ticker}`
{badge}

\U0001F4C8 Historical Spikes: `{past_spikes}`
\U0001F4B9 Avg Spike: `+{avg_spike_pct:.1f}%`
\U0001F4C5 Last Spike: `{last_spike_date}`
\U0001F30B Volatility Score: `{volatility_score:.2f}`

\U0001F3AF Current Monster Score: `{monster_score:.2f}`

\U000026A0 *Known for explosive moves - size appropriately*
"""

    send_message(msg)


# ============================
# Daily Audit Summary Alert
# ============================

def send_daily_audit_alert(report: Dict[str, Any]):
    """
    Send daily audit summary via Telegram

    Args:
        report: Audit report dict with summary, hit_analysis, miss_analysis, etc.
    """
    summary = report.get("summary", {})
    grade = report.get("performance_grade", "?")
    audit_date = report.get("audit_date", "N/A")

    # Grade emojis
    grade_emoji = {
        "A": "\U0001F3C6",  # Trophy
        "B": "\U00002705",  # Check
        "C": "\U000026A0", # Warning
        "D": "\U0001F7E0",  # Orange
        "F": "\U0000274C"   # X
    }

    hit_rate = summary.get('hit_rate', 0)
    early_catch = summary.get('early_catch_rate', 0)
    miss_rate = summary.get('miss_rate', 0)
    fp_count = summary.get('fp_count', 0)
    avg_lead = summary.get('avg_lead_time_hours', 0)

    msg = f"""
\U0001F4CA *GV2-EDGE V6.0 DAILY AUDIT*
\U0001F4C5 Date: `{audit_date}`

{grade_emoji.get(grade, '\U00002753')} *Performance Grade: {grade}*

*--- Core Metrics ---*
\U0001F4C8 Hit Rate: `{hit_rate*100:.1f}%`
\U000023F1 Early Catches: `{early_catch*100:.1f}%`
\U000023F0 Avg Lead Time: `{avg_lead:.1f}h`
\U0000274C Miss Rate: `{miss_rate*100:.1f}%`
\U0001F3AF False Positives: `{fp_count}`
"""

    # V6 Module Stats if available
    v6_stats = report.get("v6_stats", {})
    if v6_stats:
        msg += "\n*--- V6 Module Performance ---*\n"

        if 'catalyst_v3_contribution' in v6_stats:
            msg += f"\U0001F9EA Catalyst V3: `{v6_stats['catalyst_v3_contribution']:.1f}%` of hits\n"

        if 'pre_spike_accuracy' in v6_stats:
            msg += f"\U0001F4E1 Pre-Spike Radar: `{v6_stats['pre_spike_accuracy']:.1f}%` accuracy\n"

        if 'repeat_gainer_hits' in v6_stats:
            msg += f"\U0001F501 Repeat Gainer Hits: `{v6_stats['repeat_gainer_hits']}`\n"

        if 'nlp_sentiment_boost' in v6_stats:
            msg += f"\U0001F9E0 NLP Sentiment Avg Boost: `+{v6_stats['nlp_sentiment_boost']:.2f}`\n"

    # Top hits
    hit_analysis = report.get("hit_analysis", {})
    hits = hit_analysis.get("hits", [])[:3]
    if hits:
        msg += "\n*\U0001F3AF TOP HITS:*\n"
        for hit in hits:
            msg += f"  \U00002022 `{hit['ticker']}`: +{hit.get('gainer_change_pct', 0):.0f}% (lead: {hit.get('lead_time_hours', 0):.1f}h)\n"

    # Top misses
    miss_analysis = report.get("miss_analysis", {})
    misses = miss_analysis.get("missed_tickers", [])[:3]
    if misses:
        msg += f"\n*\U0000274C TOP MISSES:* {', '.join(misses)}"

    send_message(msg)


# ============================
# Weekly Audit Summary Alert
# ============================

def send_weekly_audit_alert(report: Dict[str, Any]):
    """
    Send weekly audit summary via Telegram

    Args:
        report: Weekly audit report dict
    """
    period = report.get("period", {})
    metrics = report.get("metrics", {})
    recommendations = report.get("recommendations", [])

    trend = metrics.get("trend", "stable")
    trend_emoji = "\U0001F4C8" if trend == "improving" else "\U0001F4C9" if trend == "declining" else "\U00002796"

    msg = f"""
\U0001F4CA *GV2-EDGE V6.0 WEEKLY AUDIT*
\U0001F4C5 Period: `{period.get('start', 'N/A')}` to `{period.get('end', 'N/A')}`
\U0001F4C6 Days with data: `{period.get('days_with_data', 0)}`

*--- Weekly Averages ---*
\U0001F4C8 Avg Hit Rate: `{metrics.get('avg_hit_rate', 0)*100:.1f}%`
\U000023F1 Avg Early Catch: `{metrics.get('avg_early_catch_rate', 0)*100:.1f}%`
\U000023F0 Avg Lead Time: `{metrics.get('avg_lead_time_hours', 0):.1f}h`
\U0000274C Avg Miss Rate: `{metrics.get('avg_miss_rate', 0)*100:.1f}%`
\U0001F3AF Total FPs: `{metrics.get('total_false_positives', 0)}`

{trend_emoji} *Trend: {trend.upper()}*
"""

    # Daily grades
    daily_grades = report.get("daily_grades", [])
    if daily_grades:
        grades_str = " ".join(daily_grades)
        msg += f"\n\U0001F4CA Daily Grades: `{grades_str}`\n"

    # Recommendations
    if recommendations:
        msg += "\n*\U0001F4A1 RECOMMENDATIONS:*\n"
        for rec in recommendations[:3]:
            action = rec.get('action', '').upper()
            component = rec.get('component', '')
            reason = rec.get('reason', '')
            msg += f"  \U00002022 {action} `{component}`: _{reason}_\n"

    send_message(msg)


# ============================
# System Alert
# ============================

def send_system_alert(text: str, level: str = "info"):
    """
    Send system alert

    Args:
        text: Alert message
        level: "info", "warning", "error", "critical"
    """
    level_config = {
        "info": ("\U0001F535", "INFO"),
        "warning": ("\U000026A0", "WARNING"),
        "error": ("\U0001F534", "ERROR"),
        "critical": ("\U0001F6A8", "CRITICAL")
    }

    emoji, label = level_config.get(level, level_config["info"])

    msg = f"{emoji} *GV2-EDGE V6.0 {label}*\n\n{text}"
    send_message(msg)


# ============================
# Market Session Alert
# ============================

def send_session_alert(session: str, active_signals: int = 0):
    """
    Send market session transition alert

    Args:
        session: Session name (PREMARKET, RTH, AFTER_HOURS, CLOSED)
        active_signals: Number of active signals
    """
    session_config = {
        "PREMARKET": ("\U0001F305", "Pre-Market Open", "Scanning for overnight catalysts..."),
        "RTH": ("\U0001F4C8", "Market Open", "Regular trading hours active"),
        "AFTER_HOURS": ("\U0001F319", "After-Hours", "Extended hours monitoring"),
        "CLOSED": ("\U0001F4A4", "Market Closed", "System in standby mode")
    }

    emoji, label, desc = session_config.get(session, ("\U00002753", session, ""))

    msg = f"""
{emoji} *{label}*

{desc}
\U0001F4CA Active Signals: `{active_signals}`
"""

    send_message(msg)


# ============================
# Test Connection
# ============================

if __name__ == "__main__":
    # Test basic connection
    send_message("\U00002705 GV2-EDGE V6.0 Telegram connected")

    # Test V6 signal alert
    test_signal = {
        "ticker": "TEST",
        "signal": "BUY_STRONG",
        "monster_score": 0.85,
        "confidence": 0.92
    }

    test_v6_data = {
        "event_type": "FDA_APPROVAL",
        "event_impact": 0.95,
        "catalyst_score": 0.88,
        "nlp_sentiment": "VERY_BULLISH",
        "pre_spike_signals": 3,
        "repeat_gainer": True,
        "repeat_gainer_spikes": 4
    }

    send_signal_alert(test_signal, v6_data=test_v6_data)
    print("Test alerts sent!")
