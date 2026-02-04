from datetime import datetime, time, timedelta
import pytz

from config import (
    PREMARKET_START,
    MARKET_OPEN,
    MARKET_CLOSE,
    AFTER_HOURS_END
)

US_TZ = pytz.timezone("US/Eastern")


def now_us():
    """Current US market time"""
    return datetime.now(US_TZ)


def str_to_time(t):
    return datetime.strptime(t, "%H:%M").time()


PREMARKET_START_T = str_to_time(PREMARKET_START)
MARKET_OPEN_T = str_to_time(MARKET_OPEN)
MARKET_CLOSE_T = str_to_time(MARKET_CLOSE)
AFTER_HOURS_END_T = str_to_time(AFTER_HOURS_END)


def is_premarket(now=None):
    now = now or now_us().time()
    return PREMARKET_START_T <= now < MARKET_OPEN_T


def is_market_open(now=None):
    now = now or now_us().time()
    return MARKET_OPEN_T <= now < MARKET_CLOSE_T


def is_after_hours(now=None):
    now = now or now_us().time()
    return MARKET_CLOSE_T <= now < AFTER_HOURS_END_T


def is_market_closed(now=None):
    """Check if market is completely closed (not PM, not RTH, not AH)"""
    now = now or now_us().time()
    return now < PREMARKET_START_T or now >= AFTER_HOURS_END_T


def market_session():
    """Return current session name"""

    now = now_us().time()

    if is_premarket(now):
        return "PREMARKET"
    elif is_market_open(now):
        return "RTH"
    elif is_after_hours(now):
        return "AFTER_HOURS"
    else:
        return "CLOSED"


def get_market_session():
    """Alias for market_session() - returns PRE, RTH, POST, or CLOSED"""
    session = market_session()
    
    # Map to shorter names used by extended hours module
    mapping = {
        "PREMARKET": "PRE",
        "RTH": "RTH",
        "AFTER_HOURS": "POST",
        "CLOSED": "CLOSED"
    }
    
    return mapping.get(session, "CLOSED")


def minutes_since_open():
    now = now_us()
    open_dt = now.replace(
        hour=MARKET_OPEN_T.hour,
        minute=MARKET_OPEN_T.minute,
        second=0,
        microsecond=0
    )
    return max(0, int((now - open_dt).total_seconds() / 60))


def minutes_before_close():
    now = now_us()
    close_dt = now.replace(
        hour=MARKET_CLOSE_T.hour,
        minute=MARKET_CLOSE_T.minute,
        second=0,
        microsecond=0
    )
    return max(0, int((close_dt - now).total_seconds() / 60))


def is_trading_day():
    now = now_us()
    return now.weekday() < 5  # Monday=0 ... Friday=4


def next_market_open():
    now = now_us()

    next_open = now.replace(
        hour=MARKET_OPEN_T.hour,
        minute=MARKET_OPEN_T.minute,
        second=0,
        microsecond=0
    )

    if now.time() > MARKET_OPEN_T:
        next_open += timedelta(days=1)

    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    return next_open
