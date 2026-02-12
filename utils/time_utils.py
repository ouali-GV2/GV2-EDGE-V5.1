from datetime import datetime, time, timedelta
import pytz

from config import (
    PREMARKET_START,
    MARKET_OPEN,
    MARKET_CLOSE,
    AFTER_HOURS_END
)

# Import market calendar for holiday/early close support
from utils.market_calendar import (
    is_trading_day as _is_trading_day_calendar,
    is_early_close,
    is_nyse_holiday,
    get_market_close_time,
    get_previous_trading_day,
    get_next_trading_day,
    days_to_next_trading_day,
    get_volume_adjustment_factor
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


def _get_market_close_time_today():
    """Get today's market close time (handles early close days)"""
    hour, minute = get_market_close_time()
    return time(hour, minute)


def is_premarket(now=None):
    """Check if currently in pre-market session (handles holidays)"""
    now_dt = now_us()
    now_time = now or now_dt.time()

    # Not a trading day = no pre-market
    if not _is_trading_day_calendar(now_dt.date()):
        return False

    return PREMARKET_START_T <= now_time < MARKET_OPEN_T


def is_market_open(now=None):
    """Check if market is currently open (handles early close days)"""
    now_dt = now_us()
    now_time = now or now_dt.time()

    # Not a trading day = market not open
    if not _is_trading_day_calendar(now_dt.date()):
        return False

    # Get today's close time (may be early close)
    close_time = _get_market_close_time_today()

    return MARKET_OPEN_T <= now_time < close_time


def is_after_hours(now=None):
    """Check if currently in after-hours session (handles early close days)"""
    now_dt = now_us()
    now_time = now or now_dt.time()

    # Not a trading day = no after-hours
    if not _is_trading_day_calendar(now_dt.date()):
        return False

    # Get today's close time (may be early close)
    close_time = _get_market_close_time_today()

    # After-hours starts at close and ends at AFTER_HOURS_END
    # On early close days, AH is longer (1 PM - 8 PM)
    return close_time <= now_time < AFTER_HOURS_END_T


def is_market_closed(now=None):
    """Check if market is completely closed (not PM, not RTH, not AH)"""
    now_dt = now_us()
    now_time = now or now_dt.time()

    # Holiday or weekend = closed
    if not _is_trading_day_calendar(now_dt.date()):
        return True

    return now_time < PREMARKET_START_T or now_time >= AFTER_HOURS_END_T


def market_session():
    """Return current session name (handles holidays and early close)"""
    now_dt = now_us()
    now_time = now_dt.time()

    # Check if trading day first
    if not _is_trading_day_calendar(now_dt.date()):
        if is_nyse_holiday(now_dt.date()):
            return "HOLIDAY"
        return "WEEKEND"

    if is_premarket(now_time):
        return "PREMARKET"
    elif is_market_open(now_time):
        if is_early_close(now_dt.date()):
            return "RTH_EARLY"
        return "RTH"
    elif is_after_hours(now_time):
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
        "RTH_EARLY": "RTH",
        "AFTER_HOURS": "POST",
        "CLOSED": "CLOSED",
        "HOLIDAY": "CLOSED",
        "WEEKEND": "CLOSED"
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
    """Minutes before market close (handles early close days)"""
    now = now_us()
    close_hour, close_min = get_market_close_time(now.date())
    close_dt = now.replace(
        hour=close_hour,
        minute=close_min,
        second=0,
        microsecond=0
    )
    return max(0, int((close_dt - now).total_seconds() / 60))


def is_trading_day(dt=None):
    """Check if date is a trading day (not weekend, not holiday)"""
    if dt is None:
        dt = now_us()
    return _is_trading_day_calendar(dt)


def next_market_open():
    """Get datetime of next market open (handles holidays)"""
    now = now_us()

    # Start with today's open time
    next_open = now.replace(
        hour=MARKET_OPEN_T.hour,
        minute=MARKET_OPEN_T.minute,
        second=0,
        microsecond=0
    )

    # If we're past today's open, start from tomorrow
    if now.time() >= MARKET_OPEN_T:
        next_open += timedelta(days=1)

    # Find next trading day
    while not _is_trading_day_calendar(next_open.date()):
        next_open += timedelta(days=1)

    return next_open
