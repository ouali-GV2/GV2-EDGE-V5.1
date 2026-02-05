# ============================
# US MARKET CALENDAR
# ============================
# Handles NYSE holidays, early closes, and trading day logic
# Part of GV2-EDGE V6 - Market Calendar Layer

from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple
import pytz

from utils.logger import get_logger

logger = get_logger("MARKET_CALENDAR")

US_TZ = pytz.timezone("US/Eastern")


# ============================
# NYSE HOLIDAYS (2024-2027)
# ============================
# Source: NYSE official calendar
# Updated annually

NYSE_HOLIDAYS = {
    # 2024
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # MLK Day
    date(2024, 2, 19),   # Presidents Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas

    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas

    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas

    # 2027
    date(2027, 1, 1),    # New Year's Day
    date(2027, 1, 18),   # MLK Day
    date(2027, 2, 15),   # Presidents Day
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day
    date(2027, 6, 18),   # Juneteenth (observed)
    date(2027, 7, 5),    # Independence Day (observed)
    date(2027, 9, 6),    # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed)
}


# ============================
# EARLY CLOSE DAYS (1:00 PM ET)
# ============================
# NYSE closes at 1:00 PM ET on these days

NYSE_EARLY_CLOSES = {
    # 2024
    date(2024, 7, 3),    # Day before Independence Day
    date(2024, 11, 29),  # Day after Thanksgiving
    date(2024, 12, 24),  # Christmas Eve

    # 2025
    date(2025, 7, 3),    # Day before Independence Day
    date(2025, 11, 28),  # Day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve

    # 2026
    date(2026, 7, 2),    # Day before Independence Day (observed)
    date(2026, 11, 27),  # Day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve

    # 2027
    date(2027, 7, 2),    # Day before Independence Day (observed)
    date(2027, 11, 26),  # Day after Thanksgiving
    date(2027, 12, 23),  # Day before Christmas (observed)
}


# ============================
# CORE FUNCTIONS
# ============================

def is_nyse_holiday(dt: Optional[date] = None) -> bool:
    """
    Check if date is a NYSE holiday

    Args:
        dt: Date to check (default: today US/Eastern)

    Returns:
        True if holiday
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    return dt in NYSE_HOLIDAYS


def is_early_close(dt: Optional[date] = None) -> bool:
    """
    Check if date is an early close day (1:00 PM ET)

    Args:
        dt: Date to check (default: today US/Eastern)

    Returns:
        True if early close day
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    return dt in NYSE_EARLY_CLOSES


def is_trading_day(dt: Optional[date] = None) -> bool:
    """
    Check if date is a valid trading day (not weekend, not holiday)

    Args:
        dt: Date to check (default: today US/Eastern)

    Returns:
        True if trading day
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    # Check weekend (Mon=0, Sun=6)
    if dt.weekday() >= 5:
        return False

    # Check holiday
    if is_nyse_holiday(dt):
        return False

    return True


def get_market_close_time(dt: Optional[date] = None) -> Tuple[int, int]:
    """
    Get market close time for a given date

    Args:
        dt: Date to check (default: today US/Eastern)

    Returns:
        Tuple (hour, minute) in ET - (16, 0) normal or (13, 0) early close
    """
    if is_early_close(dt):
        return (13, 0)  # 1:00 PM ET
    return (16, 0)  # 4:00 PM ET


def get_previous_trading_day(dt: Optional[date] = None) -> date:
    """
    Get the previous trading day

    Args:
        dt: Reference date (default: today US/Eastern)

    Returns:
        Previous trading day
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    prev_day = dt - timedelta(days=1)

    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)

    return prev_day


def get_next_trading_day(dt: Optional[date] = None) -> date:
    """
    Get the next trading day

    Args:
        dt: Reference date (default: today US/Eastern)

    Returns:
        Next trading day
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    next_day = dt + timedelta(days=1)

    while not is_trading_day(next_day):
        next_day += timedelta(days=1)

    return next_day


def trading_days_between(start: date, end: date) -> int:
    """
    Count trading days between two dates (exclusive of end)

    Args:
        start: Start date
        end: End date

    Returns:
        Number of trading days
    """
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    count = 0
    current = start

    while current < end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)

    return count


def get_trading_days_list(start: date, end: date) -> List[date]:
    """
    Get list of trading days between two dates (inclusive)

    Args:
        start: Start date
        end: End date

    Returns:
        List of trading day dates
    """
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    days = []
    current = start

    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)

    return days


def days_to_next_trading_day(dt: Optional[date] = None) -> int:
    """
    Get number of calendar days until next trading day

    Useful for weekend/holiday detection

    Args:
        dt: Reference date (default: today US/Eastern)

    Returns:
        Number of calendar days (0 if today is trading day and market not closed)
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    if is_trading_day(dt):
        return 0

    next_trading = get_next_trading_day(dt)
    return (next_trading - dt).days


def get_holiday_name(dt: date) -> Optional[str]:
    """
    Get the name of a NYSE holiday

    Args:
        dt: Date to check

    Returns:
        Holiday name or None
    """
    if isinstance(dt, datetime):
        dt = dt.date()

    # Holiday names mapping
    holiday_names = {
        1: {1: "New Year's Day"},
        6: {19: "Juneteenth"},
        7: {4: "Independence Day", 3: "Independence Day (observed)", 5: "Independence Day (observed)"},
        9: {1: "Labor Day", 2: "Labor Day", 6: "Labor Day", 7: "Labor Day"},
        11: {25: "Thanksgiving", 26: "Thanksgiving", 27: "Thanksgiving", 28: "Thanksgiving"},
        12: {24: "Christmas Eve", 25: "Christmas"},
    }

    # Check MLK Day (3rd Monday of January)
    if dt.month == 1 and dt.weekday() == 0 and 15 <= dt.day <= 21:
        return "Martin Luther King Jr. Day"

    # Check Presidents Day (3rd Monday of February)
    if dt.month == 2 and dt.weekday() == 0 and 15 <= dt.day <= 21:
        return "Presidents Day"

    # Check Memorial Day (last Monday of May)
    if dt.month == 5 and dt.weekday() == 0 and dt.day >= 25:
        return "Memorial Day"

    # Check Good Friday (varies)
    if dt in NYSE_HOLIDAYS and dt.month in [3, 4]:
        return "Good Friday"

    # Check static holidays
    if dt.month in holiday_names:
        if dt.day in holiday_names[dt.month]:
            return holiday_names[dt.month][dt.day]

    if dt in NYSE_HOLIDAYS:
        return "NYSE Holiday"

    return None


# ============================
# VOLUME ADJUSTMENT HELPERS
# ============================

def get_volume_adjustment_factor(dt: Optional[date] = None) -> float:
    """
    Get volume adjustment factor for a trading day

    Used to normalize volumes when comparing across different days:
    - Early close days have lower volume (factor < 1)
    - Days before holidays often have lower volume
    - Friday afternoons have lower volume

    Args:
        dt: Date to check (default: today US/Eastern)

    Returns:
        Adjustment factor (1.0 = normal, <1.0 = expect lower volume)
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    factor = 1.0

    # Early close day - significantly lower volume
    if is_early_close(dt):
        factor *= 0.6

    # Friday - slightly lower afternoon volume
    elif dt.weekday() == 4:
        factor *= 0.9

    # Day before holiday - often lower volume
    next_day = dt + timedelta(days=1)
    if is_nyse_holiday(next_day):
        factor *= 0.75

    return factor


def get_comparable_trading_day(dt: Optional[date] = None) -> date:
    """
    Get a comparable trading day for volume comparison

    Returns a similar day from the previous week that has similar
    characteristics (same weekday, not holiday-adjacent)

    Args:
        dt: Reference date (default: today US/Eastern)

    Returns:
        Comparable trading day
    """
    if dt is None:
        dt = datetime.now(US_TZ).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    # Go back 7 days (same weekday)
    comparable = dt - timedelta(days=7)

    # If that's not a trading day, find nearest
    if not is_trading_day(comparable):
        comparable = get_previous_trading_day(comparable + timedelta(days=1))

    return comparable


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    print("\n📅 US MARKET CALENDAR TEST")
    print("=" * 50)

    today = datetime.now(US_TZ).date()
    print(f"\nToday: {today} ({today.strftime('%A')})")
    print(f"Is trading day: {is_trading_day(today)}")
    print(f"Is holiday: {is_nyse_holiday(today)}")
    print(f"Is early close: {is_early_close(today)}")

    if is_nyse_holiday(today):
        print(f"Holiday name: {get_holiday_name(today)}")

    print(f"\nMarket close time: {get_market_close_time(today)}")
    print(f"Previous trading day: {get_previous_trading_day(today)}")
    print(f"Next trading day: {get_next_trading_day(today)}")
    print(f"Days to next trading day: {days_to_next_trading_day(today)}")
    print(f"Volume adjustment factor: {get_volume_adjustment_factor(today):.2f}")

    # Show upcoming holidays
    print("\n📆 Upcoming NYSE Holidays:")
    upcoming = sorted([h for h in NYSE_HOLIDAYS if h >= today])[:5]
    for h in upcoming:
        name = get_holiday_name(h) or "Holiday"
        print(f"  {h} ({h.strftime('%A')}) - {name}")

    # Show upcoming early closes
    print("\n⏰ Upcoming Early Closes (1:00 PM):")
    upcoming_early = sorted([e for e in NYSE_EARLY_CLOSES if e >= today])[:3]
    for e in upcoming_early:
        print(f"  {e} ({e.strftime('%A')})")

    # Trading days this week
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    trading_days = get_trading_days_list(week_start, week_end)
    print(f"\n📊 Trading days this week: {len(trading_days)}")
    for d in trading_days:
        marker = " ←" if d == today else ""
        early = " (early close)" if is_early_close(d) else ""
        print(f"  {d} ({d.strftime('%A')}){early}{marker}")
