"""
IBKR NEWS TRIGGER V7.0
======================

Early alert system using IBKR News headlines.

ROLE:
- Trigger rapid scans (NOT scoring)
- Feed Pre-Spike Radar and Pre-Halt Engine
- Detect halt risks BEFORE they happen
- Never block detection, only trigger alerts

POSITION IN PIPELINE:
    Market Data
       ├─ Pre-Spike Radar     (OPPORTUNITY)
       ├─ Pre-Halt Engine     (RISK & TIMING)
       └─ IBKR News Trigger   (EARLY ALERTS) ← THIS MODULE
               ↓
            Signal Engine

IMPORTANT:
- IBKR News does NOT contribute to Monster Score
- It's a TRIGGER system, not a scoring system
- Keywords trigger actions, not scores
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# KEYWORD TRIGGERS (NOT FOR SCORING)
# ============================================================================

# VERY STRONG - Halt risk / Major catalyst
HALT_KEYWORDS = [
    "halt",
    "halted",
    "pending news",
    "news pending",
    "acquired",
    "acquisition",
    "buyout",
    "merger",
    "fda",
    "approval",
    "approved",
    "trial results",
    "phase 3",
    "phase iii",
    "contract",
    "guidance raised",
    "guidance increase",
    "crl",  # Complete Response Letter (FDA)
    "pdufa",
    "breakthrough",
    "fast track",
]

# MEDIUM - Suspicious acceleration
SPIKE_KEYWORDS = [
    "surge",
    "surges",
    "surging",
    "jump",
    "jumps",
    "jumping",
    "spike",
    "spikes",
    "spiking",
    "rally",
    "rallies",
    "rallying",
    "unusual volume",
    "heavy volume",
    "breaking",
    "soar",
    "soars",
    "soaring",
    "explode",
    "explodes",
    "moon",
    "skyrocket",
]

# NEGATIVE - Risk indicators
RISK_KEYWORDS = [
    "dilution",
    "offering",
    "shelf",
    "s-3",
    "prospectus",
    "reverse split",
    "delisting",
    "compliance",
    "deficiency",
    "going concern",
    "sec investigation",
    "subpoena",
    "fraud",
    "restatement",
]


class TriggerLevel(Enum):
    """Trigger urgency levels."""
    CRITICAL = "CRITICAL"     # Immediate action required
    HIGH = "HIGH"             # High priority scan
    MEDIUM = "MEDIUM"         # Normal priority scan
    LOW = "LOW"               # Monitor only
    NONE = "NONE"             # No trigger


class TriggerType(Enum):
    """Type of trigger detected."""
    HALT_RISK = "HALT_RISK"           # Potential halt
    CATALYST_MAJOR = "CATALYST_MAJOR" # Major positive catalyst
    SPIKE_FORMING = "SPIKE_FORMING"   # Price spike forming
    RISK_ALERT = "RISK_ALERT"         # Negative news
    MOMENTUM = "MOMENTUM"             # General momentum


@dataclass
class NewsTrigger:
    """Represents a news trigger event."""
    ticker: str
    headline: str
    trigger_level: TriggerLevel
    trigger_type: TriggerType
    keywords_matched: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    # Actions triggered
    trigger_hot_scan: bool = False      # Trigger immediate scan
    trigger_pre_halt: bool = False      # Alert Pre-Halt Engine
    trigger_pre_spike: bool = False     # Alert Pre-Spike Radar

    # Source info
    source: str = "ibkr_news"
    news_id: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.trigger_level.value}] {self.ticker}: {self.trigger_type.value} - {', '.join(self.keywords_matched)}"


@dataclass
class TriggerStats:
    """Statistics for trigger monitoring."""
    total_headlines_processed: int = 0
    triggers_generated: int = 0
    halt_triggers: int = 0
    catalyst_triggers: int = 0
    spike_triggers: int = 0
    risk_triggers: int = 0
    last_trigger_time: Optional[datetime] = None


class IBKRNewsTrigger:
    """
    IBKR News Trigger - Early Alert System

    Usage:
        trigger = IBKRNewsTrigger()

        # Process incoming headline
        result = trigger.process_headline(ticker, headline)

        if result.trigger_hot_scan:
            hot_ticker_queue.push(ticker, priority=HOT)

        if result.trigger_pre_halt:
            pre_halt_engine.alert(ticker, result)

        if result.trigger_pre_spike:
            pre_spike_radar.boost(ticker, result)

    IMPORTANT:
    - This does NOT score headlines for Monster Score
    - It only triggers actions (scans, alerts)
    - Keywords matched = action triggered, not score contribution
    """

    def __init__(self):
        # Compile regex patterns for efficiency
        self._halt_patterns = self._compile_patterns(HALT_KEYWORDS)
        self._spike_patterns = self._compile_patterns(SPIKE_KEYWORDS)
        self._risk_patterns = self._compile_patterns(RISK_KEYWORDS)

        # Recent triggers cache (avoid duplicates)
        self._recent_triggers: Dict[str, datetime] = {}
        self._trigger_cooldown = timedelta(minutes=5)

        # Statistics
        self.stats = TriggerStats()

        # Listeners for trigger events
        self._listeners: List[callable] = []

    def _compile_patterns(self, keywords: List[str]) -> List[re.Pattern]:
        """Compile keyword patterns for efficient matching."""
        patterns = []
        for kw in keywords:
            # Word boundary matching, case insensitive
            pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def process_headline(
        self,
        ticker: str,
        headline: str,
        source: str = "ibkr_news",
        news_id: Optional[str] = None
    ) -> Optional[NewsTrigger]:
        """
        Process a news headline and generate trigger if needed.

        Args:
            ticker: Stock ticker
            headline: News headline text
            source: News source identifier
            news_id: Unique news ID (for dedup)

        Returns:
            NewsTrigger if keywords matched, None otherwise
        """
        ticker = ticker.upper()
        self.stats.total_headlines_processed += 1

        # Check cooldown (avoid duplicate triggers)
        cache_key = f"{ticker}:{headline[:50]}"
        if cache_key in self._recent_triggers:
            last_trigger = self._recent_triggers[cache_key]
            if datetime.now() - last_trigger < self._trigger_cooldown:
                return None

        # Match keywords
        halt_matches = self._match_keywords(headline, self._halt_patterns, HALT_KEYWORDS)
        spike_matches = self._match_keywords(headline, self._spike_patterns, SPIKE_KEYWORDS)
        risk_matches = self._match_keywords(headline, self._risk_patterns, RISK_KEYWORDS)

        all_matches = halt_matches + spike_matches + risk_matches

        if not all_matches:
            return None

        # Determine trigger level and type
        trigger_level, trigger_type = self._classify_trigger(
            halt_matches, spike_matches, risk_matches
        )

        # Create trigger
        trigger = NewsTrigger(
            ticker=ticker,
            headline=headline,
            trigger_level=trigger_level,
            trigger_type=trigger_type,
            keywords_matched=all_matches,
            source=source,
            news_id=news_id
        )

        # Set actions based on trigger type
        self._set_trigger_actions(trigger)

        # Update cache and stats
        self._recent_triggers[cache_key] = datetime.now()
        self._update_stats(trigger)

        # Notify listeners
        self._notify_listeners(trigger)

        logger.info(f"📰 NEWS TRIGGER: {trigger}")

        return trigger

    def _match_keywords(
        self,
        text: str,
        patterns: List[re.Pattern],
        keywords: List[str]
    ) -> List[str]:
        """Match keywords in text and return matched keywords."""
        matches = []
        for pattern, keyword in zip(patterns, keywords):
            if pattern.search(text):
                matches.append(keyword)
        return matches

    def _classify_trigger(
        self,
        halt_matches: List[str],
        spike_matches: List[str],
        risk_matches: List[str]
    ) -> Tuple[TriggerLevel, TriggerType]:
        """Classify trigger level and type based on matches."""

        # CRITICAL: Halt keywords detected
        if halt_matches:
            # Check for specific high-urgency keywords
            critical_keywords = {"halt", "halted", "pending news", "news pending"}
            if any(kw in halt_matches for kw in critical_keywords):
                return TriggerLevel.CRITICAL, TriggerType.HALT_RISK

            # FDA/M&A = Major catalyst
            catalyst_keywords = {"fda", "approval", "approved", "merger", "acquired", "buyout"}
            if any(kw in halt_matches for kw in catalyst_keywords):
                return TriggerLevel.HIGH, TriggerType.CATALYST_MAJOR

            return TriggerLevel.HIGH, TriggerType.CATALYST_MAJOR

        # HIGH: Risk keywords (negative news)
        if risk_matches:
            # Dilution/offering = immediate risk
            if any(kw in risk_matches for kw in {"offering", "dilution", "shelf"}):
                return TriggerLevel.HIGH, TriggerType.RISK_ALERT
            return TriggerLevel.MEDIUM, TriggerType.RISK_ALERT

        # MEDIUM: Spike keywords
        if spike_matches:
            # Multiple spike keywords = higher priority
            if len(spike_matches) >= 2:
                return TriggerLevel.HIGH, TriggerType.SPIKE_FORMING
            return TriggerLevel.MEDIUM, TriggerType.SPIKE_FORMING

        return TriggerLevel.LOW, TriggerType.MOMENTUM

    def _set_trigger_actions(self, trigger: NewsTrigger) -> None:
        """Set action flags based on trigger classification."""

        # CRITICAL/HIGH → Trigger all systems
        if trigger.trigger_level in [TriggerLevel.CRITICAL, TriggerLevel.HIGH]:
            trigger.trigger_hot_scan = True

            # Halt risk → Alert Pre-Halt Engine
            if trigger.trigger_type in [TriggerType.HALT_RISK, TriggerType.RISK_ALERT]:
                trigger.trigger_pre_halt = True

            # Catalyst/Spike → Alert Pre-Spike Radar
            if trigger.trigger_type in [TriggerType.CATALYST_MAJOR, TriggerType.SPIKE_FORMING]:
                trigger.trigger_pre_spike = True

        # MEDIUM → Hot scan + relevant engine
        elif trigger.trigger_level == TriggerLevel.MEDIUM:
            trigger.trigger_hot_scan = True

            if trigger.trigger_type == TriggerType.SPIKE_FORMING:
                trigger.trigger_pre_spike = True
            elif trigger.trigger_type == TriggerType.RISK_ALERT:
                trigger.trigger_pre_halt = True

        # LOW → Just monitor
        else:
            trigger.trigger_hot_scan = False

    def _update_stats(self, trigger: NewsTrigger) -> None:
        """Update statistics."""
        self.stats.triggers_generated += 1
        self.stats.last_trigger_time = datetime.now()

        if trigger.trigger_type == TriggerType.HALT_RISK:
            self.stats.halt_triggers += 1
        elif trigger.trigger_type == TriggerType.CATALYST_MAJOR:
            self.stats.catalyst_triggers += 1
        elif trigger.trigger_type == TriggerType.SPIKE_FORMING:
            self.stats.spike_triggers += 1
        elif trigger.trigger_type == TriggerType.RISK_ALERT:
            self.stats.risk_triggers += 1

    def _notify_listeners(self, trigger: NewsTrigger) -> None:
        """Notify registered listeners of trigger."""
        for listener in self._listeners:
            try:
                listener(trigger)
            except Exception as e:
                logger.error(f"Error in trigger listener: {e}")

    def add_listener(self, callback: callable) -> None:
        """Add trigger event listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        """Remove trigger event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def process_batch(
        self,
        headlines: List[Tuple[str, str]]
    ) -> List[NewsTrigger]:
        """
        Process batch of headlines.

        Args:
            headlines: List of (ticker, headline) tuples

        Returns:
            List of triggers generated
        """
        triggers = []
        for ticker, headline in headlines:
            trigger = self.process_headline(ticker, headline)
            if trigger:
                triggers.append(trigger)
        return triggers

    def get_recent_triggers(
        self,
        ticker: Optional[str] = None,
        minutes: int = 30
    ) -> List[str]:
        """Get recent trigger cache keys."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = []

        for key, time in self._recent_triggers.items():
            if time >= cutoff:
                if ticker is None or key.startswith(f"{ticker}:"):
                    recent.append(key)

        return recent

    def clear_cache(self, older_than_minutes: int = 60) -> int:
        """Clear old entries from trigger cache."""
        cutoff = datetime.now() - timedelta(minutes=older_than_minutes)
        old_keys = [k for k, t in self._recent_triggers.items() if t < cutoff]

        for key in old_keys:
            del self._recent_triggers[key]

        return len(old_keys)

    def get_stats(self) -> Dict:
        """Get trigger statistics."""
        return {
            "total_processed": self.stats.total_headlines_processed,
            "triggers_generated": self.stats.triggers_generated,
            "halt_triggers": self.stats.halt_triggers,
            "catalyst_triggers": self.stats.catalyst_triggers,
            "spike_triggers": self.stats.spike_triggers,
            "risk_triggers": self.stats.risk_triggers,
            "last_trigger": self.stats.last_trigger_time.isoformat() if self.stats.last_trigger_time else None,
            "cache_size": len(self._recent_triggers),
        }


# ============================================================================
# Singleton
# ============================================================================

_trigger_instance: Optional[IBKRNewsTrigger] = None


def get_ibkr_news_trigger() -> IBKRNewsTrigger:
    """Get singleton IBKRNewsTrigger instance."""
    global _trigger_instance
    if _trigger_instance is None:
        _trigger_instance = IBKRNewsTrigger()
    return _trigger_instance


# ============================================================================
# Convenience functions
# ============================================================================

def check_headline(ticker: str, headline: str) -> Optional[NewsTrigger]:
    """Quick check of a single headline."""
    return get_ibkr_news_trigger().process_headline(ticker, headline)


def get_trigger_stats() -> Dict:
    """Get trigger statistics."""
    return get_ibkr_news_trigger().get_stats()


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Classes
    "IBKRNewsTrigger",
    "NewsTrigger",
    "TriggerLevel",
    "TriggerType",
    "TriggerStats",

    # Constants
    "HALT_KEYWORDS",
    "SPIKE_KEYWORDS",
    "RISK_KEYWORDS",

    # Functions
    "get_ibkr_news_trigger",
    "check_headline",
    "get_trigger_stats",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IBKR NEWS TRIGGER TEST")
    print("=" * 60)

    trigger_engine = IBKRNewsTrigger()

    # Test headlines
    test_headlines = [
        ("AAPL", "Apple announces new product launch"),
        ("BIOX", "BIOX stock halted pending news"),
        ("MRNA", "Moderna receives FDA approval for new vaccine"),
        ("GME", "GameStop surges 20% on unusual volume"),
        ("NKLA", "Nikola announces $500M stock offering"),
        ("TSLA", "Tesla jumps on earnings beat"),
        ("XYZ", "XYZ receives SEC subpoena for investigation"),
        ("ABC", "ABC merger with DEF announced"),
    ]

    print("\nProcessing test headlines:\n")

    for ticker, headline in test_headlines:
        result = trigger_engine.process_headline(ticker, headline)
        if result:
            print(f"  {result}")
            print(f"    → Hot scan: {result.trigger_hot_scan}")
            print(f"    → Pre-Halt: {result.trigger_pre_halt}")
            print(f"    → Pre-Spike: {result.trigger_pre_spike}")
            print()
        else:
            print(f"  [{ticker}] No trigger: {headline[:50]}...")
            print()

    print("\nStatistics:")
    stats = trigger_engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
