"""
PRE-HALT ENGINE V7.0
====================

Risk and timing layer - NOT an opportunity detector.

QUESTION ANSWERED:
"Is this move at risk of becoming uncontrollable or suspended?"

MONITORED SIGNALS:
- IBKR News (halt/pending news keywords)
- Price acceleration too vertical
- Volume explosion without consolidation
- Small float + low liquidity
- Spreads widening suddenly
- Historical halt patterns

OUTPUT:
- PRE_HALT_LOW    → No restriction
- PRE_HALT_MEDIUM → Reduced size / confirmation required
- PRE_HALT_HIGH   → Signal visible BUT execution blocked/delayed

EXECUTION RECOMMENDATIONS:
- EXECUTE     → Proceed normally
- WAIT        → Wait for confirmation
- REDUCE      → Reduce position size
- POST_HALT   → Wait for halt to resume

FUNDAMENTAL RULE:
The engine CONTINUES to detect ALL signals.
Limits only block EXECUTION, never INFORMATION.
The trader ALWAYS sees: BUY/BUY_STRONG + Pre-Halt state + recommendation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import logging

# Import from existing modules
from src.models.signal_types import PreHaltState

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ExecutionRecommendation(Enum):
    """Execution recommendations from Pre-Halt Engine."""
    EXECUTE = "EXECUTE"         # Proceed normally
    WAIT = "WAIT"               # Wait for confirmation/stabilization
    REDUCE = "REDUCE"           # Reduce position size
    POST_HALT = "POST_HALT"     # Wait for halt to resume
    BLOCKED = "BLOCKED"         # Do not execute


class HaltRiskFactor(Enum):
    """Factors contributing to halt risk."""
    NEWS_PENDING = "NEWS_PENDING"           # Pending news detected
    PRICE_VERTICAL = "PRICE_VERTICAL"       # Too vertical price move
    VOLUME_EXPLOSION = "VOLUME_EXPLOSION"   # Extreme volume spike
    LOW_FLOAT = "LOW_FLOAT"                 # Small float, manipulation risk
    SPREAD_WIDENING = "SPREAD_WIDENING"     # Liquidity drying up
    LULD_NEAR = "LULD_NEAR"                 # Near LULD band
    HISTORICAL_HALTER = "HISTORICAL_HALTER" # Frequently halted ticker
    IBKR_NEWS_ALERT = "IBKR_NEWS_ALERT"     # IBKR News trigger fired


# ============================================================================
# Thresholds
# ============================================================================

# Price acceleration thresholds
PRICE_VERTICAL_5MIN_PCT = 15.0      # 15% in 5 min = vertical
PRICE_VERTICAL_15MIN_PCT = 25.0     # 25% in 15 min = vertical

# Volume thresholds
VOLUME_EXPLOSION_RATIO = 10.0       # 10x average = explosion
VOLUME_SPIKE_RATIO = 5.0            # 5x average = spike

# Float thresholds
LOW_FLOAT_SHARES = 10_000_000       # < 10M shares = low float
VERY_LOW_FLOAT_SHARES = 5_000_000   # < 5M shares = very low

# Spread thresholds
SPREAD_WIDE_PCT = 2.0               # > 2% spread = wide
SPREAD_VERY_WIDE_PCT = 5.0          # > 5% spread = very wide

# LULD proximity
LULD_NEAR_PCT = 2.0                 # < 2% from band = near
LULD_IMMINENT_PCT = 0.5             # < 0.5% from band = imminent


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PreHaltAssessment:
    """Complete Pre-Halt assessment for a ticker."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.now)

    # State
    pre_halt_state: PreHaltState = PreHaltState.LOW
    recommendation: ExecutionRecommendation = ExecutionRecommendation.EXECUTE

    # Risk factors present
    risk_factors: List[HaltRiskFactor] = field(default_factory=list)

    # Scores
    halt_probability: float = 0.0       # 0-100%
    risk_score: float = 0.0             # 0-100

    # Position sizing
    size_multiplier: float = 1.0        # 0.0 to 1.0

    # Context
    current_price: Optional[float] = None
    price_change_5min: Optional[float] = None
    price_change_15min: Optional[float] = None
    volume_ratio: Optional[float] = None
    spread_pct: Optional[float] = None
    float_shares: Optional[int] = None
    luld_distance_pct: Optional[float] = None

    # Halt history
    is_currently_halted: bool = False
    halts_today: int = 0
    last_halt_time: Optional[datetime] = None

    # IBKR News trigger
    ibkr_news_triggered: bool = False
    ibkr_news_level: Optional[str] = None

    # Message
    message: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "pre_halt_state": self.pre_halt_state.value,
            "recommendation": self.recommendation.value,
            "risk_factors": [f.value for f in self.risk_factors],
            "halt_probability": self.halt_probability,
            "risk_score": self.risk_score,
            "size_multiplier": self.size_multiplier,
            "is_currently_halted": self.is_currently_halted,
            "message": self.message,
        }


# ============================================================================
# Pre-Halt Engine
# ============================================================================

class PreHaltEngine:
    """
    Pre-Halt Engine - Risk and Timing Layer

    Usage:
        engine = PreHaltEngine()

        # Assess halt risk
        assessment = engine.assess(
            ticker="BIOX",
            current_price=2.50,
            price_5min_ago=2.20,
            current_volume=5_000_000,
            avg_volume=500_000,
            spread_pct=1.5,
            float_shares=8_000_000
        )

        # Get recommendation
        if assessment.recommendation == ExecutionRecommendation.EXECUTE:
            proceed_with_trade()
        elif assessment.recommendation == ExecutionRecommendation.REDUCE:
            reduce_position_size(assessment.size_multiplier)
        elif assessment.recommendation == ExecutionRecommendation.WAIT:
            wait_for_stabilization()
        elif assessment.recommendation == ExecutionRecommendation.POST_HALT:
            wait_for_resume()

    IMPORTANT:
    - Signal detection is NEVER blocked
    - Only execution authorization changes
    - Trader ALWAYS sees the signal + Pre-Halt state
    """

    def __init__(self):
        # Assessment cache
        self._cache: Dict[str, PreHaltAssessment] = {}
        self._cache_ttl = timedelta(seconds=30)

        # Halt history tracking
        self._halt_history: Dict[str, List[datetime]] = {}

        # Currently halted
        self._halted_tickers: Set[str] = set()

        # IBKR News integration
        self._ibkr_news_alerts: Dict[str, Tuple[datetime, str]] = {}

    def assess(
        self,
        ticker: str,
        current_price: Optional[float] = None,
        price_5min_ago: Optional[float] = None,
        price_15min_ago: Optional[float] = None,
        current_volume: Optional[int] = None,
        avg_volume: Optional[int] = None,
        spread_pct: Optional[float] = None,
        float_shares: Optional[int] = None,
        luld_upper: Optional[float] = None,
        luld_lower: Optional[float] = None,
        force_refresh: bool = False
    ) -> PreHaltAssessment:
        """
        Assess halt risk for a ticker.

        Returns PreHaltAssessment with:
        - pre_halt_state: LOW, MEDIUM, HIGH
        - recommendation: EXECUTE, WAIT, REDUCE, POST_HALT, BLOCKED
        - size_multiplier: 0.0 to 1.0
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._cache:
            cached = self._cache[ticker]
            if datetime.now() - cached.timestamp < self._cache_ttl:
                return cached

        # Create assessment
        assessment = PreHaltAssessment(ticker=ticker)
        assessment.current_price = current_price

        # Check if currently halted
        if ticker in self._halted_tickers:
            assessment.is_currently_halted = True
            assessment.pre_halt_state = PreHaltState.HIGH
            assessment.recommendation = ExecutionRecommendation.POST_HALT
            assessment.size_multiplier = 0.0
            assessment.risk_factors.append(HaltRiskFactor.NEWS_PENDING)
            assessment.message = "Trading currently halted"
            self._cache[ticker] = assessment
            return assessment

        # Collect risk factors
        risk_factors = []
        risk_score = 0.0

        # 1. Check IBKR News alerts
        if ticker in self._ibkr_news_alerts:
            alert_time, alert_level = self._ibkr_news_alerts[ticker]
            if datetime.now() - alert_time < timedelta(minutes=15):
                risk_factors.append(HaltRiskFactor.IBKR_NEWS_ALERT)
                assessment.ibkr_news_triggered = True
                assessment.ibkr_news_level = alert_level

                if alert_level == "CRITICAL":
                    risk_score += 40
                elif alert_level == "HIGH":
                    risk_score += 25
                else:
                    risk_score += 10

        # 2. Price acceleration
        if current_price and price_5min_ago and price_5min_ago > 0:
            change_5min = ((current_price - price_5min_ago) / price_5min_ago) * 100
            assessment.price_change_5min = change_5min

            if abs(change_5min) >= PRICE_VERTICAL_5MIN_PCT:
                risk_factors.append(HaltRiskFactor.PRICE_VERTICAL)
                risk_score += 30

        if current_price and price_15min_ago and price_15min_ago > 0:
            change_15min = ((current_price - price_15min_ago) / price_15min_ago) * 100
            assessment.price_change_15min = change_15min

            if abs(change_15min) >= PRICE_VERTICAL_15MIN_PCT:
                if HaltRiskFactor.PRICE_VERTICAL not in risk_factors:
                    risk_factors.append(HaltRiskFactor.PRICE_VERTICAL)
                    risk_score += 20

        # 3. Volume explosion
        if current_volume and avg_volume and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            assessment.volume_ratio = volume_ratio

            if volume_ratio >= VOLUME_EXPLOSION_RATIO:
                risk_factors.append(HaltRiskFactor.VOLUME_EXPLOSION)
                risk_score += 25
            elif volume_ratio >= VOLUME_SPIKE_RATIO:
                risk_score += 10

        # 4. Low float
        if float_shares:
            assessment.float_shares = float_shares

            if float_shares < VERY_LOW_FLOAT_SHARES:
                risk_factors.append(HaltRiskFactor.LOW_FLOAT)
                risk_score += 20
            elif float_shares < LOW_FLOAT_SHARES:
                risk_factors.append(HaltRiskFactor.LOW_FLOAT)
                risk_score += 10

        # 5. Spread widening
        if spread_pct:
            assessment.spread_pct = spread_pct

            if spread_pct >= SPREAD_VERY_WIDE_PCT:
                risk_factors.append(HaltRiskFactor.SPREAD_WIDENING)
                risk_score += 25
            elif spread_pct >= SPREAD_WIDE_PCT:
                risk_factors.append(HaltRiskFactor.SPREAD_WIDENING)
                risk_score += 10

        # 6. LULD proximity
        if current_price and luld_upper and luld_lower:
            distance_to_upper = ((luld_upper - current_price) / current_price) * 100
            distance_to_lower = ((current_price - luld_lower) / current_price) * 100
            min_distance = min(distance_to_upper, distance_to_lower)
            assessment.luld_distance_pct = min_distance

            if min_distance <= LULD_IMMINENT_PCT:
                risk_factors.append(HaltRiskFactor.LULD_NEAR)
                risk_score += 40
            elif min_distance <= LULD_NEAR_PCT:
                risk_factors.append(HaltRiskFactor.LULD_NEAR)
                risk_score += 20

        # 7. Historical halter
        if ticker in self._halt_history:
            recent_halts = [
                h for h in self._halt_history[ticker]
                if datetime.now() - h < timedelta(days=7)
            ]
            assessment.halts_today = len([
                h for h in recent_halts
                if h.date() == datetime.now().date()
            ])

            if len(recent_halts) >= 3:
                risk_factors.append(HaltRiskFactor.HISTORICAL_HALTER)
                risk_score += 15

            if recent_halts:
                assessment.last_halt_time = max(recent_halts)

        # Store risk factors
        assessment.risk_factors = risk_factors
        assessment.risk_score = min(100, risk_score)

        # Calculate halt probability
        assessment.halt_probability = self._calculate_halt_probability(assessment)

        # Determine state and recommendation
        self._determine_state_and_recommendation(assessment)

        # Generate message
        assessment.message = self._generate_message(assessment)

        # Cache result
        self._cache[ticker] = assessment

        logger.debug(
            f"Pre-Halt assessment for {ticker}: "
            f"{assessment.pre_halt_state.value} / {assessment.recommendation.value} "
            f"(score={assessment.risk_score:.0f}, prob={assessment.halt_probability:.0f}%)"
        )

        return assessment

    def _calculate_halt_probability(self, assessment: PreHaltAssessment) -> float:
        """Calculate halt probability based on risk factors."""
        base_prob = assessment.risk_score * 0.8  # Risk score maps to probability

        # Adjust for specific combinations
        factors = assessment.risk_factors

        # LULD near + volume explosion = very high
        if HaltRiskFactor.LULD_NEAR in factors and HaltRiskFactor.VOLUME_EXPLOSION in factors:
            base_prob = min(100, base_prob * 1.5)

        # Vertical price + low float = high
        if HaltRiskFactor.PRICE_VERTICAL in factors and HaltRiskFactor.LOW_FLOAT in factors:
            base_prob = min(100, base_prob * 1.3)

        # IBKR news CRITICAL + any other factor = very high
        if assessment.ibkr_news_level == "CRITICAL" and len(factors) > 1:
            base_prob = min(100, base_prob * 1.4)

        return min(100, base_prob)

    def _determine_state_and_recommendation(self, assessment: PreHaltAssessment) -> None:
        """Determine pre-halt state and execution recommendation."""

        score = assessment.risk_score
        prob = assessment.halt_probability

        # HIGH state
        if score >= 60 or prob >= 70:
            assessment.pre_halt_state = PreHaltState.HIGH
            assessment.recommendation = ExecutionRecommendation.BLOCKED
            assessment.size_multiplier = 0.0

        # MEDIUM state with LULD near → WAIT
        elif HaltRiskFactor.LULD_NEAR in assessment.risk_factors:
            assessment.pre_halt_state = PreHaltState.MEDIUM
            assessment.recommendation = ExecutionRecommendation.WAIT
            assessment.size_multiplier = 0.0

        # MEDIUM state
        elif score >= 30 or prob >= 40:
            assessment.pre_halt_state = PreHaltState.MEDIUM
            assessment.recommendation = ExecutionRecommendation.REDUCE
            assessment.size_multiplier = 0.5

        # LOW state with some factors
        elif assessment.risk_factors:
            assessment.pre_halt_state = PreHaltState.LOW
            assessment.recommendation = ExecutionRecommendation.REDUCE
            assessment.size_multiplier = 0.75

        # LOW state, clean
        else:
            assessment.pre_halt_state = PreHaltState.LOW
            assessment.recommendation = ExecutionRecommendation.EXECUTE
            assessment.size_multiplier = 1.0

    def _generate_message(self, assessment: PreHaltAssessment) -> str:
        """Generate human-readable message."""
        if assessment.is_currently_halted:
            return "Trading halted - wait for resume"

        if not assessment.risk_factors:
            return "Normal conditions"

        factors_str = ", ".join(f.value for f in assessment.risk_factors[:3])
        return f"Risk factors: {factors_str} (prob: {assessment.halt_probability:.0f}%)"

    # ========================================================================
    # IBKR News Integration
    # ========================================================================

    def on_ibkr_news_trigger(self, ticker: str, level: str) -> None:
        """
        Called when IBKR News Trigger fires.

        Args:
            ticker: Stock ticker
            level: Trigger level (CRITICAL, HIGH, MEDIUM, LOW)
        """
        ticker = ticker.upper()
        self._ibkr_news_alerts[ticker] = (datetime.now(), level)

        # Invalidate cache
        self._cache.pop(ticker, None)

        logger.info(f"Pre-Halt Engine: IBKR News alert for {ticker} ({level})")

    def clear_ibkr_alert(self, ticker: str) -> None:
        """Clear IBKR news alert for ticker."""
        self._ibkr_news_alerts.pop(ticker.upper(), None)

    # ========================================================================
    # Halt tracking
    # ========================================================================

    def record_halt(self, ticker: str, halt_time: Optional[datetime] = None) -> None:
        """Record a halt event."""
        ticker = ticker.upper()
        halt_time = halt_time or datetime.now()

        self._halted_tickers.add(ticker)

        if ticker not in self._halt_history:
            self._halt_history[ticker] = []
        self._halt_history[ticker].append(halt_time)

        # Invalidate cache
        self._cache.pop(ticker, None)

        logger.info(f"Pre-Halt Engine: Recorded halt for {ticker}")

    def record_resume(self, ticker: str) -> None:
        """Record halt resume."""
        ticker = ticker.upper()
        self._halted_tickers.discard(ticker)

        # Invalidate cache
        self._cache.pop(ticker, None)

        logger.info(f"Pre-Halt Engine: Recorded resume for {ticker}")

    def is_halted(self, ticker: str) -> bool:
        """Check if ticker is currently halted."""
        return ticker.upper() in self._halted_tickers

    def get_halted_tickers(self) -> Set[str]:
        """Get all currently halted tickers."""
        return self._halted_tickers.copy()

    # ========================================================================
    # Cache management
    # ========================================================================

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear assessment cache."""
        if ticker:
            self._cache.pop(ticker.upper(), None)
        else:
            self._cache.clear()

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "cached_assessments": len(self._cache),
            "halted_tickers": len(self._halted_tickers),
            "ibkr_alerts_active": len(self._ibkr_news_alerts),
            "tickers_with_history": len(self._halt_history),
        }


# ============================================================================
# Singleton
# ============================================================================

_engine_instance: Optional[PreHaltEngine] = None


def get_pre_halt_engine() -> PreHaltEngine:
    """Get singleton PreHaltEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PreHaltEngine()
    return _engine_instance


# ============================================================================
# Convenience functions
# ============================================================================

def assess_halt_risk(ticker: str, **kwargs) -> PreHaltAssessment:
    """Quick halt risk assessment."""
    return get_pre_halt_engine().assess(ticker, **kwargs)


def get_recommendation(ticker: str) -> ExecutionRecommendation:
    """Get execution recommendation for ticker."""
    assessment = get_pre_halt_engine().assess(ticker)
    return assessment.recommendation


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Classes
    "PreHaltEngine",
    "PreHaltAssessment",
    "ExecutionRecommendation",
    "HaltRiskFactor",

    # Functions
    "get_pre_halt_engine",
    "assess_halt_risk",
    "get_recommendation",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRE-HALT ENGINE TEST")
    print("=" * 60)

    engine = PreHaltEngine()

    # Test cases
    test_cases = [
        {
            "ticker": "NORMAL",
            "current_price": 5.0,
            "price_5min_ago": 4.95,
            "current_volume": 600_000,
            "avg_volume": 500_000,
            "spread_pct": 0.5,
            "float_shares": 50_000_000,
        },
        {
            "ticker": "VOLATILE",
            "current_price": 5.0,
            "price_5min_ago": 4.0,  # 25% move in 5 min
            "current_volume": 3_000_000,
            "avg_volume": 500_000,
            "spread_pct": 1.5,
            "float_shares": 8_000_000,
        },
        {
            "ticker": "DANGEROUS",
            "current_price": 5.0,
            "price_5min_ago": 3.5,  # 43% move
            "current_volume": 10_000_000,
            "avg_volume": 500_000,
            "spread_pct": 4.0,
            "float_shares": 3_000_000,
            "luld_upper": 5.25,
            "luld_lower": 4.25,
        },
    ]

    print("\nTest assessments:\n")

    for case in test_cases:
        ticker = case.pop("ticker")
        assessment = engine.assess(ticker, **case)

        print(f"  {ticker}:")
        print(f"    State: {assessment.pre_halt_state.value}")
        print(f"    Recommendation: {assessment.recommendation.value}")
        print(f"    Size multiplier: {assessment.size_multiplier}")
        print(f"    Risk score: {assessment.risk_score:.0f}")
        print(f"    Halt probability: {assessment.halt_probability:.0f}%")
        print(f"    Factors: {[f.value for f in assessment.risk_factors]}")
        print(f"    Message: {assessment.message}")
        print()

    # Test IBKR News integration
    print("\nIBKR News integration test:")
    engine.on_ibkr_news_trigger("ALERT", "CRITICAL")
    assessment = engine.assess("ALERT", current_price=5.0)
    print(f"  ALERT (after CRITICAL news):")
    print(f"    State: {assessment.pre_halt_state.value}")
    print(f"    Recommendation: {assessment.recommendation.value}")
    print(f"    IBKR triggered: {assessment.ibkr_news_triggered}")
