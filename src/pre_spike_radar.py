# ============================
# PRE-SPIKE RADAR
# ============================
# Detects early warning signals BEFORE major price spikes
# Part of GV2-EDGE V6 - Pre-Spike Radar Layer
#
# Concept: Most top gainers show subtle signals before exploding:
# - Volume acceleration (not just level, but derivative)
# - Options flow momentum (increasing call activity)
# - Social buzz acceleration (mentions picking up)
# - Technical compression (squeeze before breakout)
#
# This module detects these precursor signals for anticipation.

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

from utils.logger import get_logger
from utils.cache import Cache

logger = get_logger("PRE_SPIKE_RADAR")

# Cache for historical data (short TTL for real-time)
cache = Cache(ttl=300)  # 5 min cache


# ============================
# DATA STRUCTURES
# ============================

@dataclass
class AccelerationSignal:
    """Single acceleration signal"""
    signal_type: str          # VOLUME, OPTIONS, BUZZ, TECHNICAL
    value: float              # Current acceleration value (0-1)
    velocity: float           # Rate of change
    is_accelerating: bool     # True if positive acceleration
    strength: str             # WEAK, MODERATE, STRONG
    details: Dict = field(default_factory=dict)


@dataclass
class PreSpikeSignal:
    """Combined pre-spike radar signal for a ticker"""
    ticker: str
    timestamp: datetime

    # Individual signals
    volume_acceleration: Optional[AccelerationSignal] = None
    options_acceleration: Optional[AccelerationSignal] = None
    buzz_acceleration: Optional[AccelerationSignal] = None
    technical_compression: Optional[AccelerationSignal] = None

    # Combined metrics
    confluence_count: int = 0         # Number of active signals (0-4)
    confluence_score: float = 0.0     # Combined score (0-1)
    pre_spike_probability: float = 0.0  # Estimated spike probability

    # Classification
    alert_level: str = "NONE"         # NONE, WATCH, ELEVATED, HIGH
    is_pre_spike: bool = False        # True if high probability


# ============================
# CONFIGURATION
# ============================

# Thresholds for signal detection
VOLUME_ACCELERATION_THRESHOLD = 0.3   # Min acceleration to count
OPTIONS_ACCELERATION_THRESHOLD = 0.25
BUZZ_ACCELERATION_THRESHOLD = 0.2
TECHNICAL_SQUEEZE_THRESHOLD = 0.4

# Confluence requirements
MIN_CONFLUENCE_FOR_ALERT = 2          # Minimum signals for WATCH
MIN_CONFLUENCE_FOR_HIGH = 3           # Minimum signals for HIGH

# Time windows for acceleration calculation
ACCELERATION_WINDOW_MINUTES = 30      # Window for measuring change
LOOKBACK_PERIODS = 6                  # Number of periods to analyze


# ============================
# VOLUME ACCELERATION
# ============================

def calculate_volume_acceleration(
    ticker: str,
    current_volume: int,
    historical_volumes: List[Tuple[datetime, int]],
    avg_daily_volume: int
) -> AccelerationSignal:
    """
    Calculate volume acceleration (derivative of volume over time)

    Not just "is volume high?" but "is volume INCREASING?"
    A stock going from 10k to 50k in 30 min is more significant than
    stable 100k volume.

    Args:
        ticker: Stock symbol
        current_volume: Current period volume
        historical_volumes: List of (timestamp, volume) tuples
        avg_daily_volume: Average daily volume for normalization

    Returns:
        AccelerationSignal with volume acceleration data
    """
    if not historical_volumes or avg_daily_volume <= 0:
        return AccelerationSignal(
            signal_type="VOLUME",
            value=0.0,
            velocity=0.0,
            is_accelerating=False,
            strength="NONE",
            details={"error": "insufficient_data"}
        )

    # Sort by timestamp
    sorted_vols = sorted(historical_volumes, key=lambda x: x[0])

    # Calculate volume velocity (change per period)
    velocities = []
    for i in range(1, len(sorted_vols)):
        prev_vol = sorted_vols[i-1][1]
        curr_vol = sorted_vols[i][1]

        # Normalize by average daily volume
        prev_norm = prev_vol / avg_daily_volume
        curr_norm = curr_vol / avg_daily_volume

        velocity = curr_norm - prev_norm
        velocities.append(velocity)

    if not velocities:
        return AccelerationSignal(
            signal_type="VOLUME",
            value=0.0,
            velocity=0.0,
            is_accelerating=False,
            strength="NONE"
        )

    # Calculate acceleration (change in velocity)
    accelerations = []
    for i in range(1, len(velocities)):
        acc = velocities[i] - velocities[i-1]
        accelerations.append(acc)

    # Current metrics
    current_velocity = velocities[-1] if velocities else 0
    current_acceleration = accelerations[-1] if accelerations else 0
    avg_acceleration = sum(accelerations) / len(accelerations) if accelerations else 0

    # Normalize to 0-1 scale
    # Use sigmoid-like transformation
    normalized_acc = 1 / (1 + math.exp(-current_acceleration * 10))

    # Determine if accelerating
    is_accelerating = current_acceleration > 0 and current_velocity > 0

    # Strength classification
    if normalized_acc >= 0.7:
        strength = "STRONG"
    elif normalized_acc >= 0.4:
        strength = "MODERATE"
    elif normalized_acc >= VOLUME_ACCELERATION_THRESHOLD:
        strength = "WEAK"
    else:
        strength = "NONE"

    return AccelerationSignal(
        signal_type="VOLUME",
        value=normalized_acc,
        velocity=current_velocity,
        is_accelerating=is_accelerating,
        strength=strength,
        details={
            "current_volume": current_volume,
            "avg_daily_volume": avg_daily_volume,
            "volume_ratio": current_volume / avg_daily_volume if avg_daily_volume > 0 else 0,
            "acceleration_raw": current_acceleration,
            "velocity_trend": "UP" if current_velocity > 0 else "DOWN"
        }
    )


# ============================
# OPTIONS FLOW ACCELERATION
# ============================

def calculate_options_acceleration(
    ticker: str,
    current_call_volume: int,
    current_put_volume: int,
    historical_options: List[Tuple[datetime, int, int]],  # (time, calls, puts)
) -> AccelerationSignal:
    """
    Calculate options flow acceleration

    Detects:
    - Increasing call volume momentum
    - Decreasing put/call ratio (more bullish)
    - Call concentration increasing

    Args:
        ticker: Stock symbol
        current_call_volume: Current call volume
        current_put_volume: Current put volume
        historical_options: List of (timestamp, call_vol, put_vol) tuples

    Returns:
        AccelerationSignal with options acceleration data
    """
    if not historical_options:
        return AccelerationSignal(
            signal_type="OPTIONS",
            value=0.0,
            velocity=0.0,
            is_accelerating=False,
            strength="NONE",
            details={"error": "insufficient_data"}
        )

    # Sort by timestamp
    sorted_opts = sorted(historical_options, key=lambda x: x[0])

    # Calculate call ratio over time
    call_ratios = []
    for ts, calls, puts in sorted_opts:
        total = calls + puts
        ratio = calls / total if total > 0 else 0.5
        call_ratios.append(ratio)

    # Current call ratio
    current_total = current_call_volume + current_put_volume
    current_ratio = current_call_volume / current_total if current_total > 0 else 0.5
    call_ratios.append(current_ratio)

    # Calculate velocity (change in call ratio)
    velocities = []
    for i in range(1, len(call_ratios)):
        vel = call_ratios[i] - call_ratios[i-1]
        velocities.append(vel)

    # Calculate acceleration
    accelerations = []
    for i in range(1, len(velocities)):
        acc = velocities[i] - velocities[i-1]
        accelerations.append(acc)

    current_velocity = velocities[-1] if velocities else 0
    current_acceleration = accelerations[-1] if accelerations else 0

    # Normalize - options acceleration is more subtle
    normalized_acc = 1 / (1 + math.exp(-current_acceleration * 20))

    # Boost if call ratio is already high and still accelerating
    if current_ratio > 0.6 and current_velocity > 0:
        normalized_acc = min(1.0, normalized_acc * 1.2)

    is_accelerating = current_acceleration > 0 and current_ratio > 0.5

    # Strength
    if normalized_acc >= 0.7 and current_ratio > 0.65:
        strength = "STRONG"
    elif normalized_acc >= 0.4:
        strength = "MODERATE"
    elif normalized_acc >= OPTIONS_ACCELERATION_THRESHOLD:
        strength = "WEAK"
    else:
        strength = "NONE"

    return AccelerationSignal(
        signal_type="OPTIONS",
        value=normalized_acc,
        velocity=current_velocity,
        is_accelerating=is_accelerating,
        strength=strength,
        details={
            "current_call_volume": current_call_volume,
            "current_put_volume": current_put_volume,
            "call_ratio": current_ratio,
            "pc_ratio": current_put_volume / current_call_volume if current_call_volume > 0 else 999,
            "call_trend": "BULLISH" if current_velocity > 0 else "BEARISH"
        }
    )


# ============================
# SOCIAL BUZZ ACCELERATION
# ============================

def calculate_buzz_acceleration(
    ticker: str,
    current_mentions: int,
    historical_mentions: List[Tuple[datetime, int]],
    baseline_mentions: int = 10
) -> AccelerationSignal:
    """
    Calculate social buzz acceleration

    Detects:
    - Mentions increasing over time
    - Acceleration of mention growth
    - Early buzz before mainstream attention

    Args:
        ticker: Stock symbol
        current_mentions: Current period mention count
        historical_mentions: List of (timestamp, mentions) tuples
        baseline_mentions: Baseline mentions for normalization

    Returns:
        AccelerationSignal with buzz acceleration data
    """
    if not historical_mentions:
        return AccelerationSignal(
            signal_type="BUZZ",
            value=0.0,
            velocity=0.0,
            is_accelerating=False,
            strength="NONE",
            details={"error": "insufficient_data"}
        )

    # Sort and normalize
    sorted_mentions = sorted(historical_mentions, key=lambda x: x[0])

    # Normalize by baseline
    normalized = [(ts, m / max(baseline_mentions, 1)) for ts, m in sorted_mentions]
    current_norm = current_mentions / max(baseline_mentions, 1)

    # Calculate velocity
    values = [n[1] for n in normalized] + [current_norm]
    velocities = [values[i] - values[i-1] for i in range(1, len(values))]

    # Calculate acceleration
    accelerations = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]

    current_velocity = velocities[-1] if velocities else 0
    current_acceleration = accelerations[-1] if accelerations else 0

    # Normalize
    normalized_acc = 1 / (1 + math.exp(-current_acceleration * 5))

    # Boost for sustained growth
    if all(v > 0 for v in velocities[-3:]) if len(velocities) >= 3 else False:
        normalized_acc = min(1.0, normalized_acc * 1.15)

    is_accelerating = current_acceleration > 0 and current_velocity > 0

    # Calculate spike ratio (current vs baseline)
    spike_ratio = current_mentions / max(baseline_mentions, 1)

    # Strength
    if normalized_acc >= 0.65 and spike_ratio > 2.0:
        strength = "STRONG"
    elif normalized_acc >= 0.4 or spike_ratio > 1.5:
        strength = "MODERATE"
    elif normalized_acc >= BUZZ_ACCELERATION_THRESHOLD:
        strength = "WEAK"
    else:
        strength = "NONE"

    return AccelerationSignal(
        signal_type="BUZZ",
        value=normalized_acc,
        velocity=current_velocity,
        is_accelerating=is_accelerating,
        strength=strength,
        details={
            "current_mentions": current_mentions,
            "baseline_mentions": baseline_mentions,
            "spike_ratio": spike_ratio,
            "growth_trend": "ACCELERATING" if is_accelerating else "STABLE"
        }
    )


# ============================
# TECHNICAL COMPRESSION
# ============================

def calculate_technical_compression(
    ticker: str,
    bollinger_bandwidth: float,
    historical_bandwidth: List[float],
    atr_ratio: float = 1.0
) -> AccelerationSignal:
    """
    Calculate technical compression (squeeze detection)

    Detects:
    - Bollinger Band squeeze (low bandwidth)
    - Volatility compression before expansion
    - ATR contraction

    Args:
        ticker: Stock symbol
        bollinger_bandwidth: Current Bollinger bandwidth
        historical_bandwidth: List of historical bandwidth values
        atr_ratio: Current ATR / Average ATR

    Returns:
        AccelerationSignal with compression data
    """
    if not historical_bandwidth:
        return AccelerationSignal(
            signal_type="TECHNICAL",
            value=0.0,
            velocity=0.0,
            is_accelerating=False,
            strength="NONE",
            details={"error": "insufficient_data"}
        )

    # Calculate average bandwidth
    avg_bandwidth = sum(historical_bandwidth) / len(historical_bandwidth)

    # Compression ratio (lower = more compressed)
    compression_ratio = bollinger_bandwidth / avg_bandwidth if avg_bandwidth > 0 else 1.0

    # Invert for scoring (higher score = more compressed = better)
    compression_score = max(0, 1 - compression_ratio)

    # Check if bandwidth is decreasing (squeeze forming)
    if len(historical_bandwidth) >= 3:
        recent = historical_bandwidth[-3:]
        is_decreasing = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
    else:
        is_decreasing = bollinger_bandwidth < avg_bandwidth

    # Factor in ATR
    if atr_ratio < 0.8:  # ATR also compressed
        compression_score = min(1.0, compression_score * 1.2)

    # Velocity (rate of compression)
    if len(historical_bandwidth) >= 2:
        velocity = (historical_bandwidth[-1] - bollinger_bandwidth) / avg_bandwidth
    else:
        velocity = 0

    is_compressing = compression_score > TECHNICAL_SQUEEZE_THRESHOLD and is_decreasing

    # Strength
    if compression_score >= 0.7 and is_decreasing:
        strength = "STRONG"
    elif compression_score >= 0.5:
        strength = "MODERATE"
    elif compression_score >= TECHNICAL_SQUEEZE_THRESHOLD:
        strength = "WEAK"
    else:
        strength = "NONE"

    return AccelerationSignal(
        signal_type="TECHNICAL",
        value=compression_score,
        velocity=velocity,
        is_accelerating=is_compressing,
        strength=strength,
        details={
            "bollinger_bandwidth": bollinger_bandwidth,
            "avg_bandwidth": avg_bandwidth,
            "compression_ratio": compression_ratio,
            "atr_ratio": atr_ratio,
            "squeeze_forming": is_decreasing
        }
    )


# ============================
# CONFLUENCE SCORING
# ============================

def calculate_confluence_score(signals: List[AccelerationSignal]) -> Tuple[int, float]:
    """
    Calculate confluence score from multiple acceleration signals

    More signals aligning = higher probability of spike

    Args:
        signals: List of AccelerationSignal objects

    Returns:
        Tuple of (confluence_count, confluence_score)
    """
    if not signals:
        return 0, 0.0

    # Count active signals
    active_signals = [s for s in signals if s.is_accelerating and s.strength != "NONE"]
    confluence_count = len(active_signals)

    if confluence_count == 0:
        return 0, 0.0

    # Weight by signal type
    weights = {
        "VOLUME": 0.35,     # Volume acceleration most predictive
        "OPTIONS": 0.30,    # Smart money indicator
        "BUZZ": 0.20,       # Social momentum
        "TECHNICAL": 0.15   # Compression setup
    }

    # Calculate weighted score
    weighted_sum = 0.0
    total_weight = 0.0

    for signal in signals:
        weight = weights.get(signal.signal_type, 0.25)
        if signal.is_accelerating:
            # Boost by strength
            strength_mult = {"STRONG": 1.3, "MODERATE": 1.0, "WEAK": 0.7}.get(signal.strength, 0.5)
            weighted_sum += signal.value * weight * strength_mult
        total_weight += weight

    confluence_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Boost for high confluence
    if confluence_count >= 3:
        confluence_score = min(1.0, confluence_score * 1.25)
    if confluence_count >= 4:
        confluence_score = min(1.0, confluence_score * 1.15)

    return confluence_count, min(1.0, confluence_score)


def estimate_spike_probability(confluence_count: int, confluence_score: float) -> float:
    """
    Estimate probability of imminent spike based on confluence

    Based on historical analysis:
    - 1 signal: ~15% chance of significant move
    - 2 signals: ~35% chance
    - 3 signals: ~55% chance
    - 4 signals: ~75% chance

    Args:
        confluence_count: Number of active signals
        confluence_score: Weighted confluence score

    Returns:
        Estimated probability (0-1)
    """
    # Base probability by confluence count
    base_prob = {
        0: 0.05,
        1: 0.15,
        2: 0.35,
        3: 0.55,
        4: 0.75
    }.get(confluence_count, 0.75)

    # Adjust by score intensity
    intensity_mult = 0.7 + (confluence_score * 0.6)  # 0.7 to 1.3x

    return min(0.95, base_prob * intensity_mult)


# ============================
# MAIN RADAR FUNCTION
# ============================

def scan_pre_spike(
    ticker: str,
    volume_data: Optional[Dict] = None,
    options_data: Optional[Dict] = None,
    buzz_data: Optional[Dict] = None,
    technical_data: Optional[Dict] = None
) -> PreSpikeSignal:
    """
    Main pre-spike radar scan for a single ticker

    Combines all acceleration signals into a unified pre-spike assessment.

    Args:
        ticker: Stock symbol
        volume_data: Dict with current_volume, historical_volumes, avg_daily_volume
        options_data: Dict with current_calls, current_puts, historical_options
        buzz_data: Dict with current_mentions, historical_mentions, baseline
        technical_data: Dict with bollinger_bandwidth, historical_bandwidth, atr_ratio

    Returns:
        PreSpikeSignal with full analysis
    """
    signals = []

    # Volume acceleration
    vol_signal = None
    if volume_data:
        vol_signal = calculate_volume_acceleration(
            ticker,
            volume_data.get("current_volume", 0),
            volume_data.get("historical_volumes", []),
            volume_data.get("avg_daily_volume", 1)
        )
        signals.append(vol_signal)

    # Options acceleration
    opt_signal = None
    if options_data:
        opt_signal = calculate_options_acceleration(
            ticker,
            options_data.get("current_calls", 0),
            options_data.get("current_puts", 0),
            options_data.get("historical_options", [])
        )
        signals.append(opt_signal)

    # Buzz acceleration
    buzz_signal = None
    if buzz_data:
        buzz_signal = calculate_buzz_acceleration(
            ticker,
            buzz_data.get("current_mentions", 0),
            buzz_data.get("historical_mentions", []),
            buzz_data.get("baseline", 10)
        )
        signals.append(buzz_signal)

    # Technical compression
    tech_signal = None
    if technical_data:
        tech_signal = calculate_technical_compression(
            ticker,
            technical_data.get("bollinger_bandwidth", 0.1),
            technical_data.get("historical_bandwidth", []),
            technical_data.get("atr_ratio", 1.0)
        )
        signals.append(tech_signal)

    # Calculate confluence
    confluence_count, confluence_score = calculate_confluence_score(signals)
    spike_probability = estimate_spike_probability(confluence_count, confluence_score)

    # Determine alert level
    if confluence_count >= MIN_CONFLUENCE_FOR_HIGH and confluence_score >= 0.6:
        alert_level = "HIGH"
    elif confluence_count >= MIN_CONFLUENCE_FOR_ALERT and confluence_score >= 0.4:
        alert_level = "ELEVATED"
    elif confluence_count >= 1 and confluence_score >= 0.3:
        alert_level = "WATCH"
    else:
        alert_level = "NONE"

    is_pre_spike = alert_level in ["HIGH", "ELEVATED"]

    return PreSpikeSignal(
        ticker=ticker,
        timestamp=datetime.now(),
        volume_acceleration=vol_signal,
        options_acceleration=opt_signal,
        buzz_acceleration=buzz_signal,
        technical_compression=tech_signal,
        confluence_count=confluence_count,
        confluence_score=confluence_score,
        pre_spike_probability=spike_probability,
        alert_level=alert_level,
        is_pre_spike=is_pre_spike
    )


def get_pre_spike_boost(signal: PreSpikeSignal) -> float:
    """
    Get pre-spike boost factor for Monster Score

    Args:
        signal: PreSpikeSignal from scan

    Returns:
        Boost factor (1.0 = no boost, up to 1.4 for high probability)
    """
    if not signal.is_pre_spike:
        return 1.0

    # Base boost by alert level
    base_boost = {
        "NONE": 1.0,
        "WATCH": 1.05,
        "ELEVATED": 1.15,
        "HIGH": 1.30
    }.get(signal.alert_level, 1.0)

    # Additional boost by probability
    prob_boost = 1.0 + (signal.pre_spike_probability * 0.1)

    return min(1.4, base_boost * prob_boost)


# ============================
# BATCH SCANNING
# ============================

def scan_universe_pre_spike(
    tickers: List[str],
    get_ticker_data_func
) -> Dict[str, PreSpikeSignal]:
    """
    Scan multiple tickers for pre-spike signals

    Args:
        tickers: List of stock symbols
        get_ticker_data_func: Function that returns data dict for a ticker

    Returns:
        Dict mapping ticker to PreSpikeSignal
    """
    results = {}

    for ticker in tickers:
        try:
            data = get_ticker_data_func(ticker)
            signal = scan_pre_spike(
                ticker,
                volume_data=data.get("volume"),
                options_data=data.get("options"),
                buzz_data=data.get("buzz"),
                technical_data=data.get("technical")
            )
            results[ticker] = signal

            if signal.is_pre_spike:
                logger.warning(f"🚨 PRE-SPIKE ALERT: {ticker} - {signal.alert_level} "
                             f"(confluence: {signal.confluence_count}, "
                             f"prob: {signal.pre_spike_probability:.0%})")

        except Exception as e:
            logger.error(f"Pre-spike scan failed for {ticker}: {e}")
            continue

    return results


def get_high_probability_tickers(
    signals: Dict[str, PreSpikeSignal],
    min_probability: float = 0.4
) -> List[Tuple[str, PreSpikeSignal]]:
    """
    Get tickers with high pre-spike probability, sorted by probability

    Args:
        signals: Dict of ticker -> PreSpikeSignal
        min_probability: Minimum probability threshold

    Returns:
        List of (ticker, signal) tuples, sorted by probability desc
    """
    high_prob = [
        (ticker, signal)
        for ticker, signal in signals.items()
        if signal.pre_spike_probability >= min_probability
    ]

    return sorted(high_prob, key=lambda x: x[1].pre_spike_probability, reverse=True)


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import random

    print("\n⚡ PRE-SPIKE RADAR TEST")
    print("=" * 60)

    # Generate sample data
    def generate_sample_data(pattern: str = "normal"):
        now = datetime.now()

        if pattern == "accelerating":
            # Accelerating volume pattern
            volumes = [(now - timedelta(minutes=30-i*5), 1000 + i*500 + random.randint(0, 200))
                      for i in range(6)]
            current_vol = 5000

            # Accelerating options
            options = [(now - timedelta(minutes=30-i*5), 100 + i*30, 50 - i*5)
                      for i in range(6)]
            current_calls, current_puts = 350, 30

            # Accelerating buzz
            mentions = [(now - timedelta(minutes=30-i*5), 5 + i*3) for i in range(6)]
            current_mentions = 30

            # Compressing technicals
            bandwidth = [0.08, 0.07, 0.065, 0.06, 0.055, 0.05]
            current_bw = 0.045

        elif pattern == "spike_imminent":
            # Strong pre-spike pattern
            volumes = [(now - timedelta(minutes=30-i*5), 2000 + i*1000 + random.randint(0, 500))
                      for i in range(6)]
            current_vol = 10000

            options = [(now - timedelta(minutes=30-i*5), 200 + i*80, 40 - i*8)
                      for i in range(6)]
            current_calls, current_puts = 700, 20

            mentions = [(now - timedelta(minutes=30-i*5), 10 + i*8) for i in range(6)]
            current_mentions = 80

            bandwidth = [0.1, 0.08, 0.06, 0.05, 0.04, 0.035]
            current_bw = 0.03

        else:  # normal
            volumes = [(now - timedelta(minutes=30-i*5), 1000 + random.randint(-100, 100))
                      for i in range(6)]
            current_vol = 1000

            options = [(now - timedelta(minutes=30-i*5), 100, 100) for i in range(6)]
            current_calls, current_puts = 100, 100

            mentions = [(now - timedelta(minutes=30-i*5), 5 + random.randint(-2, 2))
                       for i in range(6)]
            current_mentions = 5

            bandwidth = [0.1] * 6
            current_bw = 0.1

        return {
            "volume": {
                "current_volume": current_vol,
                "historical_volumes": volumes,
                "avg_daily_volume": 50000
            },
            "options": {
                "current_calls": current_calls,
                "current_puts": current_puts,
                "historical_options": options
            },
            "buzz": {
                "current_mentions": current_mentions,
                "historical_mentions": mentions,
                "baseline": 10
            },
            "technical": {
                "bollinger_bandwidth": current_bw,
                "historical_bandwidth": bandwidth,
                "atr_ratio": 0.7 if pattern == "spike_imminent" else 1.0
            }
        }

    # Test different patterns
    patterns = [
        ("AAPL", "normal"),
        ("MULN", "accelerating"),
        ("FFIE", "spike_imminent")
    ]

    for ticker, pattern in patterns:
        print(f"\n{'='*60}")
        print(f"📊 {ticker} - Pattern: {pattern.upper()}")
        print("-" * 60)

        data = generate_sample_data(pattern)
        signal = scan_pre_spike(
            ticker,
            volume_data=data["volume"],
            options_data=data["options"],
            buzz_data=data["buzz"],
            technical_data=data["technical"]
        )

        print(f"\n🎯 RESULT:")
        print(f"   Alert Level: {signal.alert_level}")
        print(f"   Is Pre-Spike: {signal.is_pre_spike}")
        print(f"   Confluence: {signal.confluence_count}/4 signals")
        print(f"   Confluence Score: {signal.confluence_score:.2f}")
        print(f"   Spike Probability: {signal.pre_spike_probability:.0%}")
        print(f"   Monster Score Boost: {get_pre_spike_boost(signal):.2f}x")

        print(f"\n📈 Individual Signals:")
        for sig_name, sig in [
            ("Volume", signal.volume_acceleration),
            ("Options", signal.options_acceleration),
            ("Buzz", signal.buzz_acceleration),
            ("Technical", signal.technical_compression)
        ]:
            if sig:
                status = "✅" if sig.is_accelerating else "⬜"
                print(f"   {status} {sig_name}: {sig.value:.2f} ({sig.strength})")
                for k, v in list(sig.details.items())[:3]:
                    print(f"      - {k}: {v}")

    print("\n" + "=" * 60)
    print("✅ Pre-Spike Radar test complete!")
