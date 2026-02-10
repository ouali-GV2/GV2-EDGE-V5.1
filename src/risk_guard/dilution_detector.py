"""
Dilution Detector - SEC Filings and Offering Analysis

Detects dilution risks from:
- S-3 shelf registrations (potential future dilution)
- Prospectus supplements (imminent offering)
- ATM (At-The-Market) programs
- Direct offerings / registered direct
- PIPE deals
- Warrant exercises
- Convertible note conversions

Risk levels:
- CRITICAL: Active offering announced, pricing imminent
- HIGH: S-3 filed with intent to use, ATM active
- MEDIUM: Shelf registration on file, warrants outstanding
- LOW: Old shelf (>1 year), no recent activity
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class DilutionType(Enum):
    """Types of dilution events."""
    S3_SHELF = "S3_SHELF"                    # Shelf registration
    PROSPECTUS_SUPPLEMENT = "PROSPECTUS"     # 424B filing
    ATM_PROGRAM = "ATM"                      # At-the-market offering
    DIRECT_OFFERING = "DIRECT"               # Registered direct
    PIPE_DEAL = "PIPE"                       # Private investment in public equity
    WARRANT_EXERCISE = "WARRANT"             # Warrant exercise/reset
    CONVERTIBLE_NOTE = "CONVERTIBLE"         # Convert to shares
    SECONDARY_OFFERING = "SECONDARY"         # Follow-on offering
    MIXED_SHELF = "MIXED_SHELF"              # Debt + equity shelf


class DilutionRisk(Enum):
    """Risk severity levels."""
    CRITICAL = "CRITICAL"   # Imminent dilution, avoid entirely
    HIGH = "HIGH"           # Active program, reduce position
    MEDIUM = "MEDIUM"       # Potential risk, monitor closely
    LOW = "LOW"             # Minimal current risk
    NONE = "NONE"           # No dilution detected


@dataclass
class DilutionEvent:
    """Represents a dilution event or filing."""
    ticker: str
    event_type: DilutionType
    filing_date: datetime
    effective_date: Optional[datetime] = None

    # Offering details
    shares_registered: Optional[int] = None
    dollar_amount: Optional[float] = None
    price_per_share: Optional[float] = None

    # Shelf details
    shelf_capacity_remaining: Optional[float] = None
    shelf_expiry: Optional[datetime] = None

    # Status
    is_active: bool = True
    is_completed: bool = False

    # SEC filing reference
    accession_number: Optional[str] = None
    filing_url: Optional[str] = None

    # Computed
    dilution_percent: Optional[float] = None  # vs current shares outstanding

    def days_since_filing(self) -> int:
        """Days since the filing date."""
        return (datetime.now() - self.filing_date).days

    def is_stale(self, days: int = 365) -> bool:
        """Check if filing is old/stale."""
        return self.days_since_filing() > days


@dataclass
class DilutionProfile:
    """Complete dilution profile for a ticker."""
    ticker: str
    risk_level: DilutionRisk = DilutionRisk.NONE
    risk_score: float = 0.0  # 0-100

    # Active events
    events: List[DilutionEvent] = field(default_factory=list)

    # Aggregated info
    total_shelf_capacity: float = 0.0
    active_atm_capacity: float = 0.0
    warrants_outstanding: int = 0
    convertible_debt: float = 0.0

    # Shares info
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    last_filing_date: Optional[datetime] = None

    # Flags
    has_active_offering: bool = False
    has_active_atm: bool = False
    has_recent_s3: bool = False
    has_toxic_financing: bool = False  # Death spiral converts, etc.

    def get_block_reason(self) -> Optional[str]:
        """Get reason string if trade should be blocked."""
        if self.risk_level == DilutionRisk.CRITICAL:
            if self.has_active_offering:
                return "ACTIVE_OFFERING"
            if self.has_toxic_financing:
                return "TOXIC_FINANCING"
            return "CRITICAL_DILUTION"
        return None

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on risk."""
        multipliers = {
            DilutionRisk.CRITICAL: 0.0,   # Block entirely
            DilutionRisk.HIGH: 0.25,      # 25% of normal
            DilutionRisk.MEDIUM: 0.50,    # 50% of normal
            DilutionRisk.LOW: 0.75,       # 75% of normal
            DilutionRisk.NONE: 1.0        # Full size
        }
        return multipliers.get(self.risk_level, 1.0)


# SEC filing type patterns
SEC_FILING_PATTERNS = {
    DilutionType.S3_SHELF: [
        r"S-3",
        r"S-3/A",
        r"S-3ASR",  # Automatic shelf registration
    ],
    DilutionType.PROSPECTUS_SUPPLEMENT: [
        r"424B[1-5]",
        r"FWP",      # Free writing prospectus
        r"PROSUP",
    ],
    DilutionType.ATM_PROGRAM: [
        r"at.the.market",
        r"ATM\s+(?:program|offering|agreement)",
        r"equity\s+distribution\s+agreement",
        r"sales\s+agreement",
    ],
    DilutionType.DIRECT_OFFERING: [
        r"registered\s+direct",
        r"direct\s+offering",
        r"RDO",
    ],
    DilutionType.PIPE_DEAL: [
        r"private\s+placement",
        r"PIPE",
        r"private\s+investment",
    ],
    DilutionType.WARRANT_EXERCISE: [
        r"warrant\s+(?:exercise|inducement|amendment)",
        r"exercise\s+price\s+(?:reduction|reset)",
    ],
    DilutionType.CONVERTIBLE_NOTE: [
        r"convertible\s+(?:note|debenture|bond)",
        r"conversion\s+(?:price|shares)",
    ],
}

# Toxic financing patterns (death spiral, etc.)
TOXIC_PATTERNS = [
    r"variable\s+(?:rate|price)\s+convert",
    r"floating\s+conversion",
    r"reset\s+(?:provision|price)",
    r"equity\s+line",
    r"ELOC",
    r"death\s+spiral",
]


class DilutionDetector:
    """
    Detects and tracks dilution risks for tickers.

    Usage:
        detector = DilutionDetector()

        # Check single ticker
        profile = await detector.analyze(ticker, sec_filings)
        if profile.risk_level == DilutionRisk.CRITICAL:
            block_trade()

        # Batch check
        profiles = await detector.analyze_batch(tickers)
    """

    def __init__(self):
        # Cache of dilution profiles
        self._profiles: Dict[str, DilutionProfile] = {}

        # Known toxic tickers (manually flagged)
        self._toxic_tickers: Set[str] = set()

        # Cache TTL
        self._cache_ttl = timedelta(hours=1)

        # Compile regex patterns
        self._filing_patterns = self._compile_patterns(SEC_FILING_PATTERNS)
        self._toxic_patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    def _compile_patterns(
        self,
        patterns: Dict[DilutionType, List[str]]
    ) -> Dict[DilutionType, List[re.Pattern]]:
        """Compile regex patterns."""
        compiled = {}
        for dtype, pattern_list in patterns.items():
            compiled[dtype] = [
                re.compile(p, re.IGNORECASE)
                for p in pattern_list
            ]
        return compiled

    async def analyze(
        self,
        ticker: str,
        sec_filings: Optional[List[Dict]] = None,
        shares_outstanding: Optional[int] = None,
        float_shares: Optional[int] = None,
        force_refresh: bool = False
    ) -> DilutionProfile:
        """
        Analyze dilution risk for a ticker.

        Args:
            ticker: Stock ticker
            sec_filings: List of SEC filing dicts with keys:
                - form_type: "S-3", "424B5", etc.
                - filed_date: datetime or str
                - description: Filing description
                - accession_number: SEC accession number
                - url: Link to filing
            shares_outstanding: Current shares outstanding
            float_shares: Current float
            force_refresh: Force cache refresh

        Returns:
            DilutionProfile with risk assessment
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._profiles:
            cached = self._profiles[ticker]
            if datetime.now() - cached.last_updated < self._cache_ttl:
                return cached

        # Create new profile
        profile = DilutionProfile(
            ticker=ticker,
            shares_outstanding=shares_outstanding,
            float_shares=float_shares
        )

        # Check if manually flagged as toxic
        if ticker in self._toxic_tickers:
            profile.has_toxic_financing = True
            profile.risk_level = DilutionRisk.CRITICAL
            profile.risk_score = 100.0
            self._profiles[ticker] = profile
            return profile

        # Analyze SEC filings if provided
        if sec_filings:
            await self._analyze_filings(profile, sec_filings)

        # Calculate overall risk
        self._calculate_risk(profile)

        # Cache result
        self._profiles[ticker] = profile

        return profile

    async def _analyze_filings(
        self,
        profile: DilutionProfile,
        filings: List[Dict]
    ) -> None:
        """Analyze SEC filings for dilution events."""
        for filing in filings:
            event = self._parse_filing(profile.ticker, filing)
            if event:
                profile.events.append(event)

                # Update flags
                if event.event_type == DilutionType.ATM_PROGRAM and event.is_active:
                    profile.has_active_atm = True
                    if event.shelf_capacity_remaining:
                        profile.active_atm_capacity += event.shelf_capacity_remaining

                if event.event_type == DilutionType.S3_SHELF:
                    if event.days_since_filing() < 90:
                        profile.has_recent_s3 = True
                    if event.shelf_capacity_remaining:
                        profile.total_shelf_capacity += event.shelf_capacity_remaining

                if event.event_type in [
                    DilutionType.DIRECT_OFFERING,
                    DilutionType.PROSPECTUS_SUPPLEMENT
                ]:
                    if event.days_since_filing() < 7:
                        profile.has_active_offering = True

                if event.filing_date:
                    if not profile.last_filing_date or event.filing_date > profile.last_filing_date:
                        profile.last_filing_date = event.filing_date

        # Check for toxic patterns in filing descriptions
        for filing in filings:
            desc = filing.get("description", "")
            for pattern in self._toxic_patterns:
                if pattern.search(desc):
                    profile.has_toxic_financing = True
                    break

    def _parse_filing(
        self,
        ticker: str,
        filing: Dict
    ) -> Optional[DilutionEvent]:
        """Parse a single SEC filing into a DilutionEvent."""
        form_type = filing.get("form_type", "")
        description = filing.get("description", "")

        # Determine dilution type
        event_type = None
        for dtype, patterns in self._filing_patterns.items():
            for pattern in patterns:
                if pattern.search(form_type) or pattern.search(description):
                    event_type = dtype
                    break
            if event_type:
                break

        if not event_type:
            return None

        # Parse filing date
        filed_date = filing.get("filed_date")
        if isinstance(filed_date, str):
            try:
                filed_date = datetime.fromisoformat(filed_date.replace("Z", "+00:00"))
            except ValueError:
                filed_date = datetime.now()
        elif not isinstance(filed_date, datetime):
            filed_date = datetime.now()

        # Extract dollar amounts if present
        dollar_amount = self._extract_dollar_amount(description)
        shares = self._extract_share_count(description)

        event = DilutionEvent(
            ticker=ticker,
            event_type=event_type,
            filing_date=filed_date,
            dollar_amount=dollar_amount,
            shares_registered=shares,
            accession_number=filing.get("accession_number"),
            filing_url=filing.get("url")
        )

        # Set shelf capacity for S-3 filings
        if event_type == DilutionType.S3_SHELF and dollar_amount:
            event.shelf_capacity_remaining = dollar_amount
            # Shelves typically valid for 3 years
            event.shelf_expiry = filed_date + timedelta(days=365 * 3)

        return event

    def _extract_dollar_amount(self, text: str) -> Optional[float]:
        """Extract dollar amount from text."""
        patterns = [
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)",
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:billion|B)",
            r"([\d,]+(?:\.\d+)?)\s*(?:million|M)\s*(?:dollars|\$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(",", ""))
                if "billion" in text.lower() or "B" in match.group(0):
                    amount *= 1_000_000_000
                else:
                    amount *= 1_000_000
                return amount

        return None

    def _extract_share_count(self, text: str) -> Optional[int]:
        """Extract share count from text."""
        patterns = [
            r"([\d,]+)\s*shares",
            r"([\d,]+(?:\.\d+)?)\s*(?:million|M)\s*shares",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                count = float(match.group(1).replace(",", ""))
                if "million" in text.lower() or "M" in match.group(0):
                    count *= 1_000_000
                return int(count)

        return None

    def _calculate_risk(self, profile: DilutionProfile) -> None:
        """Calculate overall risk level and score."""
        score = 0.0

        # Critical flags
        if profile.has_toxic_financing:
            score += 50
        if profile.has_active_offering:
            score += 40

        # High risk factors
        if profile.has_active_atm:
            score += 25
            # Additional risk if ATM capacity is high vs market cap
            if profile.active_atm_capacity > 0:
                score += min(15, profile.active_atm_capacity / 10_000_000)

        if profile.has_recent_s3:
            score += 20

        # Medium risk factors
        if profile.total_shelf_capacity > 50_000_000:
            score += 15
        elif profile.total_shelf_capacity > 0:
            score += 10

        # Event-based scoring
        for event in profile.events:
            if event.is_stale(365):
                continue  # Old events don't count much

            days = event.days_since_filing()
            recency_multiplier = max(0.1, 1.0 - (days / 180))  # Decay over 6 months

            event_scores = {
                DilutionType.PROSPECTUS_SUPPLEMENT: 30,
                DilutionType.DIRECT_OFFERING: 25,
                DilutionType.PIPE_DEAL: 20,
                DilutionType.ATM_PROGRAM: 15,
                DilutionType.S3_SHELF: 10,
                DilutionType.WARRANT_EXERCISE: 15,
                DilutionType.CONVERTIBLE_NOTE: 20,
            }

            base_score = event_scores.get(event.event_type, 5)
            score += base_score * recency_multiplier

        # Cap at 100
        profile.risk_score = min(100.0, score)

        # Determine risk level
        if profile.risk_score >= 70 or profile.has_toxic_financing or profile.has_active_offering:
            profile.risk_level = DilutionRisk.CRITICAL
        elif profile.risk_score >= 45:
            profile.risk_level = DilutionRisk.HIGH
        elif profile.risk_score >= 25:
            profile.risk_level = DilutionRisk.MEDIUM
        elif profile.risk_score >= 10:
            profile.risk_level = DilutionRisk.LOW
        else:
            profile.risk_level = DilutionRisk.NONE

    async def analyze_batch(
        self,
        tickers: List[str],
        filings_by_ticker: Optional[Dict[str, List[Dict]]] = None
    ) -> Dict[str, DilutionProfile]:
        """Analyze multiple tickers concurrently."""
        filings_by_ticker = filings_by_ticker or {}

        tasks = [
            self.analyze(ticker, filings_by_ticker.get(ticker))
            for ticker in tickers
        ]

        profiles = await asyncio.gather(*tasks)

        return {
            ticker: profile
            for ticker, profile in zip(tickers, profiles)
        }

    def flag_toxic(self, ticker: str) -> None:
        """Manually flag a ticker as having toxic financing."""
        self._toxic_tickers.add(ticker.upper())
        # Invalidate cache
        if ticker.upper() in self._profiles:
            del self._profiles[ticker.upper()]

    def unflag_toxic(self, ticker: str) -> None:
        """Remove toxic flag from a ticker."""
        self._toxic_tickers.discard(ticker.upper())
        if ticker.upper() in self._profiles:
            del self._profiles[ticker.upper()]

    def get_cached_profile(self, ticker: str) -> Optional[DilutionProfile]:
        """Get cached profile without refresh."""
        return self._profiles.get(ticker.upper())

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cache for a ticker or all tickers."""
        if ticker:
            self._profiles.pop(ticker.upper(), None)
        else:
            self._profiles.clear()

    def get_high_risk_tickers(self) -> List[str]:
        """Get list of tickers with HIGH or CRITICAL risk."""
        return [
            ticker for ticker, profile in self._profiles.items()
            if profile.risk_level in [DilutionRisk.HIGH, DilutionRisk.CRITICAL]
        ]


# Singleton instance
_detector: Optional[DilutionDetector] = None


def get_dilution_detector() -> DilutionDetector:
    """Get singleton DilutionDetector instance."""
    global _detector
    if _detector is None:
        _detector = DilutionDetector()
    return _detector
