"""
Risk Guard Module - Unified Risk Management for GV2-EDGE

Components:
- DilutionDetector: SEC filings, offerings, toxic financing detection
- ComplianceChecker: Exchange deficiencies, delisting risk monitoring
- HaltMonitor: Trading halts, LULD tracking, halt prediction
- UnifiedGuard: Central orchestrator combining all risk modules

Usage:
    from src.risk_guard import get_unified_guard, RiskLevel, TradeAction

    guard = get_unified_guard()

    # Full assessment
    assessment = await guard.assess(
        ticker,
        current_price=price,
        volatility=vol
    )

    if assessment.is_blocked:
        # Don't trade
        print(f"Blocked: {assessment.block_reasons}")
    else:
        # Adjust position size
        adjusted_size = base_size * assessment.position_multiplier

    # Quick check
    can_trade, reason = guard.quick_check(ticker)

    # Monitor halt
    guard.on_halt(ticker, HaltCode.LUDP)
"""

# Dilution Detection
from .dilution_detector import (
    DilutionDetector,
    DilutionProfile,
    DilutionEvent,
    DilutionType,
    DilutionRisk,
    get_dilution_detector,
)

# Compliance Checking
from .compliance_checker import (
    ComplianceChecker,
    ComplianceProfile,
    ComplianceEvent,
    ComplianceIssue,
    ComplianceStatus,
    ComplianceRisk,
    get_compliance_checker,
)

# Halt Monitoring
from .halt_monitor import (
    HaltMonitor,
    HaltProfile,
    HaltEvent,
    HaltPrediction,
    HaltCode,
    HaltReason,
    HaltRisk,
    LULDState,
    get_halt_monitor,
)

# Unified Guard
from .unified_guard import (
    UnifiedGuard,
    RiskAssessment,
    RiskFlag,
    RiskLevel,
    RiskCategory,
    TradeAction,
    GuardConfig,
    get_unified_guard,
)

__all__ = [
    # Dilution
    "DilutionDetector",
    "DilutionProfile",
    "DilutionEvent",
    "DilutionType",
    "DilutionRisk",
    "get_dilution_detector",
    # Compliance
    "ComplianceChecker",
    "ComplianceProfile",
    "ComplianceEvent",
    "ComplianceIssue",
    "ComplianceStatus",
    "ComplianceRisk",
    "get_compliance_checker",
    # Halt
    "HaltMonitor",
    "HaltProfile",
    "HaltEvent",
    "HaltPrediction",
    "HaltCode",
    "HaltReason",
    "HaltRisk",
    "LULDState",
    "get_halt_monitor",
    # Unified
    "UnifiedGuard",
    "RiskAssessment",
    "RiskFlag",
    "RiskLevel",
    "RiskCategory",
    "TradeAction",
    "GuardConfig",
    "get_unified_guard",
]
