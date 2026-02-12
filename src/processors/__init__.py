"""
GV2-EDGE V6.1 Processors
========================

Data processing modules.

Modules:
- keyword_filter: Fast regex pre-filter (NO LLM)
- ticker_extractor: Symbol extraction and validation
- nlp_classifier: Grok NLP for EVENT_TYPE classification ONLY
- catalyst_detector: Unified catalyst scoring (Phase 2)
"""

from .keyword_filter import (
    KeywordFilter,
    FilterResult,
    FilterPriority,
    get_keyword_filter,
    quick_filter,
    is_critical,
    is_noise,
)

from .ticker_extractor import (
    TickerExtractor,
    BatchTickerExtractor,
    ExtractedTicker,
    get_ticker_extractor,
    extract_tickers,
    has_ticker,
)

from .nlp_classifier import (
    NLPClassifier,
    ClassificationResult,
    EventType,
    EVENT_IMPACT,
    get_classifier,
    classify_news,
    get_event_impact,
    get_event_tier,
    get_tier,
)

__all__ = [
    # Keyword Filter
    "KeywordFilter",
    "FilterResult",
    "FilterPriority",
    "get_keyword_filter",
    "quick_filter",
    "is_critical",
    "is_noise",
    # Ticker Extractor
    "TickerExtractor",
    "BatchTickerExtractor",
    "ExtractedTicker",
    "get_ticker_extractor",
    "extract_tickers",
    "has_ticker",
    # NLP Classifier
    "NLPClassifier",
    "ClassificationResult",
    "EventType",
    "EVENT_IMPACT",
    "get_classifier",
    "classify_news",
    "get_event_impact",
    "get_event_tier",
    "get_tier",
]
