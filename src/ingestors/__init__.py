"""
GV2-EDGE V6.1 Ingestors
=======================

Data ingestion modules for real sources.
NO LLM simulation - 100% real APIs.

Modules:
- sec_filings_ingestor: SEC EDGAR 8-K + Form 4 (FREE)
- global_news_ingestor: Finnhub general + SEC 8-K discovery
- company_news_scanner: Finnhub company-specific deep scan
- social_buzz_engine: Reddit + StockTwits buzz detection
"""

from .sec_filings_ingestor import (
    SECIngestor,
    SEC8KIngestor,
    SECForm4Ingestor,
    SECFiling,
    InsiderTransaction,
    CIKMapper,
)

from .global_news_ingestor import (
    GlobalNewsIngestor,
    GlobalNewsItem,
    GlobalScanResult,
    get_global_ingestor,
    quick_global_scan,
)

from .company_news_scanner import (
    CompanyNewsScanner,
    CompanyNewsItem,
    CompanyScanResult,
    ScanPriority,
    get_company_scanner,
    quick_company_scan,
)

from .social_buzz_engine import (
    SocialBuzzEngine,
    BuzzMetrics,
    BaselineTracker,
    get_buzz_engine,
    quick_buzz_check,
)

__all__ = [
    # SEC Filings
    "SECIngestor",
    "SEC8KIngestor",
    "SECForm4Ingestor",
    "SECFiling",
    "InsiderTransaction",
    "CIKMapper",
    # Global News
    "GlobalNewsIngestor",
    "GlobalNewsItem",
    "GlobalScanResult",
    "get_global_ingestor",
    "quick_global_scan",
    # Company News
    "CompanyNewsScanner",
    "CompanyNewsItem",
    "CompanyScanResult",
    "ScanPriority",
    "get_company_scanner",
    "quick_company_scan",
    # Social Buzz
    "SocialBuzzEngine",
    "BuzzMetrics",
    "BaselineTracker",
    "get_buzz_engine",
    "quick_buzz_check",
]
