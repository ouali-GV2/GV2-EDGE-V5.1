"""
GV2-EDGE V6.1 Ingestors
=======================

Data ingestion modules for real sources.
NO LLM simulation - 100% real APIs.

Modules:
- sec_filings_ingestor: SEC EDGAR 8-K + Form 4 (FREE)
- global_news_ingestor: Finnhub general news (Phase 2)
- company_news_scanner: Finnhub company-specific (Phase 2)
- social_buzz_engine: Reddit + StockTwits (Phase 3)
"""

from .sec_filings_ingestor import (
    SECFiling,
    InsiderTransaction,
    SEC8KIngestor,
    SECForm4Ingestor,
    SECIngestor,
    get_cik_mapper,
)

__all__ = [
    "SECFiling",
    "InsiderTransaction",
    "SEC8KIngestor",
    "SECForm4Ingestor",
    "SECIngestor",
    "get_cik_mapper",
]
