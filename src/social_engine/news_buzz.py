from datetime import datetime, timedelta
from collections import defaultdict

from utils.api_guard import safe_get
from utils.cache import Cache
from utils.logger import get_logger

from config import FINNHUB_API_KEY

logger = get_logger("NEWS_BUZZ")

cache = Cache(ttl=180)  # refresh toutes les 3 minutes

FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"


# ============================
# Fetch company news
# ============================

def fetch_company_news(ticker, hours=6):

    now = datetime.utcnow()
    past = now - timedelta(hours=hours)

    params = {
        "symbol": ticker,
        "from": past.strftime("%Y-%m-%d"),
        "to": now.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY
    }

    r = safe_get(FINNHUB_NEWS, params=params)
    return r.json()


# ============================
# Buzz score computation
# ============================

def compute_buzz_score(ticker):

    cached = cache.get(f"buzz_{ticker}")
    if cached:
        return cached

    try:
        news = fetch_company_news(ticker)

        if not news:
            score = 0
        else:
            # simple but effective:
            # more articles in short time = more buzz
            score = min(1.0, len(news) / 10)

        data = {
            "ticker": ticker,
            "buzz_score": score,
            "article_count": len(news)
        }

        cache.set(f"buzz_{ticker}", data)

        logger.info(f"Buzz {ticker}: {len(news)} articles")

        return data

    except Exception as e:
        logger.error(f"Buzz error {ticker}: {e}")
        return {
            "ticker": ticker,
            "buzz_score": 0,
            "article_count": 0
        }


# ============================
# Batch helper
# ============================

def compute_many(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        results[t] = compute_buzz_score(t)

    return results


if __name__ == "__main__":
    print(compute_buzz_score("AAPL"))
