import json
from utils.api_guard import safe_post
from utils.logger import get_logger
from utils.cache import Cache

from config import GROK_API_KEY

logger = get_logger("GROK_SENTIMENT")

cache = Cache(ttl=120)  # refresh toutes les 2 min

GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


SYSTEM_PROMPT = """
You are a market sentiment engine.

Analyze social sentiment (Twitter/X + trading discussions)
for the given ticker.

Return JSON only:

{
 "ticker": "XYZ",
 "sentiment": -1.0 to 1.0,
 "confidence": 0.0 to 1.0,
 "summary": "short explanation"
}

Where:
- sentiment > 0 bullish
- sentiment < 0 bearish
"""


# ============================
# Call Grok
# ============================

def call_grok(prompt):

    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    r = safe_post(GROK_ENDPOINT, json=payload, headers=headers)
    return r.json()


# ============================
# Sentiment fetch
# ============================

def get_social_sentiment(ticker):

    cached = cache.get(f"sent_{ticker}")
    if cached:
        return cached

    try:
        prompt = f"""
        Analyze real-time social sentiment for stock {ticker}.
        Focus on traders activity, hype, bullish/bearish tone.
        """

        response = call_grok(prompt)

        content = response["choices"][0]["message"]["content"]

        data = json.loads(content)

        cache.set(f"sent_{ticker}", data)

        logger.info(f"Sentiment fetched for {ticker}")

        return data

    except Exception as e:
        logger.error(f"Sentiment error {ticker}: {e}")
        return {
            "ticker": ticker,
            "sentiment": 0,
            "confidence": 0,
            "summary": "no data"
        }


# ============================
# Batch helper
# ============================

def get_many(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        results[t] = get_social_sentiment(t)

    return results


if __name__ == "__main__":
    print(get_social_sentiment("AAPL"))
