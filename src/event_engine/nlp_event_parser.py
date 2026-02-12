import json
from datetime import datetime

from config import GROK_API_KEY
from utils.api_guard import safe_post
from utils.logger import get_logger
from utils.data_validator import validate_event

logger = get_logger("NLP_EVENT_PARSER")

GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


SYSTEM_PROMPT = """
You are a financial event extraction engine for small-cap US stocks.

From the given text, extract ONLY real bullish catalysts.

EVENT TYPES (use EXACTLY these names):

TIER 1 - CRITICAL IMPACT (impact: 0.90-1.00):
- FDA_APPROVAL: Drug/device approved by FDA
- PDUFA_DECISION: Positive PDUFA decision (FDA deadline met positively)
- BUYOUT_CONFIRMED: Confirmed acquisition/buyout announcement

TIER 2 - HIGH IMPACT (impact: 0.75-0.89):
- FDA_TRIAL_POSITIVE: Positive Phase II/III trial results, met endpoints
- BREAKTHROUGH_DESIGNATION: FDA Breakthrough Therapy designation
- FDA_FAST_TRACK: FDA Fast Track designation granted
- MERGER_ACQUISITION: M&A announcement, takeover bid
- EARNINGS_BEAT_BIG: Earnings beat >20% above estimates
- MAJOR_CONTRACT: Large contract win (>$50M or transformational)

TIER 3 - MEDIUM-HIGH IMPACT (impact: 0.60-0.74):
- GUIDANCE_RAISE: Company raises forward guidance
- EARNINGS_BEAT: Standard earnings beat (<20% above estimates)
- PARTNERSHIP: Strategic partnership/collaboration announced
- PRICE_TARGET_RAISE: Significant price target increase by analyst

TIER 4 - MEDIUM IMPACT (impact: 0.45-0.59):
- ANALYST_UPGRADE: Analyst upgrades rating (Sell→Hold, Hold→Buy)
- SHORT_SQUEEZE_SIGNAL: Short squeeze setup or trigger
- UNUSUAL_VOLUME_NEWS: News explaining unusual volume spike

TIER 5 - SPECULATIVE (impact: 0.30-0.44):
- BUYOUT_RUMOR: Unconfirmed buyout/acquisition rumor
- SOCIAL_MEDIA_SURGE: Viral social media attention (WSB, Twitter)
- BREAKING_POSITIVE: Other positive breaking news

For each event return JSON list:

[
 {
  "ticker": "XYZ",
  "type": "EVENT_TYPE",
  "impact": 0.0 to 1.0,
  "date": "YYYY-MM-DD",
  "summary": "short explanation"
 }
]

IMPORTANT:
- Use impact scores aligned with the tiers above
- FDA_APPROVAL should be ~0.95, ANALYST_UPGRADE should be ~0.50
- Only output valid JSON. No text outside JSON.
"""


def call_grok(text):
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    r = safe_post(GROK_ENDPOINT, json=payload, headers=headers)
    return r.json()


def parse_events_from_text(text):
    try:
        response = call_grok(text)

        content = response["choices"][0]["message"]["content"]

        events = json.loads(content)

        clean_events = []

        for e in events:
            if validate_event(e):
                clean_events.append(e)

        logger.info(f"NLP extracted {len(clean_events)} events")

        return clean_events

    except Exception as e:
        logger.error(f"NLP parse failed: {e}")
        return []


# ============================
# Batch helper
# ============================

def parse_many_texts(texts):
    all_events = []

    for t in texts:
        ev = parse_events_from_text(t)
        all_events.extend(ev)

    return all_events


if __name__ == "__main__":
    sample = """
    TCGL announces FDA approval for its new cancer drug.
    FEED receives major $200M contract with US government.
    """

    print(parse_events_from_text(sample))
