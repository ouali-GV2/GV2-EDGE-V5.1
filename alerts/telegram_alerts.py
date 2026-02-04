import requests

from utils.logger import get_logger
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger("TELEGRAM_ALERTS")

TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


# ============================
# Send message
# ============================

def send_message(text):
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }

        r = requests.post(TELEGRAM_URL, json=payload, timeout=5)

        if r.status_code != 200:
            logger.warning(f"Telegram error: {r.text}")

    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


# ============================
# Signal alert
# ============================

def send_signal_alert(signal, position=None):
    msg = f"""
🚀 *GV2-EDGE SIGNAL*

📊 Ticker: `{signal['ticker']}`
⚡ Signal: *{signal['signal']}*
🎯 Score: `{signal['monster_score']:.2f}`
📈 Confidence: `{signal['confidence']:.2f}`

"""

    if position:
        msg += f"""
💰 Entry: `{position['entry']}`
🛑 Stop: `{position['stop']}`
📦 Shares: `{position['shares']}`
⚖ Risk: `${position['risk_amount']}`
🕒 Session: `{position['session']}`
"""

    send_message(msg)


# ============================
# System alert (future)
# ============================

def send_system_alert(text):
    msg = f"⚠️ *GV2-EDGE SYSTEM*\n{text}"
    send_message(msg)


if __name__ == "__main__":
    send_message("GV2-EDGE Telegram connected ✅")
