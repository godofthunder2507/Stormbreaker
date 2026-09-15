"""
XAUUSD Hourly Market Report - descriptive only, sent via the same Telegram
chat as Stormbreaker's signal bot.

This is a completely SEPARATE script from bot.py. It does not import from,
call, or modify anything in bot.py - Stormbreaker's signal engine (analyse(),
MIN_CONFIRMATIONS, SL/TP sizing, cooldown logic, etc.) is untouched by this
file. This script's only job is to post a plain descriptive market summary
once an hour: current price, recent price action, a qualitative trend/
volatility read, and any fresh gold-relevant news. It NEVER sends a BUY/SELL/
HOLD call, a confidence score, an entry price, a stop-loss, or a take-profit
- if you want that, that's what bot.py already does separately.

Data sources ("global" per Sam's request - as many reasonably-available free
sources as make sense, without adding paid/complex integrations for a simple
hourly text report):
  - Twelve Data (same provider bot.py already uses) for price/candles.
  - Finnhub Market News (general + forex categories, same as bot.py) for
    gold-relevant headlines, using the same keyword list so "gold-relevant"
    means the same thing in both places.

Secrets required (reuses what's already configured for bot.py - no new
secrets needed unless you want to add more sources later):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVE_DATA_API_KEY, FINNHUB_API_KEY
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

SYMBOL = "XAU/USD"
CANDLE_INTERVAL = "1h"
CANDLES_TO_FETCH = 200
ATR_PERIOD = 14
EMA_FAST, EMA_SLOW = 50, 200
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 20.0

NEWS_CATEGORIES = ["general", "forex"]
GOLD_NEWS_KEYWORDS = [
    "gold", "xau", "bullion", "safe haven", "safe-haven",
    "fed", "fomc", "powell", "rate hike", "rate cut", "interest rate",
    "inflation", "cpi", "pce", "nonfarm", "non-farm", "jobs report", "payrolls",
    "dollar", "usd", "treasury yield", "treasury yields",
    "tariff", "war", "geopolitical", "recession", "sanctions",
]
NEWS_LOOKBACK_HOURS = 2  # only show news fresh enough to matter for an hourly cadence
NEWS_MAX_ITEMS = 3

REPORT_STATE_FILE = Path(__file__).parent / "report_state.json"


# =====================================================================
# DATA
# =====================================================================
def get_candles(symbol, interval, count):
    params = {
        "symbol": symbol, "interval": interval, "outputsize": count,
        "apikey": TWELVE_DATA_API_KEY, "timezone": "UTC",
    }
    resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Twelve Data error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def adx(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_sm = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / atr_sm
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / atr_sm
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def describe_market(df):
    close = df["close"]
    last_price = close.iloc[-1]

    last_24h = df[df["time"] >= df["time"].iloc[-1] - timedelta(hours=24)]
    change_24h = last_price - last_24h["close"].iloc[0] if len(last_24h) > 1 else 0.0
    change_24h_pct = (change_24h / last_24h["close"].iloc[0] * 100) if len(last_24h) > 1 else 0.0
    high_24h, low_24h = last_24h["high"].max(), last_24h["low"].min()

    ema_fast, ema_slow = ema(close, EMA_FAST).iloc[-1], ema(close, EMA_SLOW).iloc[-1]
    trend_desc = "uptrend (short-term EMA above long-term EMA)" if ema_fast > ema_slow else \
                 "downtrend (short-term EMA below long-term EMA)" if ema_fast < ema_slow else "flat/no clear trend"

    atr_now = atr(df, ATR_PERIOD).iloc[-1]
    adx_now = adx(df, ADX_PERIOD).iloc[-1]
    vol_desc = "trending / directional" if adx_now > ADX_TREND_THRESHOLD else "choppy / range-bound"

    return {
        "last_price": last_price, "change_24h": change_24h, "change_24h_pct": change_24h_pct,
        "high_24h": high_24h, "low_24h": low_24h, "trend_desc": trend_desc,
        "atr_now": atr_now, "adx_now": adx_now, "vol_desc": vol_desc,
    }


# =====================================================================
# NEWS
# =====================================================================
def fetch_gold_news():
    if not FINNHUB_API_KEY:
        return []
    articles = []
    for category in NEWS_CATEGORIES:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": category, "token": FINNHUB_API_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            articles.extend(resp.json())
        except requests.RequestException as e:
            print(f"[News] Failed to fetch '{category}': {e}")
    return articles


def is_gold_relevant(article):
    text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
    return any(kw in text for kw in GOLD_NEWS_KEYWORDS)


def recent_gold_news(now_utc):
    cutoff = now_utc - timedelta(hours=NEWS_LOOKBACK_HOURS)
    articles = fetch_gold_news()
    seen_ids = set()
    relevant = []
    for a in articles:
        article_id = a.get("id")
        if article_id is not None and article_id in seen_ids:
            continue  # same article can appear in multiple categories (e.g. general + forex)
        ts = a.get("datetime")
        if not ts:
            continue
        article_time = datetime.fromtimestamp(ts, tz=timezone.utc)
        if article_time >= cutoff and is_gold_relevant(a):
            if article_id is not None:
                seen_ids.add(article_id)
            relevant.append(a)
    relevant.sort(key=lambda a: a.get("datetime", 0), reverse=True)
    return relevant[:NEWS_MAX_ITEMS]


# =====================================================================
# TELEGRAM
# =====================================================================
def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Skipped - secrets not configured.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[Telegram] Failed: {e}")
        return False


def format_report(market, news, now_utc):
    direction_word = "up" if market["change_24h"] > 0 else "down" if market["change_24h"] < 0 else "flat"
    lines = [
        "\U0001F4CA *XAUUSD HOURLY MARKET REPORT*",
        "_(Descriptive only - not a trade signal, no entry/stop/target)_",
        "",
        f"Price: `{market['last_price']:.2f}`",
        f"24h change: {direction_word} {abs(market['change_24h']):.2f} ({market['change_24h_pct']:+.2f}%)",
        f"24h range: {market['low_24h']:.2f} - {market['high_24h']:.2f}",
        f"Trend: {market['trend_desc']}",
        f"Volatility character: {market['vol_desc']} (ADX {market['adx_now']:.1f}, ATR {market['atr_now']:.2f})",
    ]
    if news:
        lines.append("")
        lines.append("*Gold-relevant news (last %dh):*" % NEWS_LOOKBACK_HOURS)
        for a in news:
            lines.append(f"- {a.get('headline', '(no headline)')} ({a.get('source', 'unknown')})")
    lines.append("")
    lines.append(f"_{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC_")
    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================
def main():
    now = datetime.now(timezone.utc)
    df = get_candles(SYMBOL, CANDLE_INTERVAL, CANDLES_TO_FETCH)
    market = describe_market(df)
    news = recent_gold_news(now)
    message = format_report(market, news, now)
    print(message)
    if send_telegram(message):
        print("Hourly report sent.")
    else:
        print("Report generated but not sent (check Telegram secrets).")


if __name__ == "__main__":
    main()
