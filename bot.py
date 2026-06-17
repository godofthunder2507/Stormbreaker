"""
XAUUSD Signal Bot - single-file cloud version.

Runs on a schedule (GitHub Actions) instead of as an always-on desktop app.
Each run: fetch latest gold price data -> analyse with a multi-indicator
strategy -> send a Telegram alert if a BUY/SELL signal fires -> save state
for cooldown tracking -> exit.

Secrets (Telegram token/chat ID, Twelve Data API key) are read from
environment variables, which GitHub Actions injects from encrypted
repository secrets - nothing sensitive lives in this file.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =====================================================================
# CONFIG - secrets come from environment variables (GitHub Secrets)
# =====================================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

SYMBOL = "XAU/USD"
ENTRY_TIMEFRAME = "15min"
TREND_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 200

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_TREND_FAST, EMA_TREND_SLOW = 50, 200
ATR_PERIOD = 14
SUPPORT_RESISTANCE_LOOKBACK = 50
SR_PROXIMITY = 3.0   # how close price must be to S/R to count as "near" ($)

MIN_CONFIRMATIONS = 3          # out of 4 conditions
SIGNAL_COOLDOWN_MINUTES = 30
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

STATE_FILE = Path(__file__).parent / "state.json"


# =====================================================================
# MARKET DATA (Twelve Data free API)
# =====================================================================
def get_candles(symbol, interval, count):
    params = {"symbol": symbol, "interval": interval, "outputsize": count, "apikey": TWELVE_DATA_API_KEY}
    resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Twelve Data error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


# =====================================================================
# INDICATORS
# =====================================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(series, fast, slow, signal):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def find_support_resistance(df, lookback):
    recent = df.tail(lookback)
    highs, lows = recent["high"], recent["low"]
    last_close = df["close"].iloc[-1]
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    resistances = swing_highs[swing_highs > last_close]
    supports = swing_lows[swing_lows < last_close]
    nearest_resistance = resistances.min() if not resistances.empty else recent["high"].max()
    nearest_support = supports.max() if not supports.empty else recent["low"].min()
    return nearest_support, nearest_resistance


# =====================================================================
# SIGNAL ENGINE
# =====================================================================
@dataclass
class SignalResult:
    timestamp: datetime
    direction: str
    confidence: int
    entry_price: float
    stop_loss: float
    take_profit: float
    rsi_value: float
    trend: str
    support: float
    resistance: float
    reasons: list


def determine_trend(htf_df):
    close = htf_df["close"]
    fast, slow = ema(close, EMA_TREND_FAST).iloc[-1], ema(close, EMA_TREND_SLOW).iloc[-1]
    return "UP" if fast > slow else "DOWN" if fast < slow else "RANGE"


def analyse(entry_df, htf_df):
    close = entry_df["close"]
    last_price = close.iloc[-1]
    last_rsi = rsi(close, RSI_PERIOD).iloc[-1]
    macd_line, signal_line = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    bullish_cross = macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
    bearish_cross = macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]
    support, resistance = find_support_resistance(entry_df, SUPPORT_RESISTANCE_LOOKBACK)
    near_support = abs(last_price - support) <= SR_PROXIMITY
    near_resistance = abs(last_price - resistance) <= SR_PROXIMITY
    trend = determine_trend(htf_df)
    atr_value = atr(entry_df, ATR_PERIOD).iloc[-1]

    buy_reasons, sell_reasons = [], []
    if bullish_cross:
        buy_reasons.append("MACD bullish crossover")
    if RSI_OVERSOLD < last_rsi < RSI_OVERBOUGHT and last_rsi > 40:
        buy_reasons.append(f"RSI healthy at {last_rsi:.1f}")
    if near_support:
        buy_reasons.append(f"Price near support ({support:.2f})")
    if trend == "UP":
        buy_reasons.append("H1 trend is UP")

    if bearish_cross:
        sell_reasons.append("MACD bearish crossover")
    if RSI_OVERSOLD < last_rsi < RSI_OVERBOUGHT and last_rsi < 60:
        sell_reasons.append(f"RSI healthy at {last_rsi:.1f}")
    if near_resistance:
        sell_reasons.append(f"Price near resistance ({resistance:.2f})")
    if trend == "DOWN":
        sell_reasons.append("H1 trend is DOWN")

    direction, reasons, confidence = "HOLD", [], 0
    if len(buy_reasons) >= MIN_CONFIRMATIONS and len(buy_reasons) >= len(sell_reasons):
        direction, reasons, confidence = "BUY", buy_reasons, len(buy_reasons)
    elif len(sell_reasons) >= MIN_CONFIRMATIONS:
        direction, reasons, confidence = "SELL", sell_reasons, len(sell_reasons)

    if direction == "BUY":
        stop_loss = last_price - atr_value * ATR_SL_MULTIPLIER
        take_profit = last_price + atr_value * ATR_TP_MULTIPLIER
    elif direction == "SELL":
        stop_loss = last_price + atr_value * ATR_SL_MULTIPLIER
        take_profit = last_price - atr_value * ATR_TP_MULTIPLIER
    else:
        stop_loss = take_profit = 0.0

    return SignalResult(
        timestamp=datetime.now(), direction=direction, confidence=confidence,
        entry_price=round(last_price, 2), stop_loss=round(stop_loss, 2), take_profit=round(take_profit, 2),
        rsi_value=round(last_rsi, 2), trend=trend, support=round(support, 2), resistance=round(resistance, 2),
        reasons=reasons,
    )


# =====================================================================
# TELEGRAM
# =====================================================================
def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Skipped - secrets not configured.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[Telegram] Failed: {e}")
        return False


def format_message(signal):
    emoji = "🟢" if signal.direction == "BUY" else "🔴"
    lines = [
        f"{emoji} *XAUUSD {signal.direction} SIGNAL*",
        f"Confidence: {signal.confidence}/4",
        f"Entry: `{signal.entry_price}`",
        f"Stop Loss: `{signal.stop_loss}`",
        f"Take Profit: `{signal.take_profit}`",
        f"Trend (H1): {signal.trend}",
        f"RSI: {signal.rsi_value}",
        "",
        "*Reasons:*",
    ] + [f"- {r}" for r in signal.reasons] + [f"\n_{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"]
    return "\n".join(lines)


# =====================================================================
# STATE (so cooldown logic persists across scheduled runs)
# =====================================================================
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_direction": None, "last_sent_time": None}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


# =====================================================================
# MAIN
# =====================================================================
def main():
    state = load_state()
    last_direction = state.get("last_direction")
    last_sent_time = datetime.fromisoformat(state["last_sent_time"]) if state.get("last_sent_time") else None

    entry_df = get_candles(SYMBOL, ENTRY_TIMEFRAME, CANDLES_TO_FETCH)
    htf_df = get_candles(SYMBOL, TREND_TIMEFRAME, CANDLES_TO_FETCH)
    result = analyse(entry_df, htf_df)

    print(f"[{result.timestamp}] {result.direction} (confidence {result.confidence}/4) @ {result.entry_price}")
    print(f"  RSI: {result.rsi_value} | Trend(H1): {result.trend} | Support: {result.support} | Resistance: {result.resistance}")
    for r in result.reasons:
        print(f"  - {r}")

    now = datetime.now()
    cooldown_ok = last_sent_time is None or (now - last_sent_time).total_seconds() > SIGNAL_COOLDOWN_MINUTES * 60
    is_new_direction = result.direction != last_direction

    if result.direction in ("BUY", "SELL") and (is_new_direction or cooldown_ok):
        if send_telegram(format_message(result)):
            state["last_direction"] = result.direction
            state["last_sent_time"] = now.isoformat()
            save_state(state)
            print("Telegram alert sent.")
    else:
        print("No alert sent this cycle.")


if __name__ == "__main__":
    main()
