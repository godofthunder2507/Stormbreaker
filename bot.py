"""
XAUUSD Signal Bot - single-file cloud version.

Runs on a schedule (GitHub Actions) instead of as an always-on desktop app.
Each run: fetch latest gold price data -> analyse with a multi-indicator
strategy -> send a Telegram alert if a BUY/SELL signal fires -> save state
for cooldown tracking -> exit.

Secrets (Telegram token/chat ID, Twelve Data API key) are read from
environment variables, which GitHub Actions injects from encrypted
repository secrets - nothing sensitive lives in this file.

CHANGELOG (this version):
- Fixed RSI dead-zone: BUY required RSI>40 and SELL required RSI<60, so
  anywhere RSI sat between 40-60 BOTH sides scored a point and cancelled
  each other out. Now split cleanly at 50.
- Added an ATR-based volatility filter: a BUY/SELL that would otherwise
  fire is suppressed if the market is unusually quiet (current ATR well
  below its recent average), since signals in dead markets are lower
  quality and more likely to chop/stop out.
- Added near-miss logging: every HOLD that was one confirmation short of
  firing, or that fired but got suppressed by the volatility filter, gets
  appended to near_misses.log so you have an audit trail of what the bot
  almost (or technically did) call instead of total silence.
"""

import json
import os
from dataclasses import dataclass, field
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
SR_PROXIMITY = 5.0   # widened from 3.0 so "near" S/R triggers more easily

MIN_CONFIRMATIONS = 2          # lowered from 3, out of 4 - fires more often
SIGNAL_COOLDOWN_MINUTES = 10   # lowered from 30 - shorter than the 15min check cycle

# ---- Volatility filter ----
# Compares the current ATR to its own recent average. If the market is
# unusually quiet (current ATR < ratio * baseline ATR), a would-be BUY/SELL
# is downgraded to HOLD instead of being sent. This is intentionally a
# *floor* (filters dead/choppy quiet periods) not a spike requirement -
# it does not block normal-volatility moves.
ATR_BASELINE_LOOKBACK = 50
MIN_VOLATILITY_RATIO = 0.7

# ---- Near-miss logging ----
# Anything that didn't fire but came close (or fired and got volatility-
# suppressed) gets written here so you can see what you're NOT being
# alerted on, instead of it disappearing into a silent HOLD.
NEAR_MISS_LOG = Path(__file__).parent / "near_misses.log"
NEAR_MISS_LOG_MAX_LINES = 300

# ---- Fixed dollar profit/stop targets (replaces the old ATR-based sizing) ----
# Calculated from your position size: 1 standard lot = 100 oz, so the price
# move needed for a given $ profit/loss = target_dollars / (lots * 100).
# At 0.01 lot (1 oz), $1 of price movement = $1 of profit/loss, so the
# distances below are just the dollar targets directly.
POSITION_SIZE_LOTS = 0.01
OZ_PER_LOT = 100
TARGET_PROFIT_USD = 15
TARGET_STOP_USD = 10           # ~1:1.5 reward:risk - edit to taste

position_oz = POSITION_SIZE_LOTS * OZ_PER_LOT
TP_DISTANCE = TARGET_PROFIT_USD / position_oz
SL_DISTANCE = TARGET_STOP_USD / position_oz

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


def compute_volatility_ok(atr_series):
    """Compare current ATR to its recent baseline average.

    Returns (atr_now, atr_baseline, volatility_ok). If there isn't enough
    history to compute a meaningful baseline, defaults to volatility_ok=True
    so the filter never blocks signals purely due to a cold start.
    """
    atr_now = atr_series.iloc[-1]
    baseline_window = atr_series.iloc[-(ATR_BASELINE_LOOKBACK + 1):-1].dropna()
    if baseline_window.empty:
        return atr_now, float("nan"), True
    atr_baseline = baseline_window.mean()
    if not atr_baseline or np.isnan(atr_baseline) or atr_baseline <= 0:
        return atr_now, atr_baseline, True
    return atr_now, atr_baseline, bool(atr_now >= MIN_VOLATILITY_RATIO * atr_baseline)


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
    # --- diagnostics for near-miss logging / debugging ---
    raw_direction: str = "HOLD"
    raw_confidence: int = 0
    raw_reasons: list = field(default_factory=list)
    # best candidate regardless of whether it cleared MIN_CONFIRMATIONS -
    # used purely for near-miss diagnostics, never for sending alerts
    best_direction: str = "HOLD"
    best_confidence: int = 0
    best_reasons: list = field(default_factory=list)
    atr_value: float = 0.0
    atr_baseline: float = 0.0
    volatility_ok: bool = True
    suppressed_by_volatility: bool = False


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

    atr_series = atr(entry_df, ATR_PERIOD)
    atr_now, atr_baseline, volatility_ok = compute_volatility_ok(atr_series)

    buy_reasons, sell_reasons = [], []
    if bullish_cross:
        buy_reasons.append("MACD bullish crossover")
    # RSI split cleanly at 50 - previously BUY needed >40 and SELL needed <60,
    # so 40-60 scored BOTH sides and the RSI point cancelled itself out.
    if RSI_OVERSOLD < last_rsi < RSI_OVERBOUGHT and last_rsi >= 50:
        buy_reasons.append(f"RSI healthy/bullish at {last_rsi:.1f}")
    if near_support:
        buy_reasons.append(f"Price near support ({support:.2f})")
    if trend == "UP":
        buy_reasons.append("H1 trend is UP")

    if bearish_cross:
        sell_reasons.append("MACD bearish crossover")
    if RSI_OVERSOLD < last_rsi < RSI_OVERBOUGHT and last_rsi < 50:
        sell_reasons.append(f"RSI healthy/bearish at {last_rsi:.1f}")
    if near_resistance:
        sell_reasons.append(f"Price near resistance ({resistance:.2f})")
    if trend == "DOWN":
        sell_reasons.append("H1 trend is DOWN")

    confirmations_buy, confirmations_sell = len(buy_reasons), len(sell_reasons)

    raw_direction, raw_reasons, raw_confidence = "HOLD", [], 0
    if confirmations_buy >= MIN_CONFIRMATIONS and confirmations_buy >= confirmations_sell:
        raw_direction, raw_reasons, raw_confidence = "BUY", buy_reasons, confirmations_buy
    elif confirmations_sell >= MIN_CONFIRMATIONS:
        raw_direction, raw_reasons, raw_confidence = "SELL", sell_reasons, confirmations_sell

    # Best candidate regardless of whether it cleared the threshold - this is
    # what makes near-miss logging actually work, since raw_confidence above
    # is 0 whenever neither side reached MIN_CONFIRMATIONS.
    if confirmations_buy >= confirmations_sell:
        best_direction, best_confidence, best_reasons = "BUY", confirmations_buy, buy_reasons
    else:
        best_direction, best_confidence, best_reasons = "SELL", confirmations_sell, sell_reasons

    # Apply volatility filter: a real BUY/SELL signal gets downgraded to HOLD
    # if the market is unusually quiet right now.
    suppressed = False
    if raw_direction in ("BUY", "SELL") and not volatility_ok:
        direction, reasons, confidence = "HOLD", [], 0
        suppressed = True
    else:
        direction, reasons, confidence = raw_direction, raw_reasons, raw_confidence

    if direction == "BUY":
        stop_loss = last_price - SL_DISTANCE
        take_profit = last_price + TP_DISTANCE
    elif direction == "SELL":
        stop_loss = last_price + SL_DISTANCE
        take_profit = last_price - TP_DISTANCE
    else:
        stop_loss = take_profit = 0.0

    return SignalResult(
        timestamp=datetime.now(), direction=direction, confidence=confidence,
        entry_price=round(last_price, 2), stop_loss=round(stop_loss, 2), take_profit=round(take_profit, 2),
        rsi_value=round(last_rsi, 2), trend=trend, support=round(support, 2), resistance=round(resistance, 2),
        reasons=reasons,
        raw_direction=raw_direction, raw_confidence=raw_confidence, raw_reasons=raw_reasons,
        best_direction=best_direction, best_confidence=best_confidence, best_reasons=best_reasons,
        atr_value=round(float(atr_now), 3) if pd.notna(atr_now) else 0.0,
        atr_baseline=round(float(atr_baseline), 3) if pd.notna(atr_baseline) else 0.0,
        volatility_ok=volatility_ok, suppressed_by_volatility=suppressed,
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
        f"(targets ~${TARGET_PROFIT_USD} profit / ${TARGET_STOP_USD} risk at {POSITION_SIZE_LOTS} lot)",
        f"Trend (H1): {signal.trend}",
        f"RSI: {signal.rsi_value}",
        "",
        "*Reasons:*",
    ] + [f"- {r}" for r in signal.reasons] + [f"\n_{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"]
    return "\n".join(lines)


# =====================================================================
# NEAR-MISS LOGGING
# =====================================================================
def log_near_miss(result):
    """Append a line if this cycle was a near-miss or a volatility-suppressed
    signal, so silent HOLDs don't disappear without a trace. Keeps the log
    trimmed to the last NEAR_MISS_LOG_MAX_LINES lines."""
    line = None
    ts = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if result.suppressed_by_volatility:
        line = (
            f"[{ts}] SUPPRESSED (volatility) - would have been {result.raw_direction} "
            f"{result.raw_confidence}/4 @ {result.entry_price} | ATR {result.atr_value} "
            f"vs baseline {result.atr_baseline} (ratio {MIN_VOLATILITY_RATIO}) | "
            f"reasons: {', '.join(result.raw_reasons)}"
        )
    elif result.direction == "HOLD" and result.best_confidence == MIN_CONFIRMATIONS - 1:
        line = (
            f"[{ts}] NEAR MISS - would be {result.best_direction} {result.best_confidence}/4, "
            f"{MIN_CONFIRMATIONS - result.best_confidence} short @ {result.entry_price} | "
            f"RSI {result.rsi_value} | Trend(H1) {result.trend} | "
            f"reasons: {', '.join(result.best_reasons) if result.best_reasons else 'none'}"
        )

    if line is None:
        return

    print(line)
    existing = NEAR_MISS_LOG.read_text().splitlines() if NEAR_MISS_LOG.exists() else []
    existing.append(line)
    trimmed = existing[-NEAR_MISS_LOG_MAX_LINES:]
    NEAR_MISS_LOG.write_text("\n".join(trimmed) + "\n")


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
    print(f"  ATR: {result.atr_value} | ATR baseline: {result.atr_baseline} | Volatility OK: {result.volatility_ok}")
    for r in result.reasons:
        print(f"  - {r}")

    log_near_miss(result)

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
        save_state(state)


if __name__ == "__main__":
    main()
