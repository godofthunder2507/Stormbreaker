"""
XAUUSD Signal Bot - single-file cloud version.

Runs on a schedule (GitHub Actions) instead of as an always-on desktop app.
Each run: check the market is actually open -> fetch latest gold price data
-> analyse with a multi-indicator strategy -> send a Telegram alert if a
BUY/SELL signal fires -> save state for cooldown tracking -> exit.

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
- NEW: Market-hours gate. Spot gold trades Sun 17:00 ET - Fri 17:00 ET.
  The bot now hard-checks the actual wall-clock day/time in New York
  (DST-aware, via zoneinfo - not a guessed fixed UTC window) before doing
  ANY work, and exits immediately if the market is closed. This is on top
  of trimming the GitHub Actions cron itself to skip Saturday. Belt and
  suspenders: cron handles the bulk of Saturday for free (saves Action
  minutes), this in-code check is the actual authoritative gate and
  correctly handles the Friday-evening-close / Sunday-evening-open edges
  that a cron day-of-week field can't express on its own.
- NEW: Closed-candle trim + staleness check. The most recent candle
  returned by the API is dropped if it isn't finished yet (its close time
  hasn't passed), so indicators never fire off a still-forming bar. After
  that trim, if the latest CLOSED candle is still implausibly old (well
  beyond normal scheduling jitter), the cycle is skipped entirely - this
  catches market holidays that a plain weekday check can't (Christmas,
  New Year, etc.), where the feed just repeats a stale last close.
- NEW: ATR-scaled SL/TP. Stop-loss and take-profit are no longer a fixed
  $10/$15 - they're computed from the CURRENT ATR reading at signal time
  (1.5x ATR stop / 2.5x ATR target), then widened further if needed so
  the stop sits beyond the bot's own calculated support/resistance level
  rather than arbitrarily close to it. Both figures come from live market
  data every run, not a static guess.
- NEW: H1 trend is now also a hard filter, not just one of four optional
  votes. A BUY is blocked outright if H1 trend is DOWN, and a SELL is
  blocked outright if H1 trend is UP - stops the bot calling reversals
  straight into a strong opposing trend. Trend agreement still counts as
  a confirmation vote as before; this adds a veto on top when it actively
  disagrees, it doesn't replace the vote.
- NEW: minimum reward:risk gate. Found during local validation testing -
  when the ATR-based stop gets widened past support/resistance (see
  above), the realised reward:risk on that specific signal can come out
  worse than intended. Rather than firing anyway because the confirmation
  count looks fine, the signal is suppressed if reward:risk would be
  below 1:1 once the real (possibly-widened) stop is accounted for.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
ENTRY_TIMEFRAME_MINUTES = 15
TREND_TIMEFRAME = "1h"
TREND_TIMEFRAME_MINUTES = 60
CANDLES_TO_FETCH = 200

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_TREND_FAST, EMA_TREND_SLOW = 50, 200
ATR_PERIOD = 14
SUPPORT_RESISTANCE_LOOKBACK = 50
SR_PROXIMITY = 5.0  # widened from 3.0 so "near" S/R triggers more easily

MIN_CONFIRMATIONS = 2  # lowered from 3, out of 4 - fires more often
SIGNAL_COOLDOWN_MINUTES = 10  # lowered from 30 - shorter than the 15min check cycle

# ---- Volatility filter ----
# Compares the current ATR to its own recent average. If the market is
# unusually quiet (current ATR < ratio * baseline ATR), a would-be BUY/SELL
# is downgraded to HOLD instead of being sent. This is intentionally a
# *floor* (filters dead/choppy quiet periods) not a spike requirement -
# it does not block normal-volatility moves.
ATR_BASELINE_LOOKBACK = 50
MIN_VOLATILITY_RATIO = 0.7

# ---- Market hours gate ----
# Spot gold (XAUUSD) trades continuously from Sunday 17:00 ET to Friday
# 17:00 ET. Using zoneinfo (not a fixed UTC offset) so this stays correct
# across the US DST transitions automatically - a guessed fixed UTC
# window would silently be an hour wrong for half the year.
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE_WEEKDAY = 4  # Monday=0 ... Friday=4
MARKET_CLOSE_HOUR = 17
MARKET_OPEN_WEEKDAY = 6  # Sunday
MARKET_OPEN_HOUR = 17

# ---- Closed-candle / staleness handling ----
# After trimming to the last fully-closed candle, if that candle is still
# older than this, treat it as "no fresh data" and skip the cycle rather
# than analysing a stale repeat of the last real close. Deliberately
# generous (not set to ~20-30min) because GitHub Actions' own scheduler
# does not reliably honour a */15 cron under load - observed gaps between
# runs of an hour or more are normal scheduling jitter, not a market
# closure, and this check must not mistake one for the other.
MAX_CANDLE_AGE_MINUTES = 150

# ---- Fixed dollar profit/stop targets (legacy - kept only for the
# Telegram $ display conversion below, no longer used to size the stop) ----
POSITION_SIZE_LOTS = 0.01
OZ_PER_LOT = 100
position_oz = POSITION_SIZE_LOTS * OZ_PER_LOT

# ---- ATR-based SL/TP ----
# Stop/target are computed per-run from the CURRENT ATR reading, not a
# static guess. 1.5x ATR for the stop keeps it outside normal single-candle
# noise (the bot's own logged ATR readings during active hours run
# roughly $3-10 per 15min candle - a flat $10 stop, as this bot used to
# have, sits inside that noise band). 2.5x ATR target keeps a ~1:1.67
# reward:risk, similar to the previous 1:1.5 but now scaling with real
# conditions instead of being fixed. These are standard, widely-used
# ATR-stop multiples (not tuned to this bot's own trade history yet,
# since trades.csv only has 1 logged trade so far - not enough sample to
# fit bot-specific multiples). Keep logging real trades in trades.csv;
# analyze_thresholds.py can be extended later to tune these multiples
# once there's enough evidence to do so responsibly.
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5
# If ATR-based stop would sit tighter than the bot's own calculated
# support/resistance level, push it just beyond that level instead - the
# stop should never be closer than known structure. Buffer is itself
# ATR-scaled rather than a flat guessed number.
SR_STOP_BUFFER_ATR = 0.25
# Found during local validation testing: widening the stop past support/
# resistance can turn a healthy ~1:1.67 planned reward:risk into a much
# worse one on that particular signal. Rather than firing a trade with a
# degraded ratio just because the direction/confirmations look right,
# suppress it - a trade isn't worth taking on confirmations alone if the
# risk you'd actually be taking no longer justifies the reward.
MIN_REWARD_RISK_RATIO = 1.0

# ---- Near-miss logging ----
# Anything that didn't fire but came close (or fired and got volatility-
# or trend-suppressed) gets written here so you can see what you're NOT
# being alerted on, instead of it disappearing into a silent HOLD.
NEAR_MISS_LOG = Path(__file__).parent / "near_misses.log"
NEAR_MISS_LOG_MAX_LINES = 300

STATE_FILE = Path(__file__).parent / "state.json"

# =====================================================================
# MARKET HOURS
# =====================================================================
def market_is_open(now_utc):
    """True if spot gold is trading right now (Sun 17:00 ET - Fri 17:00 ET).

    DST-aware via zoneinfo - deliberately not a fixed UTC offset, since a
    fixed offset would be wrong for roughly half the year across the US
    DST transitions.
    """
    now_et = now_utc.astimezone(MARKET_TZ)
    weekday = now_et.weekday()  # Monday=0 ... Sunday=6

    if weekday == 5:  # Saturday - always closed
        return False
    if weekday == 6 and now_et.hour < MARKET_OPEN_HOUR:  # Sunday before open
        return False
    if weekday == MARKET_CLOSE_WEEKDAY and now_et.hour >= MARKET_CLOSE_HOUR:  # Friday after close
        return False
    return True


def trim_to_closed_candles(df, timeframe_minutes, now_utc):
    """Drop the last row if it's still forming (its close time is in the future).

    Guarantees indicators are always computed off a fully-closed candle,
    regardless of whether the data provider includes an in-progress bar.
    """
    if df.empty:
        return df
    last_open = df["time"].iloc[-1]
    if last_open.tzinfo is None:
        last_open = last_open.tz_localize("UTC")
    close_time = last_open + timedelta(minutes=timeframe_minutes)
    if close_time > now_utc:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def latest_candle_age_minutes(df, now_utc):
    if df.empty:
        return float("inf")
    last_time = df["time"].iloc[-1]
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")
    return (now_utc - last_time).total_seconds() / 60.0

# =====================================================================
# MARKET DATA (Twelve Data free API)
# =====================================================================
def get_candles(symbol, interval, count):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": count,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC",  # explicit, so staleness/candle-close math is unambiguous
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
    suppressed_by_trend: bool = False
    suppressed_by_poor_rr: bool = False
    sl_tp_note: str = ""

def determine_trend(htf_df):
    close = htf_df["close"]
    fast, slow = ema(close, EMA_TREND_FAST).iloc[-1], ema(close, EMA_TREND_SLOW).iloc[-1]
    return "UP" if fast > slow else "DOWN" if fast < slow else "RANGE"

def compute_sl_tp(direction, last_price, atr_now, support, resistance):
    """ATR-scaled SL/TP, widened past the bot's own S/R level if needed.

    Both distances are derived from live data every run (current ATR,
    current calculated support/resistance) rather than a fixed number.
    """
    note = ""
    if direction == "BUY":
        sl_from_atr = last_price - ATR_SL_MULTIPLIER * atr_now
        tp = last_price + ATR_TP_MULTIPLIER * atr_now
        sl = sl_from_atr
        if pd.notna(support) and support < last_price:
            sl_beyond_support = support - SR_STOP_BUFFER_ATR * atr_now
            if sl_beyond_support < sl:
                sl = sl_beyond_support
                note = "SL widened beyond support"
    elif direction == "SELL":
        sl_from_atr = last_price + ATR_SL_MULTIPLIER * atr_now
        tp = last_price - ATR_TP_MULTIPLIER * atr_now
        sl = sl_from_atr
        if pd.notna(resistance) and resistance > last_price:
            sl_beyond_resistance = resistance + SR_STOP_BUFFER_ATR * atr_now
            if sl_beyond_resistance > sl:
                sl = sl_beyond_resistance
                note = "SL widened beyond resistance"
    else:
        sl = tp = 0.0
    return sl, tp, note

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

    # Hard trend filter: block a signal that calls a reversal straight into
    # a strong opposing H1 trend, on top of trend counting as a vote above.
    suppressed_by_trend = False
    if raw_direction == "BUY" and trend == "DOWN":
        suppressed_by_trend = True
    elif raw_direction == "SELL" and trend == "UP":
        suppressed_by_trend = True

    if suppressed_by_trend:
        direction, reasons, confidence = "HOLD", [], 0
        suppressed = False
    else:
        # Apply volatility filter: a real BUY/SELL signal gets downgraded to
        # HOLD if the market is unusually quiet right now.
        suppressed = False
        if raw_direction in ("BUY", "SELL") and not volatility_ok:
            direction, reasons, confidence = "HOLD", [], 0
            suppressed = True
        else:
            direction, reasons, confidence = raw_direction, raw_reasons, raw_confidence

    sl_tp_note = ""
    suppressed_by_poor_rr = False
    if direction in ("BUY", "SELL") and pd.notna(atr_now) and atr_now > 0:
        stop_loss, take_profit, sl_tp_note = compute_sl_tp(direction, last_price, atr_now, support, resistance)
        risk = abs(last_price - stop_loss)
        reward = abs(take_profit - last_price)
        if risk <= 0 or (reward / risk) < MIN_REWARD_RISK_RATIO:
            # Structural stop (beyond S/R) made this trade's real reward:risk
            # worse than acceptable - don't fire on confirmations alone.
            suppressed_by_poor_rr = True
            direction, reasons, confidence = "HOLD", [], 0
            stop_loss = take_profit = 0.0
    elif direction in ("BUY", "SELL"):
        # No usable ATR reading - can't size a fact-based stop, so don't fire.
        direction, reasons, confidence = "HOLD", [], 0
        stop_loss = take_profit = 0.0
        sl_tp_note = "suppressed - no usable ATR reading to size SL/TP"
    else:
        stop_loss = take_profit = 0.0

    return SignalResult(
        timestamp=datetime.now(timezone.utc), direction=direction, confidence=confidence,
        entry_price=round(last_price, 2), stop_loss=round(stop_loss, 2), take_profit=round(take_profit, 2),
        rsi_value=round(last_rsi, 2), trend=trend, support=round(support, 2), resistance=round(resistance, 2),
        reasons=reasons,
        raw_direction=raw_direction, raw_confidence=raw_confidence, raw_reasons=raw_reasons,
        best_direction=best_direction, best_confidence=best_confidence, best_reasons=best_reasons,
        atr_value=round(float(atr_now), 3) if pd.notna(atr_now) else 0.0,
        atr_baseline=round(float(atr_baseline), 3) if pd.notna(atr_baseline) else 0.0,
        volatility_ok=volatility_ok, suppressed_by_volatility=suppressed,
        suppressed_by_trend=suppressed_by_trend, suppressed_by_poor_rr=suppressed_by_poor_rr,
        sl_tp_note=sl_tp_note,
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
    sl_distance = abs(signal.entry_price - signal.stop_loss)
    tp_distance = abs(signal.take_profit - signal.entry_price)
    est_risk_usd = sl_distance * position_oz
    est_profit_usd = tp_distance * position_oz
    lines = [
        f"{emoji} *XAUUSD {signal.direction} SIGNAL*",
        f"Confidence: {signal.confidence}/4",
        f"Entry: `{signal.entry_price}`",
        f"Stop Loss: `{signal.stop_loss}`",
        f"Take Profit: `{signal.take_profit}`",
        f"(ATR-based: {ATR_SL_MULTIPLIER}x/{ATR_TP_MULTIPLIER}x ATR({signal.atr_value})"
        + (f", {signal.sl_tp_note}" if signal.sl_tp_note else "")
        + f" | ~${est_risk_usd:.2f} risk / ${est_profit_usd:.2f} target at {POSITION_SIZE_LOTS} lot)",
        f"Trend (H1): {signal.trend}",
        f"RSI: {signal.rsi_value}",
        "",
        "*Reasons:*",
    ] + [f"- {r}" for r in signal.reasons] + [f"\n_{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC_"]
    return "\n".join(lines)

# =====================================================================
# NEAR-MISS LOGGING
# =====================================================================
def log_near_miss(result):
    """Append a line if this cycle was a near-miss or a suppressed signal
    (by volatility or trend), so silent HOLDs don't disappear without a
    trace. Keeps the log trimmed to the last NEAR_MISS_LOG_MAX_LINES lines.

    Deliberately does NOT get called at all when the market is closed or
    the candle is stale (see main()) - that would just recreate the same
    noise problem in a different file.
    """
    line = None
    ts = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if result.suppressed_by_trend:
        line = (
            f"[{ts}] BLOCKED (trend filter) - would have been {result.raw_direction} "
            f"{result.raw_confidence}/4 @ {result.entry_price} | H1 trend {result.trend} "
            f"actively opposes | reasons: {', '.join(result.raw_reasons)}"
        )
    elif result.suppressed_by_poor_rr:
        line = (
            f"[{ts}] SUPPRESSED (poor R:R after SR widening) - would have been "
            f"{result.raw_direction} {result.raw_confidence}/4 @ {result.entry_price} | "
            f"ATR {result.atr_value} | reasons: {', '.join(result.raw_reasons)}"
        )
    elif result.suppressed_by_volatility:
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
    now = datetime.now(timezone.utc)

    # Market-hours gate FIRST, before any API calls - cheapest possible
    # short-circuit, and the authoritative fix for the weekend false-fire
    # bug (the GitHub Actions cron is trimmed to skip Saturday too, but
    # this in-code check is what actually enforces it precisely).
    if not market_is_open(now):
        print(f"[{now}] Market closed (spot gold trades Sun 17:00 ET - Fri 17:00 ET). Skipping cycle.")
        return

    entry_df = get_candles(SYMBOL, ENTRY_TIMEFRAME, CANDLES_TO_FETCH)
    htf_df = get_candles(SYMBOL, TREND_TIMEFRAME, CANDLES_TO_FETCH)

    entry_df = trim_to_closed_candles(entry_df, ENTRY_TIMEFRAME_MINUTES, now)
    htf_df = trim_to_closed_candles(htf_df, TREND_TIMEFRAME_MINUTES, now)

    age_minutes = latest_candle_age_minutes(entry_df, now)
    if age_minutes > MAX_CANDLE_AGE_MINUTES:
        print(
            f"[{now}] Latest closed candle is {age_minutes:.0f} min old "
            f"(> {MAX_CANDLE_AGE_MINUTES} min threshold) - likely a holiday/feed gap, "
            f"not a normal scheduling delay. Skipping cycle rather than analysing stale data."
        )
        return

    state = load_state()
    last_direction = state.get("last_direction")
    last_sent_time = datetime.fromisoformat(state["last_sent_time"]) if state.get("last_sent_time") else None

    result = analyse(entry_df, htf_df)

    print(f"[{result.timestamp}] {result.direction} (confidence {result.confidence}/4) @ {result.entry_price}")
    print(f"  RSI: {result.rsi_value} | Trend(H1): {result.trend} | Support: {result.support} | Resistance: {result.resistance}")
    print(f"  ATR: {result.atr_value} | ATR baseline: {result.atr_baseline} | Volatility OK: {result.volatility_ok}")
    for r in result.reasons:
        print(f"  - {r}")

    log_near_miss(result)

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
    else:
        print("No alert sent this cycle.")
        save_state(state)

if __name__ == "__main__":
    main()
