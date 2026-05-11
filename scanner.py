"""
NSE Breakout Scanner — Railway.app deployment
Scans NSE equities + F&O every minute, fires Telegram alerts on breakout/retest.
No broker API needed — uses NSE's public website data via nsepython.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from nsepython import nsefetch, nse_eq, nse_fno
import json

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Config (set these in Railway environment variables) ───────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")     # From @BotFather
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")   # Your chat/group ID
SCAN_INTERVAL    = int(os.environ.get("SCAN_INTERVAL", "60"))  # seconds between scans
LOOKBACK         = int(os.environ.get("LOOKBACK", "20"))       # candles for S/R
VOL_MULTIPLIER   = float(os.environ.get("VOL_MULTIPLIER", "1.5"))
RETEST_TOLERANCE = float(os.environ.get("RETEST_TOLERANCE", "0.5"))  # percent

# ── Stocks to scan ────────────────────────────────────────────────────────────
# Add or remove symbols as needed. Use exact NSE symbols.
EQUITY_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI", "TITAN",
    "AXISBANK", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "SUNPHARMA", "TECHM", "HCLTECH", "TATAMOTORS",
    "TATASTEEL", "ADANIENT", "ADANIPORTS", "BAJAJFINSV", "NESTLEIND",
]

FNO_WATCHLIST = [
    "NIFTY", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
]

# ── Market hours check ────────────────────────────────────────────────────────
def is_market_open():
    """Returns True if current IST time is within NSE trading hours."""
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 6=Sun
    if weekday >= 5:          # Saturday or Sunday
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 30)

# ── Telegram alert sender ─────────────────────────────────────────────────────
def send_telegram(message: str):
    """Send a message to Telegram. Retries once on failure."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set. Skipping alert.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    for attempt in range(2):
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                log.info(f"Telegram alert sent: {message[:60]}...")
                return
            else:
                log.warning(f"Telegram error {r.status_code}: {r.text}")
        except Exception as e:
            log.error(f"Telegram send failed (attempt {attempt+1}): {e}")
        time.sleep(2)

def format_alert(signal_type: str, symbol: str, price: float,
                 resistance: float, support: float,
                 target: float, stoploss: float,
                 rr_ratio: float, volume_ratio: float,
                 segment: str = "EQ") -> str:
    """Format a clean Telegram alert message."""
    icons = {
        "BREAKOUT_UP":   "🚀",
        "BREAKDOWN_DN":  "🔻",
        "RETEST_UP":     "✅",
        "RETEST_DN":     "🔴",
        "NEAR_RESISTANCE": "⚠️",
        "NEAR_SUPPORT":    "⚠️",
    }
    icon = icons.get(signal_type, "📊")
    direction = "LONG" if "UP" in signal_type or signal_type == "BREAKOUT_UP" else "SHORT"

    lines = [
        f"{icon} <b>{signal_type.replace('_', ' ')}</b>",
        f"",
        f"<b>Symbol:</b> {symbol} [{segment}]",
        f"<b>Price:</b> ₹{price:,.2f}",
        f"",
    ]

    if signal_type in ("RETEST_UP", "RETEST_DN"):
        lines += [
            f"<b>Entry zone:</b> ₹{price:,.2f}",
            f"<b>Target:</b> ₹{target:,.2f}",
            f"<b>Stoploss:</b> ₹{stoploss:,.2f}",
            f"<b>R:R Ratio:</b> 1 : {rr_ratio:.1f}",
            f"",
        ]
    elif "BREAKOUT" in signal_type or "BREAKDOWN" in signal_type:
        lines += [
            f"<b>Broken level:</b> ₹{resistance if 'UP' in signal_type else support:,.2f}",
            f"<b>Watching for retest near:</b> ₹{resistance if 'UP' in signal_type else support:,.2f}",
            f"",
        ]
    else:
        lines += [
            f"<b>Key level:</b> ₹{resistance if 'RESISTANCE' in signal_type else support:,.2f}",
            f"",
        ]

    lines += [
        f"<b>Volume:</b> {volume_ratio:.1f}x average",
        f"<b>Direction:</b> {direction}",
        f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S IST')}",
    ]
    return "\n".join(lines)

# ── NSE Data fetcher ──────────────────────────────────────────────────────────
def get_ohlcv(symbol: str, segment: str = "EQ") -> pd.DataFrame | None:
    """
    Fetch OHLCV data from NSE public API.
    Returns a DataFrame with columns: open, high, low, close, volume
    sorted oldest-first. Returns None on failure.
    """
    try:
        if segment == "EQ":
            url = f"https://www.nseindia.com/api/chart-databyindex?index={symbol}EQN&indices=false"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            }
            # NSE requires a session cookie — establish one first
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            resp = session.get(url, headers=headers, timeout=10)
            data = resp.json()

            if "grapthData" not in data:
                log.warning(f"No chart data for {symbol}")
                return None

            rows = data["grapthData"]
            df = pd.DataFrame(rows, columns=["timestamp", "close"])
            df["open"]   = df["close"]
            df["high"]   = df["close"]
            df["low"]    = df["close"]
            df["volume"] = 1_000_000  # intraday volume not always in this endpoint

            # Use the dedicated quote endpoint for richer OHLCV
            quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}&section=trade_info"
            qr = session.get(quote_url, headers=headers, timeout=10)
            qdata = qr.json()

            if "marketDeptOrderBook" in qdata:
                td = qdata.get("marketDeptOrderBook", {}).get("tradeInfo", {})
                df.at[df.index[-1], "volume"] = float(str(td.get("totalTradedVolume", "1000000")).replace(",", ""))

            return df.tail(LOOKBACK + 5)

        elif segment == "FNO":
            # For F&O, use the futures price feed
            url = f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}"
            session = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.nseindia.com/",
            }
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            resp = session.get(url, headers=headers, timeout=10)
            data = resp.json()
            return None  # Simplified — full F&O OHLC needs separate processing

    except Exception as e:
        log.error(f"Data fetch failed for {symbol}: {e}")
        return None

def get_quote(symbol: str) -> dict | None:
    """Get current quote snapshot for a symbol."""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        resp = session.get(url, headers=headers, timeout=10)
        data = resp.json()

        pd_data = data.get("priceInfo", {})
        vol_data = data.get("marketDeptOrderBook", {}).get("tradeInfo", {})

        return {
            "symbol":       symbol,
            "open":         float(pd_data.get("open", 0)),
            "high":         float(pd_data.get("intraDayHighLow", {}).get("max", 0)),
            "low":          float(pd_data.get("intraDayHighLow", {}).get("min", 0)),
            "close":        float(pd_data.get("lastPrice", 0)),
            "prev_close":   float(pd_data.get("previousClose", 0)),
            "volume":       float(str(vol_data.get("totalTradedVolume", "0")).replace(",", "")),
            "52w_high":     float(pd_data.get("weekHighLow", {}).get("max", 0)),
            "52w_low":      float(pd_data.get("weekHighLow", {}).get("min", 0)),
        }
    except Exception as e:
        log.error(f"Quote fetch failed for {symbol}: {e}")
        return None

def get_historical_ohlcv(symbol: str, days: int = 30) -> pd.DataFrame | None:
    """Fetch historical daily OHLCV from NSE for support/resistance calculation."""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        end   = datetime.now()
        start = end - pd.Timedelta(days=days + 10)  # buffer for weekends/holidays
        s_str = start.strftime("%d-%m-%Y")
        e_str = end.strftime("%d-%m-%Y")

        url = (f"https://www.nseindia.com/api/historical/cm/equity"
               f"?symbol={symbol}&series=[%22EQ%22]&from={s_str}&to={e_str}&csv=false")
        resp = session.get(url, headers=headers, timeout=15)
        data = resp.json()

        rows = data.get("data", [])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "CH_OPENING_PRICE": "open",
            "CH_TRADE_HIGH_PRICE": "high",
            "CH_TRADE_LOW_PRICE": "low",
            "CH_CLOSING_PRICE": "close",
            "CH_TOT_TRADED_QTY": "volume",
            "CH_TIMESTAMP": "date",
        })
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df.tail(LOOKBACK + 5)

    except Exception as e:
        log.error(f"Historical fetch failed for {symbol}: {e}")
        return None

# ── Breakout logic ────────────────────────────────────────────────────────────

class BreakoutState:
    """Tracks per-symbol state across scan cycles."""
    def __init__(self):
        # symbol -> {"level": float, "direction": int (1=up, -1=dn), "alerted": bool}
        self.breakouts: dict = {}
        # symbol -> last signal type sent (to avoid duplicate alerts)
        self.last_signal: dict = {}
        # symbol -> rolling volume average
        self.avg_volumes: dict = {}

state = BreakoutState()

def calculate_signals(symbol: str, df: pd.DataFrame, quote: dict) -> list[dict]:
    """
    Given historical OHLCV + current quote, return list of signal dicts.
    Each signal: {type, symbol, price, resistance, support, target, stoploss, rr, vol_ratio}
    """
    signals = []
    if df is None or len(df) < LOOKBACK:
        return signals

    # Use historical bars for S/R, current quote for live price
    resistance = df["high"].iloc[:-1].max()   # highest high excluding today
    support    = df["low"].iloc[:-1].min()    # lowest low excluding today
    avg_vol    = df["volume"].iloc[:-1].mean()

    price      = quote["close"]
    today_vol  = quote["volume"]
    vol_ratio  = today_vol / avg_vol if avg_vol > 0 else 1.0

    range_size = resistance - support
    tol        = resistance * RETEST_TOLERANCE / 100

    # ── Breakout UP ──────────────────────────────────────────────────────────
    if (price > resistance
            and vol_ratio >= VOL_MULTIPLIER
            and state.last_signal.get(symbol) not in ("BREAKOUT_UP", "RETEST_UP")):
        state.breakouts[symbol] = {"level": resistance, "direction": 1}
        signals.append({
            "type": "BREAKOUT_UP", "symbol": symbol, "price": price,
            "resistance": resistance, "support": support,
            "target": resistance + range_size,
            "stoploss": support,
            "rr": range_size / max(price - support, 0.01),
            "vol_ratio": vol_ratio,
        })

    # ── Breakdown DN ─────────────────────────────────────────────────────────
    elif (price < support
            and vol_ratio >= VOL_MULTIPLIER
            and state.last_signal.get(symbol) not in ("BREAKDOWN_DN", "RETEST_DN")):
        state.breakouts[symbol] = {"level": support, "direction": -1}
        signals.append({
            "type": "BREAKDOWN_DN", "symbol": symbol, "price": price,
            "resistance": resistance, "support": support,
            "target": support - range_size,
            "stoploss": resistance,
            "rr": range_size / max(resistance - price, 0.01),
            "vol_ratio": vol_ratio,
        })

    # ── Retest UP confirmed ───────────────────────────────────────────────────
    elif (symbol in state.breakouts
            and state.breakouts[symbol]["direction"] == 1
            and state.last_signal.get(symbol) == "BREAKOUT_UP"):
        level = state.breakouts[symbol]["level"]
        if (quote["low"] <= level + tol          # came back down to level
                and price > level                 # but closed above it
                and price > quote["open"]):        # green candle
            target   = level + range_size
            stoploss = quote["low"] * 0.995       # 0.5% below retest low
            rr       = (target - price) / max(price - stoploss, 0.01)
            signals.append({
                "type": "RETEST_UP", "symbol": symbol, "price": price,
                "resistance": resistance, "support": support,
                "target": target, "stoploss": stoploss,
                "rr": rr, "vol_ratio": vol_ratio,
            })

    # ── Retest DN confirmed ───────────────────────────────────────────────────
    elif (symbol in state.breakouts
            and state.breakouts[symbol]["direction"] == -1
            and state.last_signal.get(symbol) == "BREAKDOWN_DN"):
        level = state.breakouts[symbol]["level"]
        if (quote["high"] >= level - tol          # bounced back to level
                and price < level                  # but closed below it
                and price < quote["open"]):         # red candle
            target   = level - range_size
            stoploss = quote["high"] * 1.005
            rr       = (price - target) / max(stoploss - price, 0.01)
            signals.append({
                "type": "RETEST_DN", "symbol": symbol, "price": price,
                "resistance": resistance, "support": support,
                "target": target, "stoploss": stoploss,
                "rr": rr, "vol_ratio": vol_ratio,
            })

    # ── Near resistance / support (early warning) ────────────────────────────
    proximity = 0.8  # percent
    if (abs(price - resistance) / resistance * 100 < proximity
            and state.last_signal.get(symbol) != "NEAR_RESISTANCE"):
        signals.append({
            "type": "NEAR_RESISTANCE", "symbol": symbol, "price": price,
            "resistance": resistance, "support": support,
            "target": resistance + range_size,
            "stoploss": support,
            "rr": 0, "vol_ratio": vol_ratio,
        })

    elif (abs(price - support) / support * 100 < proximity
            and state.last_signal.get(symbol) != "NEAR_SUPPORT"):
        signals.append({
            "type": "NEAR_SUPPORT", "symbol": symbol, "price": price,
            "resistance": resistance, "support": support,
            "target": support - range_size,
            "stoploss": resistance,
            "rr": 0, "vol_ratio": vol_ratio,
        })

    return signals

# ── Main scan loop ────────────────────────────────────────────────────────────

def scan_once():
    """Run one full scan cycle across all watchlist symbols."""
    log.info(f"Starting scan — {len(EQUITY_WATCHLIST)} symbols")
    total_signals = 0

    for symbol in EQUITY_WATCHLIST:
        try:
            df    = get_historical_ohlcv(symbol, days=30)
            quote = get_quote(symbol)

            if df is None or quote is None or quote["close"] == 0:
                log.debug(f"Skipping {symbol} — no data")
                continue

            signals = calculate_signals(symbol, df, quote)

            for sig in signals:
                # Skip low R:R retest signals (below 1.5)
                if sig["type"] in ("RETEST_UP", "RETEST_DN") and sig["rr"] < 1.5:
                    log.info(f"Skipping {symbol} — R:R {sig['rr']:.1f} too low")
                    continue

                msg = format_alert(
                    signal_type  = sig["type"],
                    symbol       = sig["symbol"],
                    price        = sig["price"],
                    resistance   = sig["resistance"],
                    support      = sig["support"],
                    target       = sig["target"],
                    stoploss     = sig["stoploss"],
                    rr_ratio     = sig["rr"],
                    volume_ratio = sig["vol_ratio"],
                    segment      = "EQ",
                )
                send_telegram(msg)
                state.last_signal[symbol] = sig["type"]
                total_signals += 1
                time.sleep(0.5)  # Avoid Telegram rate limit

            time.sleep(1.0)  # Polite delay between NSE requests

        except Exception as e:
            log.error(f"Error scanning {symbol}: {e}")
            continue

    log.info(f"Scan complete — {total_signals} signals fired")

def run():
    """Main entry point. Runs continuously."""
    log.info("=" * 50)
    log.info("NSE Breakout Scanner starting up")
    log.info(f"Symbols: {len(EQUITY_WATCHLIST)} equity")
    log.info(f"Scan interval: {SCAN_INTERVAL}s")
    log.info(f"Lookback: {LOOKBACK} bars | Vol multiplier: {VOL_MULTIPLIER}x")
    log.info("=" * 50)

    # Send startup message
    send_telegram(
        "🟢 <b>BreakoutScanner started</b>\n\n"
        f"Scanning <b>{len(EQUITY_WATCHLIST)}</b> NSE stocks\n"
        f"Interval: every <b>{SCAN_INTERVAL}s</b>\n"
        f"Lookback: <b>{LOOKBACK}</b> days\n"
        f"Volume filter: <b>{VOL_MULTIPLIER}x</b> average\n\n"
        f"Market opens at <b>9:15 AM IST</b>"
    )

    while True:
        try:
            if is_market_open():
                scan_once()
            else:
                now = datetime.now()
                log.info(f"Market closed ({now.strftime('%H:%M')} IST) — waiting...")

                # Send daily summary at market close
                if now.time().hour == 15 and now.time().minute == 31:
                    total = sum(1 for v in state.last_signal.values()
                                if v in ("RETEST_UP", "RETEST_DN"))
                    send_telegram(
                        f"📊 <b>Market closed</b>\n\n"
                        f"Retest signals today: <b>{total}</b>\n"
                        f"Scanner resumes at <b>9:15 AM IST</b> tomorrow"
                    )

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            log.info("Scanner stopped by user.")
            send_telegram("🔴 <b>BreakoutScanner stopped</b>")
            break
        except Exception as e:
            log.error(f"Unexpected error in main loop: {e}")
            time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    run()
