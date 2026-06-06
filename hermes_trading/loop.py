"""
3-Step Price Action — Supply & Demand strategy for BTC/USD 5-minute chart.

Strategy (youtu.be/e-QmGJU1XYc):
  Step 1 — Market Structure: Determine uptrend (HH + HL) or downtrend (LH + LL)
            using validated swing highs and lows.
  Step 2 — Supply & Demand Zones: In uptrend, mark demand zones (consolidation
            before bullish impulse). In downtrend, mark supply zones (consolidation
            before bearish impulse). Enter when price re-tests the zone.
            Stop = beyond the zone. Target = nearest swing high/low.
  Step 3 — R:R Filter: Skip any trade where reward/risk < min_rr (default 2.5).
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from hermes_trading.adapters.price import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

PAPER_MODE    = os.getenv("HERMES_TRADING_MODE", "paper") == "paper"
STATE_FILE    = Path("state/worker_state.json")
TRADES_FILE   = Path("state/trades.jsonl")
STRATEGY_FILE = Path("state/strategy.yaml")
GOAL_FILE     = Path("state/goal.yaml")
GRANULARITY   = 300  # 5-minute candles


def load_params() -> dict:
    try:
        raw = yaml.safe_load(STRATEGY_FILE.read_text()) or {}
        e = raw.get("entry", {})
        return {
            "swing_n":            int(e.get("swing_n",            3)),
            "impulse_threshold":float(e.get("impulse_threshold",  1.5)),
            "impulse_body_ratio":float(e.get("impulse_body_ratio",0.5)),
            "min_rr":           float(e.get("min_rr",             2.5)),
            "zone_buffer_pct":  float(e.get("zone_buffer_pct",  0.001)),
            "zone_max_age_bars":  int(e.get("zone_max_age_bars",   80)),
        }
    except Exception:
        return {
            "swing_n": 3, "impulse_threshold": 1.5, "impulse_body_ratio": 0.5,
            "min_rr": 2.5, "zone_buffer_pct": 0.001, "zone_max_age_bars": 80,
        }


def reflection_due() -> bool:
    try:
        goal = yaml.safe_load(GOAL_FILE.read_text()) or {}
    except Exception:
        goal = {}
    every = int(goal.get("reflection_every", 5))
    if not TRADES_FILE.exists():
        return False
    closed = sum(1 for line in TRADES_FILE.read_text().splitlines() if line.strip())
    return closed > 0 and closed % every == 0


def load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"current_trade": None, "used_zone_ts": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log_trade(record: dict) -> None:
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── price action helpers ───────────────────────────────────────────────────────

def _swing_highs(candles: list, n: int) -> list:
    result = []
    for i in range(n, len(candles) - n):
        c = candles[i]
        if all(c["high"] >= candles[j]["high"]
               for j in range(i - n, i + n + 1) if j != i):
            result.append({"price": c["high"], "ts": c["timestamp"], "idx": i})
    return result


def _swing_lows(candles: list, n: int) -> list:
    result = []
    for i in range(n, len(candles) - n):
        c = candles[i]
        if all(c["low"] <= candles[j]["low"]
               for j in range(i - n, i + n + 1) if j != i):
            result.append({"price": c["low"], "ts": c["timestamp"], "idx": i})
    return result


def detect_trend(candles: list, n: int = 3) -> str | None:
    """Return 'uptrend', 'downtrend', or None (ranging/unclear)."""
    highs = _swing_highs(candles, n)
    lows  = _swing_lows(candles, n)
    if len(highs) < 2 or len(lows) < 2:
        return None
    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"]  > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"]  < lows[-2]["price"]
    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    return None


def _atr(candles: list, period: int = 14) -> float:
    trs = [candles[i]["high"] - candles[i]["low"] for i in range(max(0, len(candles) - period), len(candles))]
    return sum(trs) / len(trs) if trs else 1.0


def detect_zones(candles: list, trend: str, params: dict) -> list:
    """
    Find supply/demand zones in the candle history.
    Demand = consolidation before bullish impulse (uptrend only).
    Supply  = consolidation before bearish impulse (downtrend only).
    """
    threshold   = params["impulse_threshold"]
    body_ratio  = params["impulse_body_ratio"]
    max_age     = params["zone_max_age_bars"]
    atr         = _atr(candles)
    zones       = []
    start_idx   = max(1, len(candles) - max_age)

    for i in range(start_idx, len(candles) - 1):
        c    = candles[i]
        prev = candles[i - 1]
        c_range = c["high"] - c["low"]
        if atr <= 0 or c_range / atr < threshold:
            continue
        body = abs(c["close"] - c["open"])
        if c_range <= 0 or body / c_range < body_ratio:
            continue

        if c["close"] > c["open"] and trend == "uptrend":
            zones.append({
                "direction": "demand",
                "zone_low":  prev["low"],
                "zone_high": prev["high"],
                "origin_ts": prev["timestamp"],
            })
        elif c["close"] < c["open"] and trend == "downtrend":
            zones.append({
                "direction": "supply",
                "zone_low":  prev["low"],
                "zone_high": prev["high"],
                "origin_ts": prev["timestamp"],
            })

    return zones


def zone_is_active(zone: dict, candles: list) -> bool:
    """
    A zone is invalidated if any candle AFTER the impulse closed through it
    (demand broken below zone_low; supply broken above zone_high).
    """
    after = [c for c in candles if c["timestamp"] > zone["origin_ts"]]
    for c in after:
        if zone["direction"] == "demand" and c["close"] < zone["zone_low"]:
            return False
        if zone["direction"] == "supply" and c["close"] > zone["zone_high"]:
            return False
    return True


def zone_touched(candle: dict, zone: dict) -> bool:
    """Price has re-entered the zone on this candle."""
    if zone["direction"] == "demand":
        return candle["low"] <= zone["zone_high"] and candle["close"] >= zone["zone_low"]
    else:
        return candle["high"] >= zone["zone_low"] and candle["close"] <= zone["zone_high"]


def build_trade(zone: dict, entry_price: float, candles: list,
                params: dict) -> dict | None:
    """Compute SL, TP, check R:R. Returns trade dict or None."""
    buf     = params["zone_buffer_pct"]
    min_rr  = params["min_rr"]
    n       = params["swing_n"]
    highs   = _swing_highs(candles, n)
    lows    = _swing_lows(candles, n)

    if zone["direction"] == "demand":
        sl = zone["zone_low"] * (1 - buf)
        risk = entry_price - sl
        if risk <= 0:
            return None
        tp_candidates = [h["price"] for h in highs if h["price"] > entry_price]
        if not tp_candidates:
            return None
        tp = min(tp_candidates)
        rr = (tp - entry_price) / risk
        direction = "long"
    else:
        sl = zone["zone_high"] * (1 + buf)
        risk = sl - entry_price
        if risk <= 0:
            return None
        tp_candidates = [l["price"] for l in lows if l["price"] < entry_price]
        if not tp_candidates:
            return None
        tp = max(tp_candidates)
        rr = (entry_price - tp) / risk
        direction = "short"

    if rr < min_rr:
        logger.info(f"Zone {zone['direction']} R:R={rr:.2f} < {min_rr} — skipping")
        return None

    return {
        "direction": direction,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(sl, 2),
        "target": round(tp, 2),
        "rr": round(rr, 2),
        "zone_low": zone["zone_low"],
        "zone_high": zone["zone_high"],
        "zone_ts": zone["origin_ts"],
    }


# ── main loop tick ─────────────────────────────────────────────────────────────

async def loop_once(state: dict) -> dict:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    p = load_params()

    candles = await fetch_ohlcv(granularity=GRANULARITY, limit=120)
    if len(candles) < 20:
        logger.warning(f"Only {len(candles)} candles — need 20+, skipping")
        return state

    completed = candles[:-1]
    last      = completed[-1]
    price     = last["close"]

    # ── exit check ─────────────────────────────────────────────────────────────
    if state["current_trade"]:
        trade = state["current_trade"]
        sl, tp = trade["stop_loss"], trade["target"]
        closed = None

        if trade["direction"] == "long":
            if last["high"] >= tp:
                closed = {**trade, "exit_price": tp, "exit_reason": "tp",
                          "pnl_pct": round((tp - trade["entry_price"]) / trade["entry_price"], 6)}
            elif last["low"] <= sl:
                closed = {**trade, "exit_price": sl, "exit_reason": "sl",
                          "pnl_pct": round((sl - trade["entry_price"]) / trade["entry_price"], 6)}
        else:
            if last["low"] <= tp:
                closed = {**trade, "exit_price": tp, "exit_reason": "tp",
                          "pnl_pct": round((trade["entry_price"] - tp) / trade["entry_price"], 6)}
            elif last["high"] >= sl:
                closed = {**trade, "exit_price": sl, "exit_reason": "sl",
                          "pnl_pct": round((trade["entry_price"] - sl) / trade["entry_price"], 6)}

        if closed:
            closed["close_ts"] = now_ts
            log_trade(closed)
            logger.info(
                f"Trade CLOSED | {closed['direction']} | entry={closed['entry_price']:.2f} "
                f"exit={closed['exit_price']:.2f} reason={closed['exit_reason']} "
                f"pnl={closed['pnl_pct']*100:.3f}%"
            )
            state["current_trade"] = None
            if reflection_due():
                logger.info("Reflection triggered")
                try:
                    from hermes_trading.reflect import run_reflection
                    run_reflection()
                except Exception as e:
                    logger.error(f"Reflection failed: {e}")
        else:
            logger.info(
                f"Trade OPEN | {trade['direction']} @ {trade['entry_price']:.2f} "
                f"SL={sl:.2f} TP={tp:.2f} R:R={trade['rr']} | price={price:.2f}"
            )
        return state

    # ── market structure ───────────────────────────────────────────────────────
    trend = detect_trend(completed, n=p["swing_n"])
    if not trend:
        logger.info(f"No clear trend (ranging) — standing by | price={price:.2f}")
        return state

    # ── zone scan ─────────────────────────────────────────────────────────────
    zones = detect_zones(completed, trend, p)
    used  = set(state.get("used_zone_ts", []))

    for zone in reversed(zones):  # most recent first
        if zone["origin_ts"] in used:
            continue
        if not zone_is_active(zone, completed):
            continue
        if not zone_touched(last, zone):
            continue

        trade = build_trade(zone, price, completed, p)
        if not trade:
            continue

        trade["entry_ts"] = now_ts
        trade["open_ts"]  = now_ts
        trade["mode"]     = "paper"
        state["current_trade"] = trade
        used.add(zone["origin_ts"])
        state["used_zone_ts"] = list(used)[-100:]
        logger.info(
            f"ENTRY | {trade['direction']} @ {trade['entry_price']:.2f} "
            f"SL={trade['stop_loss']:.2f} TP={trade['target']:.2f} R:R={trade['rr']} "
            f"| trend={trend} zone=[{zone['zone_low']:.0f}-{zone['zone_high']:.0f}]"
        )
        return state

    logger.info(
        f"Watching | trend={trend} | price={price:.2f} | "
        f"active zones={sum(1 for z in zones if zone_is_active(z, completed))}"
    )
    return state


async def loop_forever() -> None:
    logger.info("Booting hermes-trading worker | BTC-5M-SupplyDemand | paper mode")
    logger.info("Strategy: 3-step price action — market structure + S/D zones + R:R filter")

    state = load_state()
    tick  = 0

    while True:
        tick += 1
        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] Tick {tick}")
        try:
            state = await loop_once(state)
            save_state(state)
        except Exception as e:
            logger.error(f"Tick error: {e}", exc_info=True)
        await asyncio.sleep(GRANULARITY)
