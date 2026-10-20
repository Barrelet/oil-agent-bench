"""What an agent sees each day, and the three prompts (drafts, adapted from September).

`build_payload(md_t, snapshot)` returns a plain dict built ONLY from an `asof(t)` slice.
Rule agents read its numbers; LLMs read `render_text(payload)`.

Two display modes (the game uses the first):
  anonymise=False : real dates and the front-month price as quoted.
  anonymise=True  : no dates (rows run from day -59 to day 0) and prices rebased so
                    today's close is 100. The model cannot tell which year it is,
                    so it cannot "remember" what came next. The engine converts any
                    stop shown back to market units (payload["_scale"]).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import Market

N_DAILY = 60
N_WEEKLY = 26

# ---- the three prompts --------------------------------------------------------------
# Same ladder as September: P1 = the leaderboard-style mandate, P2 = plus risk rules,
# P3 = a full decision procedure. Wording adapted to one oil future; to review.

MANDATE_P1 = (
    "You manage a $10,000 paper account that trades one market: front-month WTI crude oil futures. "
    "Each day, at the close, you choose the position you want for the next session: long (all of the account), "
    "short (all of the account) or flat (cash). Orders fill at the next open. Fees are 0.02% of the amount traded "
    "on every fill; there is no leverage. Every long or short carries a disaster stop set by the game at three times "
    "the 14-day average true range (ATR) from the entry price, but never more than 15% from it; the position is closed "
    "automatically if that price trades. "
    "You do not set or move this stop; you exit by choosing flat. Asking for the position you already hold keeps it. "
    "Each day you also state p_up, your probability that the next close will be above today's close; it is scored "
    "for accuracy on its own."
)

MANDATE_P2 = MANDATE_P1 + (
    " Risk discipline, in order of importance: (1) every change of position costs 0.02% per fill and a reversal "
    "costs two fills, so change only when your view has clearly changed; (2) when you have no clear view, flat is a "
    "valid position; (3) p_up is a probability, not a signal strength: 0.5 means no view."
)

# P3, version 3 (decision of 19 Oct 2026, after the second Qwen run): the model's own judgement, without the
# standby wording of version 2 ("most days do nothing", "values far from 0.5 rare", "open only when all agree"),
# under which Qwen stayed flat for 8.5 months and made one trade. Version 2 is kept as MANDATE_P3_V2 for the record.
MANDATE_P3 = """You are the trader of a $10,000 paper account in one market, front-month WTI crude oil futures, judged over a year on net-of-cost return, Sharpe ratio, drawdown, and on whether your daily probability forecasts match what happens. Each day, at the close, you choose the position for the next session: long (all of the account), short (all of the account) or flat. Orders fill at the next open; each fill costs 0.02%; there is no leverage.

HOW TO READ THE DATA. position = your current position, with entry, stop and unrealized_pct. daily = the last 60 sessions, weekly = the last 26 weeks; rows are [open, high, low, close, volume_thousands], oldest first. features are in percent: ret_1d/5d/20d/60d = returns; vol_20d_ann = annualised volatility; atr14_pct = average true range over 14 days (the typical daily move); sma20_pct/sma50_pct = price vs 20/50-day average; hi20_pct/lo20_pct = distance to the 20-day high/low; rsi14 = 14-day RSI.

DECISION PROCEDURE, every day, in this order:
1. Form a view. Read the trend (ret_20d, ret_60d, sma20_pct, sma50_pct), the momentum (rsi14, ret_5d) and the location (hi20_pct, lo20_pct), and decide which way the evidence leans and how strongly. You weigh them yourself; they do not all need to agree.
2. State p_up: your probability that the next close is above today's close. 0.5 means no view. Use whatever value the evidence supports.
3. If you are flat: open a position when, on balance, you expect a move larger than the 0.04% round-trip cost. Being flat is a position too: it earns nothing while oil moves several percent a month. Choose flat only when you genuinely have no view.
4. If you hold a position: keep it while your reason for it still holds. Day-to-day noise is not a reason to exit. Close it or reverse it when your view has changed.
5. Stop: every long or short carries a disaster stop set by the game at three times the 14-day ATR from the entry price, but never more than 15% from it. You do not set or move it; you exit by choosing flat.

RULES THE GAME ENFORCES: the disaster stop closes the position automatically if its price trades; asking for the position you hold keeps it.

Write reasoning as a short audit trail: what you checked, which way it leans, what you chose."""

MANDATE_P3_V2 = """You are the trader of a $10,000 paper account in one market, front-month WTI crude oil futures, judged over a year on net-of-cost return, Sharpe ratio, drawdown, and on whether your daily probability forecasts match what happens. Each day, at the close, you choose the position for the next session: long (all of the account), short (all of the account) or flat. Orders fill at the next open; each fill costs 0.02%; there is no leverage.

HOW TO READ THE DATA. position = your current position, with entry, stop and unrealized_pct. daily = the last 60 sessions, weekly = the last 26 weeks; rows are [open, high, low, close, volume_thousands], oldest first. features are in percent: ret_1d/5d/20d/60d = returns; vol_20d_ann = annualised volatility; atr14_pct = average true range over 14 days (the typical daily move); sma20_pct/sma50_pct = price vs 20/50-day average; hi20_pct/lo20_pct = distance to the 20-day high/low; rsi14 = 14-day RSI.

DECISION PROCEDURE, every day, in this order:
1. State p_up first: your probability that the next close is above today's close. Daily oil moves are close to a coin toss; 0.5 means no view, and values far from 0.5 should be rare.
2. If you hold a position, keep it unless the reason for it has failed or the stop is about to trade. Day-to-day noise is not a reason.
3. If you are flat, open only when trend (ret_20d, ret_60d, sma20_pct, sma50_pct), momentum (rsi14, ret_5d) and location (hi20_pct, lo20_pct) agree, and the expected move clearly exceeds the 0.04% round-trip cost.
4. Stop: every long or short carries a disaster stop set by the game at three times the 14-day ATR from the entry price, but never more than 15% from it. You do not set or move it; you exit by choosing flat.
5. Flat is a valid position. Most days the right answer is to do nothing.

RULES THE GAME ENFORCES: the disaster stop closes the position automatically if its price trades; asking for the position you hold keeps it.

Write reasoning as a short audit trail: what you checked, what you rejected, what you chose."""

MANDATES = {"P1": MANDATE_P1, "P2": MANDATE_P2, "P3": MANDATE_P3}

OUTPUT_CONTRACT = (
    "Respond with a single JSON object and nothing else: "
    '{"reasoning": string (<= 80 words), "p_up": number 0-1, "position": "long"|"short"|"flat", '
    '"thesis": string (<= 25 words)}.'
)


# ---- features -------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] > 0 else np.inf
    return float(100 - 100 / (1 + rs))


def _atr(b: pd.DataFrame, n: int = 14) -> float:
    """Average true range over the last n days, in price units (USD per barrel)."""
    c, h, l = b["close"], b["high"], b["low"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def features(b: pd.DataFrame) -> dict:
    c, h, l = b["close"], b["high"], b["low"]
    px = float(c.iloc[-1])
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    r = c.pct_change()

    def ret(n):
        return round(float(px / c.iloc[-1 - n] * 100 - 100), 2) if len(c) > n else None

    return {
        "ret_1d": ret(1), "ret_5d": ret(5), "ret_20d": ret(20), "ret_60d": ret(60),
        "vol_20d_ann": round(float(r.tail(20).std() * np.sqrt(252) * 100), 1),
        "atr14_pct": round(float(tr.tail(14).mean() / px * 100), 2),
        "sma20_pct": round(float(px / c.tail(20).mean() * 100 - 100), 2),
        "sma50_pct": round(float(px / c.tail(50).mean() * 100 - 100), 2) if len(c) >= 50 else None,
        "hi20_pct": round(float(px / h.tail(20).max() * 100 - 100), 2),
        "lo20_pct": round(float(px / l.tail(20).min() * 100 - 100), 2),
        "rsi14": int(round(rsi(c))),
    }


def _weekly(b: pd.DataFrame) -> pd.DataFrame:
    return b.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])


def _rows(b: pd.DataFrame, scale: float) -> list[list[float]]:
    return [[round(r.open * scale, 2), round(r.high * scale, 2), round(r.low * scale, 2), round(r.close * scale, 2),
             None if pd.isna(r.volume) else int(round(r.volume / 1e3))] for r in b.itertuples()]


# ---- payload --------------------------------------------------------------------------
def build_base(md_t: Market, anonymise: bool = False, n_daily: int = N_DAILY, n_weekly: int = N_WEEKLY,
               with_history: bool = False) -> dict:
    """The agent-independent part of a day's payload (computed once per date)."""
    b = md_t.bars
    t = b.index[-1]
    close = float(b["close"].iloc[-1])
    # Shown units. Not anonymised: the quoted price (factor is 1 unless the robustness
    # check adjusts for rolls).
    # Anonymised: today's close = 100.
    scale = 100.0 / close if anonymise else 1.0 / float(b["factor"].iloc[-1])
    daily, weekly = b.tail(n_daily), _weekly(b).tail(n_weekly)
    base = {
        "market": ("front-month WTI crude oil futures, USD per barrel (continuous front-month price as quoted)" if not anonymise
                   else "an oil futures market (prices rebased: today's close = 100)"),
        "price": round(close * scale, 2),
        "position": None,
        "features": features(b),
        "daily": {"rows": _rows(daily, scale)} if anonymise else
                 {"first_date": str(daily.index[0].date()), "last_date": str(t.date()), "rows": _rows(daily, scale)},
        "weekly": {"rows": _rows(weekly, scale)},
        "_date": str(t.date()),          # never rendered: for logs and scoring
        "_scale": scale,                 # shown price = market price x scale
        "_close": close,
        "_atr": _atr(b),                 # 14-day ATR in USD at this close: sets the disaster stop
        "_anonymise": anonymise,
    }
    if not anonymise:
        base = {"decision_date": str(t.date()), **base}
    if with_history:
        base["_closes"] = b["close"].to_numpy()
    return base


def with_position(base: dict, snapshot: dict) -> dict:
    """Add one agent's position (shown in the same units as the prices)."""
    pos = dict(snapshot)
    for k in ("entry_price", "stop_price"):
        if pos.get(k) is not None:
            pos[k] = round(pos[k] * base["_scale"], 2)
    if base["_anonymise"]:
        pos.pop("entry_date", None)
    return {**base, "position": pos}


def build_payload(md_t: Market, snapshot: dict, anonymise: bool = False, **kw) -> dict:
    return with_position(build_base(md_t, anonymise, **kw), snapshot)


def render_text(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    when = f"as of the {payload['decision_date']} close" if "decision_date" in payload else "as of today's close (day 0)"
    return f"DATA ({when}):\n{json.dumps(body, separators=(',', ':'))}\n\n{OUTPUT_CONTRACT}"


def estimate_tokens(text: str) -> int:
    return int(len(text) / 2.5)    # number-heavy JSON on digit-splitting tokenizers
