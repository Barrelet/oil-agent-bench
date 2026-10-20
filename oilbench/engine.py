"""Paper account for one market: long 100%, short 100% or flat.

Mechanics carried over from the September engine:
  * a decision taken at the close of t fills at the open of t+1
  * a fee on every fill (opening and closing), no separate slippage
  * the stop is checked against the low (long) or high (short) of each day from the
    fill day on; a stop gapped through fills at the open, otherwise at the stop
  * marked at the close; everything still open is closed at the last close (fee charged)
New for oil (decisions of 19 Oct 2026):
  * one market, three positions; size = the whole account, no leverage
  * fee 0.02% per fill (rules.FEE_RATE)
  * the game, not the player, sets a disaster stop when a position opens: rules.STOP_ATR_MULT
    times the ATR known at the decision close, measured from the fill price, but never more
    than rules.STOP_MAX_PCT (15%) from it. It never moves.
    The hold-oil benchmark is the one exemption (Book(disaster_stop=False)).
  * asking for the position already held is a hold
Prices are the front-month price as quoted (see data.py): a position held across a
monthly contract switch carries the small jump between the two contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .rules import FEE_RATE, INITIAL_CAPITAL, SIZE, STOP_ATR_MULT, STOP_MAX_PCT
from .schema import Decision

SIDE = {"long": 1, "short": -1, "flat": 0}


@dataclass
class Trade:
    side: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float              # net of both fees
    pnl_pct: float          # gross price move in the direction of the trade
    fees: float
    reason: str             # close | stop | end_of_window
    holding_days: int
    p_up_at_entry: float


@dataclass
class Book:
    initial_capital: float = INITIAL_CAPITAL
    fee_rate: float = FEE_RATE
    cash: float = field(init=False)
    side: int = 0
    qty: float = 0.0
    entry_price: float = np.nan
    entry_date: pd.Timestamp | None = None
    entry_fee: float = 0.0
    stop: float | None = None
    entry_p_up: float = np.nan
    trades: list[Trade] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    balance_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    fees_paid: float = 0.0
    disaster_stop: bool = True     # False only for the hold-oil benchmark
    stop_atr_mult: float = STOP_ATR_MULT
    stop_max_pct: float = STOP_MAX_PCT

    def __post_init__(self):
        self.cash = float(self.initial_capital)

    @property
    def position(self) -> str:
        return {1: "long", -1: "short", 0: "flat"}[self.side]

    def balance(self, price: float) -> float:
        return self.cash + self.side * self.qty * price

    def snapshot(self, close: float) -> dict:
        snap = {"position": self.position, "balance": round(self.balance(close), 2)}
        if self.side:
            snap.update(entry_price=round(self.entry_price, 2), stop_price=None if self.stop is None else round(self.stop, 2),
                        entry_date=str(self.entry_date.date()), unrealized_pct=round(100 * self.side * (close / self.entry_price - 1), 2))
        return snap

    # ---- fills -----------------------------------------------------------------------
    def _close(self, date, price: float, reason: str) -> Trade:
        notional = self.qty * price
        fee = notional * self.fee_rate
        self.cash += self.side * notional - fee        # sell a long / buy back a short
        self.fees_paid += fee
        t = Trade(side=self.side, entry_date=self.entry_date, exit_date=pd.Timestamp(date), entry_price=self.entry_price,
                  exit_price=price, qty=self.qty, pnl=self.side * self.qty * (price - self.entry_price) - fee - self.entry_fee,
                  pnl_pct=self.side * (price / self.entry_price - 1.0), fees=fee + self.entry_fee, reason=reason,
                  holding_days=int((pd.Timestamp(date) - self.entry_date).days), p_up_at_entry=self.entry_p_up)
        self.trades.append(t)
        self.side, self.qty, self.entry_price, self.entry_date, self.entry_fee, self.stop = 0, 0.0, np.nan, None, 0.0, None
        return t

    def _open(self, date, side: int, price: float, atr: float | None, p_up: float | None):
        bal = self.balance(price)
        qty = SIZE * bal / (price * (1.0 + self.fee_rate))   # the whole account, fee included
        fee = qty * price * self.fee_rate
        self.cash += -side * qty * price - fee
        self.fees_paid += fee
        self.side, self.qty, self.entry_price, self.entry_date, self.entry_fee = side, qty, price, pd.Timestamp(date), fee
        dist = min(self.stop_atr_mult * atr, self.stop_max_pct * price) if atr else None   # 3 ATRs, at most 15%
        self.stop = (price - side * dist) if (self.disaster_stop and dist) else None
        self.entry_p_up = np.nan if p_up is None else float(p_up)

    # ---- one trading day ------------------------------------------------------------
    def step(self, decision: Decision, decision_close: float, date, bar: pd.Series, atr: float | None = None) -> None:
        """Apply the decision taken at `decision_close` on day `date` whose bar is `bar`:
        fill at the open (placing the disaster stop from `atr`, the ATR known at the decision
        close), check the stop against the day's range, mark at the close."""
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        want = SIDE.get(decision.position, None)
        if want is None:
            self.rejections.append({"date": date, "why": f"unknown position {decision.position!r}"})
            want = self.side
        if want != 0 and want != self.side and self.disaster_stop and not (atr and atr > 0):
            raise ValueError(f"no ATR to place the disaster stop on {date}")
        if want != self.side:
            if self.side:
                self._close(date, o, "close")
            if want:
                self._open(date, want, o, atr, decision.p_up)
        if self.side and self.stop is not None:
            if self.side > 0 and l <= self.stop:
                self._close(date, min(self.stop, o), "stop")
            elif self.side < 0 and h >= self.stop:
                self._close(date, max(self.stop, o), "stop")
        self.balance_curve.append((pd.Timestamp(date), self.balance(c)))

    def close_all(self, date, close: float) -> None:
        if self.side:
            self._close(date, close, "end_of_window")
            self.balance_curve[-1] = (pd.Timestamp(date), self.balance(close))

    # ---- export -----------------------------------------------------------------------
    def balance_series(self) -> pd.Series:
        if not self.balance_curve:
            return pd.Series(dtype=float, name="balance")
        d, v = zip(*self.balance_curve)
        return pd.Series(v, index=pd.DatetimeIndex(d), name="balance")

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])
