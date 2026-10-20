"""WTI front-month prices: load, mark contract rolls, and (optionally) adjust for them.

DECISION (19 Oct 2026): the game trades the front-month price AS QUOTED.
No roll adjustment. `Market.from_raw(raw)` keeps Yahoo's CL=F prices unchanged.

Why. A continuous front-month series switches to the next contract every month,
and the jump between two contracts is not a return anyone could earn. The usual fix
is to set aside the price change on each roll day. On the 2025-26 data that fix did
more harm than good:
  * Yahoo's series does not show clearly when it switches contracts: on most expiry
    days there is no visible gap at the open, so the switch cannot be isolated.
  * Two of the year's roll days were large real market moves: 23 March 2026 (-10.4%,
    opened higher at 100.51, traded down to 84.37) and 20 May 2026 (-8.8%).
    Setting them aside spared a long holder two large losses and turned holding oil
    from +46.1% (quoted price, 61.78 to 90.42, after fees) into +79.6%.
Using the quoted price keeps what readers see on an oil chart. The cost: each monthly
switch adds a small jump to the price (contango or backwardation, usually 0.5-2%),
which every player faces equally. This is stated in the notebook and the posts.

The roll-adjusted version is kept (`Market.from_raw(raw, adjust_rolls=True)`) for a
robustness check. It works as follows (forward adjustment, point-in-time): on a roll
day the close-to-close move is set aside, the new contract is scaled so its close
equals the previous close, and every later day of that contract is scaled by the same
factor. The factor on any date depends only on rolls up to that date, so the adjusted
series never uses future information.

Roll dates (marked in `is_roll` either way, for information). WTI (CL) trading ends 3
business days before the 25th calendar day of the month before delivery (if the 25th
is not a business day, 3 business days before the last business day preceding it).
Same rule as the oil lab. The switch day is taken as the first trading day after expiry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")
PRICE_FIELDS = ("open", "high", "low", "close")


# ---- loading ---------------------------------------------------------------------
def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV with columns date, open, high, low, close[, volume]; one row per trading day."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return clean(df.set_index("date")[list(FIELDS)])


def download_yahoo(start: str = "2007-01-01", end: str | None = None, ticker: str = "CL=F") -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance (runs on your Mac; not from Claude's workspace).
    Prices are consumed, not stored: keep any saved copy out of a public repo."""
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return clean(raw[list(FIELDS)])


def load_or_download(path: str | Path, refresh: bool = False, start: str = "2015-01-01") -> pd.DataFrame:
    """Prices from the local file; download from Yahoo first if it is missing (or refresh=True).
    The file stays on the Mac and is never committed."""
    path = Path(path)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        download_yahoo(start=start).reset_index().to_csv(path, index=False)
    return load_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be traded honestly: missing prices, non-positive prices
    (WTI printed below zero on 20 Apr 2020, outside our window), and bars whose high/low
    do not contain the open and close."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=list(PRICE_FIELDS))
    df = df[(df[list(PRICE_FIELDS)] > 0).all(axis=1)]
    ok = (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9) & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)
    return df[ok].astype(float)


# ---- contract rolls ---------------------------------------------------------------
def expiry_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each expiring front-month contract, on the given trading calendar."""
    days = pd.DatetimeIndex(trading_days).sort_values()
    last = days[-1]
    # The exchange calendar is known in advance; past the end of the data we count plain
    # weekdays, so the roll dates inside the data never depend on how far the data runs.
    days = days.union(pd.bdate_range(last + pd.Timedelta(days=1), periods=30))
    out = []
    for month_start in pd.date_range(days[0].replace(day=1), last, freq="MS"):
        d25 = month_start + pd.Timedelta(days=24)
        i = days.searchsorted(d25)                 # first trading day on/after the 25th
        if i < len(days) and days[i] == d25:       # the 25th is a business day
            k = i - 3
        else:                                      # last business day before the 25th, then 3 more back
            k = (i - 1) - 3
        if 0 <= k < len(days):
            out.append(days[k])
    return pd.DatetimeIndex(out)


def roll_days(trading_days: pd.DatetimeIndex, overrides: list[str] | None = None) -> pd.DatetimeIndex:
    """First trading day of each new front-month contract (the day after expiry).
    `overrides` replaces the calendar with a verified list (e.g. from the oil lab)."""
    days = pd.DatetimeIndex(trading_days).sort_values()
    if overrides is not None:
        return pd.DatetimeIndex(pd.to_datetime(overrides)).intersection(days)
    exp = expiry_dates(days)
    idx = days.searchsorted(exp, side="right")
    return pd.DatetimeIndex([days[i] for i in idx if i < len(days)])


def roll_adjusted(raw: pd.DataFrame, rolls: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-adjusted OHLC (see module docstring). Adds `factor` and `is_roll` columns;
    volume is left as reported."""
    df = raw.copy()
    is_roll = df.index.isin(rolls)
    factor = np.ones(len(df))
    close = df["close"].to_numpy()
    for i in range(1, len(df)):
        factor[i] = factor[i - 1] * (close[i - 1] / close[i] if is_roll[i] else 1.0)
    for f in PRICE_FIELDS:
        df[f] = df[f] * factor
    df["factor"] = factor
    df["is_roll"] = is_roll
    return df


# ---- the object everything downstream receives --------------------------------------
@dataclass(frozen=True)
class Market:
    """Daily bars of one market: the quoted front-month price by default, roll-adjusted only
    for the robustness check. `asof(t)` returns only bars <= t: the single look-ahead guard. Payloads are built only from an `asof` slice; the engine
    reads the next day's bar explicitly."""
    bars: pd.DataFrame   # index = date; open, high, low, close, volume, factor (1 unless adjusted), is_roll

    @classmethod
    def from_raw(cls, raw: pd.DataFrame, adjust_rolls: bool = False, roll_overrides: list[str] | None = None) -> "Market":
        """Quoted front-month prices by default (see the module docstring); adjust_rolls=True
        gives the roll-adjusted series used only as a robustness check."""
        rolls = roll_days(raw.index, roll_overrides)
        if adjust_rolls:
            return cls(roll_adjusted(raw, rolls))
        bars = raw.copy()
        bars["factor"] = 1.0
        bars["is_roll"] = bars.index.isin(rolls)
        return cls(bars)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.bars.index

    def asof(self, t) -> "Market":
        return Market(self.bars.loc[: pd.Timestamp(t)])

    def bar(self, t) -> pd.Series:
        return self.bars.loc[pd.Timestamp(t)]

    def next_date(self, t) -> pd.Timestamp | None:
        i = self.dates.searchsorted(pd.Timestamp(t), side="right")
        return self.dates[i] if i < len(self.dates) else None


def synthetic(n_days: int = 900, seed: int = 0, start: str = "2023-01-02", vol_ann: float = 0.32,
              roll_gap: float = 0.01) -> pd.DataFrame:
    """Fake WTI-like bars for tests: a random walk with intraday ranges and a monthly
    contract switch that jumps the price by `roll_gap` (contango)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    rolls = roll_days(days)
    sd = vol_ann / np.sqrt(252)
    close = np.empty(n_days); open_ = np.empty(n_days)
    px = 75.0
    for i, d in enumerate(days):
        if d in rolls:
            px *= 1 + roll_gap                     # new contract trades higher: not a return
        open_[i] = px * np.exp(rng.normal(0, sd / 3))
        px = open_[i] * np.exp(rng.normal(0, sd))
        close[i] = px
    span = np.abs(rng.normal(0, sd * 0.4, n_days)) * np.maximum(open_, close)   # keeps ATR near real WTI (~2%)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    vol = rng.integers(150_000, 400_000, n_days).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=pd.DatetimeIndex(days, name="date"))
