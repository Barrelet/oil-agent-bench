"""Scoring: returns, trades, the 1,000 random traders, and the daily forecasts.

Three questions, the three lessons of September, asked of every agent:
  1. the trades behind the return       -> trade_metrics
  2. doing nothing, holding oil, a coin -> baselines + random_traders + percentile
  3. does the stated probability mean anything -> forecast_scores, calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import RandomTrader
from .data import Market
from .replay import run

TRADING_DAYS = 252


def pnl_metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1.0
    return {"total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
            "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
            "max_drawdown": float(dd.min()), "volatility_ann": float(r.std() * np.sqrt(TRADING_DAYS))}


def trade_metrics(trades: pd.DataFrame, log: pd.DataFrame | None = None) -> dict:
    out = {"n_trades": 0, "win_rate": np.nan, "avg_holding_days": np.nan, "stopped_out": 0, "fees": 0.0}
    if trades is not None and not trades.empty:
        out.update(n_trades=int(len(trades)), win_rate=float((trades["pnl"] > 0).mean()),
                   avg_holding_days=float(trades["holding_days"].mean()), stopped_out=int((trades["reason"] == "stop").sum()),
                   fees=float(trades["fees"].sum()), best_trade_pnl=float(trades["pnl"].max()),
                   best_trade_share=float(trades["pnl"].max() / trades["pnl"].sum()) if trades["pnl"].sum() > 0 else np.nan)
    if log is not None and not log.empty:
        out["days_long"] = int((log["held"] == "long").sum())
        out["days_short"] = int((log["held"] == "short").sum())
        out["days_flat"] = int((log["held"] == "flat").sum())
        out["invalid_answers"] = int((log["invalid"].fillna("") != "").sum())
    return out


def scoreboard(results: dict[str, dict]) -> pd.DataFrame:
    rows = {k: {**pnl_metrics(v["balance"]), **trade_metrics(v["trades"], v["log"])} for k, v in results.items()}
    return pd.DataFrame(rows).T


# ---- the coin ----------------------------------------------------------------------------
def random_traders(md: Market, n: int = 1000, batch: int = 100, on_batch=None, **kw) -> pd.DataFrame:
    """n seeded RandomTraders over the same days and rules; one row per trader.
    on_batch(done, n, seconds) is called after each batch (for a progress bar)."""
    import time
    rows, t0 = [], time.time()
    for b0 in range(0, n, batch):
        agents = [RandomTrader(seed=s) for s in range(b0, min(b0 + batch, n))]
        res = run(md, agents, **kw)
        for a in agents:
            r = res[a.name]
            rows.append({"seed": a.seed, **pnl_metrics(r["balance"]), **trade_metrics(r["trades"])})
        if on_batch is not None:
            on_batch(len(rows), n, time.time() - t0)
    return pd.DataFrame(rows)


def random_traders_cached(md: Market, folder, start, end, n: int = 1000, on_batch=None) -> pd.DataFrame:
    """The 1,000 random traders for exactly this window, played once and saved under a file
    name that carries the window (so a 10-day test can never be mistaken for the year).
    The decision dates inside the file are checked again on load."""
    from pathlib import Path
    from .replay import decision_dates
    dds = decision_dates(md, start, end)
    tag = f"{dds[0].date()}_{dds[-1].date()}_{len(dds)}d_n{n}"
    path = Path(folder) / f"random_traders_{tag}.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) == n and set(df.get("window", [tag])) == {tag}:
            return df
    df = random_traders(md, n=n, start=start, end=end, on_batch=on_batch)
    df["window"] = tag
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def beats(null: pd.Series, value: float) -> float:
    """How many random traders in 100 the value beats (ties count as half). A coin scores 50."""
    v = null.dropna().to_numpy()
    return float(100.0 * ((v < value).mean() + 0.5 * (v == value).mean()))


# ---- the forecasts -------------------------------------------------------------------------
def forecast_frame(log: pd.DataFrame) -> pd.DataFrame:
    """One row per day: the stated p_up and whether the next close was higher than today's.
    Every player is scored on the SAME days: a day without a usable answer counts as 0.5
    ("no view", flagged in `missing`), and only days when the close did not change at all
    are dropped, which depends on prices alone. With quoted prices, a contract-switch day's
    move includes the small jump between two contracts, the same for every player."""
    f = log[["decision_date", "p_up", "close", "next_close"]].copy()
    f["missing"] = f["p_up"].isna()
    f["p_up"] = f["p_up"].astype(float).fillna(0.5)
    f["up"] = (f["next_close"] > f["close"]).astype(int)
    return f[f["next_close"] != f["close"]]


def forecast_scores(log: pd.DataFrame) -> dict:
    f = forecast_frame(log)
    if f.empty:
        return {}
    p, y = f["p_up"].to_numpy(float), f["up"].to_numpy(int)
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((0.5 - y) ** 2))                      # the 'no view' forecast: always 0.5
    hits = (p > 0.5) == (y == 1)
    called = p != 0.5
    return {"n_forecasts": int(len(f)), "n_missing": int(f["missing"].sum()), "brier": brier, "brier_no_view": base, "brier_skill": 1.0 - brier / base if base else np.nan,
            "hit_rate": float(hits[called].mean()) if called.any() else np.nan, "share_up_days": float(y.mean()),
            "mean_p_up": float(p.mean()), "distinct_values": int(len(np.unique(np.round(p, 3))))}


def calibration(log: pd.DataFrame, edges=(0, 0.35, 0.45, 0.55, 0.65, 1.0001)) -> pd.DataFrame:
    """Stated probability bucket vs the share of days that actually closed higher."""
    f = forecast_frame(log)
    f["bucket"] = pd.cut(f["p_up"], edges, right=False)
    return f.groupby("bucket", observed=True).agg(days=("up", "size"), stated=("p_up", "mean"), realised=("up", "mean"))
