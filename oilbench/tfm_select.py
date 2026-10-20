"""Choosing what TimesFM is shown (prices or returns) and how much history, on the PREVIOUS year.

Decision of 19 Oct 2026, fixed before any of these versions was scored:
  * 12 versions: 2 inputs (prices, daily returns) x 6 history lengths (3 months, 6 months,
    1, 2, 4 and 10 years, in trading days).
  * Each version forecasts the next close on every trading day of the development year,
    1 October 2024 to 30 September 2025 (outside the test year, and before it).
  * Scored on its forecasts only, never on P&L: the Brier skill of its daily chance of a rise
    against always saying 50%. Direction accuracy is reported alongside.
  * The version with the best Brier skill is chosen, IF that skill is above MIN_SKILL (+0.01).
    Otherwise no version has real forecasting skill, and Google's default (prices, 1,024 days)
    is kept. The chosen version then plays the test year once, with no further changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import tfm_series, tfm_view
from .data import Market
from .metrics import forecast_scores
from .replay import decision_dates

DEV_START, DEV_END = "2024-10-01", "2025-09-30"
INPUTS = ("price", "returns")
CONTEXTS = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504, "4 years": 1024, "10 years": 2520}
MIN_SKILL = 0.01
DEFAULT = ("price", 1024)


def dev_forecasts(md: Market, forecaster, start: str = DEV_START, end: str = DEV_END,
                  inputs=INPUTS, contexts: dict = CONTEXTS, on_version=None) -> pd.DataFrame:
    """One row per (version, day): the forecast for the next close, scored against what happened.
    Uses only prices up to each day (the same look-ahead guard as the game)."""
    dds = decision_dates(md, start, end)
    closes_all = md.bars["close"]
    hist = [closes_all.loc[:t].to_numpy() for t in dds]
    nxt = np.array([float(md.bar(md.next_date(t))["close"]) for t in dds])
    rows = []
    for kind in inputs:
        for label, n in contexts.items():
            series = [tfm_series(h, kind, n) for h in hist]
            qs = forecaster.many(series) if hasattr(forecaster, "many") else [forecaster(s) for s in series]
            for t, h, q, nc in zip(dds, hist, qs, nxt):
                move, p_up = tfm_view(q, h[-1], kind)
                rows.append({"input": kind, "history": label, "context": n, "decision_date": str(t.date()),
                             "close": h[-1], "next_close": nc, "expected_move": move, "p_up": p_up,
                             "history_available": min(n, len(h) - (kind == "returns"))})
            if on_version is not None:
                on_version(kind, label)
    return pd.DataFrame(rows)


def score_versions(f: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (kind, label, n), g in f.groupby(["input", "history", "context"], sort=False):
        s = forecast_scores(g)
        up = g["next_close"] > g["close"]
        moved = g["next_close"] != g["close"]
        out.append({"input": kind, "history": label, "days": int(moved.sum()), "brier_skill": s["brier_skill"],
                    "brier": s["brier"], "direction_right": float(((g["expected_move"] > 0) == up)[moved].mean()),
                    "forecasts_down": float((g["expected_move"] < 0).mean()), "share_up_days": float(up[moved].mean()),
                    "median_abs_move": float(g["expected_move"].abs().median()),
                    "min_history_used": int(g["history_available"].min())})
    return pd.DataFrame(out).set_index(["input", "history"])


def choose(scores: pd.DataFrame) -> tuple[str, int, str]:
    """Apply the rule fixed in advance. Returns (input, context_days, reason)."""
    best = scores["brier_skill"].idxmax()
    skill = scores.loc[best, "brier_skill"]
    if skill > MIN_SKILL:
        return best[0], CONTEXTS[best[1]], f"best Brier skill {skill:+.3f} ({best[0]}, {best[1]}), above the +{MIN_SKILL} bar"
    return DEFAULT[0], DEFAULT[1], (f"no version beat always-50% by more than +{MIN_SKILL} (best {skill:+.3f}, {best[0]}, {best[1]}): "
                                    f"Google's default kept (prices, {DEFAULT[1]} days)")
