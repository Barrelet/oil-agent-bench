"""Walk the year day by day and run every agent through the same days.

For each decision date t whose next trading day t+1 is still inside the window:
  1. slice the market as of t (the look-ahead guard) and build the day's payload once
  2. each agent sees its own position and decides
  3. its book fills at the open of t+1, checks the stop on t+1, marks at the close of t+1
Everything still open is closed at the last close. Agents step in lockstep, so the
1,000 random traders share the per-day work.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .agents import Agent
from .data import Market
from .engine import Book
from .prompt import build_base, with_position
from .rules import INITIAL_CAPITAL, WINDOW_END, WINDOW_START
from .schema import Decision

log = logging.getLogger(__name__)
WARMUP_BARS = 60


def decision_dates(md: Market, start=WINDOW_START, end=WINDOW_END, warmup: int = WARMUP_BARS) -> pd.DatetimeIndex:
    """Days on which decisions are taken: from `start`, up to the last day whose NEXT trading
    day (the fill and mark) is still on or before `end`. So the season's first trade fills on
    the first trading day after `start`, and its last mark is the last trading day on or
    before `end`: nothing is measured outside the window."""
    dates = md.dates
    i0 = max(dates.searchsorted(pd.Timestamp(start)), warmup)
    i_end = dates.searchsorted(pd.Timestamp(end), side="right") - 1      # last trading day <= end
    return dates[i0:max(i0, i_end)]


def season_span(md: Market, start=WINDOW_START, end=WINDOW_END) -> dict:
    """What a season covers, for labels: first decision, first fill, last mark, number of days."""
    dds = decision_dates(md, start, end)
    return {"first_decision": str(dds[0].date()), "first_fill": str(md.next_date(dds[0]).date()),
            "last_mark": str(md.next_date(dds[-1]).date()), "days": len(dds)}


def end_after(md: Market, n_days: int, start=WINDOW_START) -> pd.Timestamp:
    """The `end` that gives exactly n_days decisions from `start` (for a quick look)."""
    dds = decision_dates(md, start, md.dates[-1])
    return md.next_date(dds[n_days - 1])


def run(md: Market, agents: list[Agent], start=WINDOW_START, end=WINDOW_END, anonymise: bool = False,
        on_day=None, on_decision=None, log_dir: Path | None = None) -> dict[str, dict]:
    """on_day(i, n, date, seconds) after each day; on_decision(agent_name, row) after each
    agent's decision (row = the day's log entry), for a live view in a notebook."""
    dds = decision_dates(md, start, end)
    if len(dds) == 0:
        raise ValueError("no decision dates: check the window and the data")
    books = {a.name: Book(disaster_stop=not a.exempt_from_stop) for a in agents}
    logs = {a.name: [] for a in agents}
    history = any(a.needs_history for a in agents)
    for a in agents:
        a.reset()
        books[a.name].balance_curve.append((dds[0], INITIAL_CAPITAL))
    t0 = time.time()
    for i, t in enumerate(dds):
        md_t = md.asof(t)
        base = build_base(md_t, anonymise=anonymise, with_history=history)
        t1 = md.next_date(t)
        bar1 = md.bar(t1)
        close_t = base["_close"]
        for a in agents:
            book = books[a.name]
            payload = with_position(base, book.snapshot(close_t))
            try:
                d = a.decide(payload)
            except Exception as e:
                # A failure to get an answer (model server down, network, a bug) stops the run.
                # It must never be scored as the player choosing to hold. Answers already
                # received are saved, so running again resumes where this stopped.
                raise RuntimeError(f"{a.name} could not answer for {t.date()} ({type(e).__name__}: {e}). "
                                   "Nothing is scored for this day. Fix the cause and run again: "
                                   "answers already received are saved and will not be asked again.") from e
            if not isinstance(d, Decision):
                raise TypeError(f"{a.name} must return a Decision")
            n_trades = len(book.trades)
            book.step(d, close_t, t1, bar1, atr=base["_atr"])
            logs[a.name].append({"decision_date": str(t.date()), "fill_date": str(t1.date()), "close": close_t,
                                 "next_close": float(bar1["close"]), "wanted": d.position, "held": book.position,
                                 "p_up": d.p_up, "active_stop": book.stop, "atr": base["_atr"], "invalid": d.invalid, "thesis": d.thesis,
                                 "reasoning": d.reasoning, "closed_trades": len(book.trades) - n_trades,
                                 "balance": round(book.balance_curve[-1][1], 2)})
            if on_decision is not None:
                on_decision(a.name, logs[a.name][-1])
        if on_day is not None:
            on_day(i + 1, len(dds), t, time.time() - t0)
    out = {}
    last = dds[-1]
    last_close = float(md.bar(md.next_date(last))["close"])
    for a in agents:
        b = books[a.name]
        b.close_all(b.balance_curve[-1][0], last_close)
        out[a.name] = {"balance": b.balance_series(), "trades": b.trades_df(), "log": pd.DataFrame(logs[a.name]),
                       "rejections": pd.DataFrame(b.rejections), "book": b}
    if log_dir:
        save(out, Path(log_dir))
    return out


def load_saved(folder: Path) -> dict[str, dict]:
    """Reload a run saved by `run(..., log_dir=folder)`: {agent: {balance, trades, log}}."""
    folder = Path(folder)
    bal = pd.read_csv(folder / "balance.csv", index_col=0, parse_dates=True)
    trades = pd.read_csv(folder / "trades.csv") if (folder / "trades.csv").stat().st_size > 2 else pd.DataFrame(columns=["agent"])
    for c in ("entry_date", "exit_date"):
        if c in trades:
            trades[c] = pd.to_datetime(trades[c])
    logs = pd.read_json(folder / "decisions.jsonl", lines=True, dtype=False)
    out = {}
    for name in bal.columns:
        log = logs[logs["agent"] == name].drop(columns="agent").reset_index(drop=True)
        for c in ("invalid", "thesis", "reasoning"):
            if c in log:
                log[c] = log[c].fillna("")
        out[name] = {"balance": bal[name].dropna().rename("balance"),
                     "trades": trades[trades["agent"] == name].drop(columns="agent").reset_index(drop=True) if len(trades) else trades,
                     "log": log}
    return out


def collect_saved(results_dir: Path, names: tuple[str, ...] = ("qwen3-8b_P1", "qwen3-8b_P2", "qwen3-8b_P3",
                                                              "gpt-5-1_P1", "gpt-5-1_P2", "gpt-5-1_P3", "timesfm_2p5")) -> dict[str, dict]:
    """Every player already played and saved under results_dir/<name>/, in a fixed order."""
    found = {}
    for n in names:
        if (Path(results_dir) / n / "balance.csv").exists():
            found.update(load_saved(Path(results_dir) / n))
    return found


def save(results: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v["balance"] for k, v in results.items()}).to_csv(out / "balance.csv")
    parts = [v["trades"].assign(agent=k) for k, v in results.items() if not v["trades"].empty]
    (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_csv(out / "trades.csv", index=False)
    with open(out / "decisions.jsonl", "w") as f:
        for k, v in results.items():
            for row in v["log"].to_dict("records"):
                f.write(json.dumps({"agent": k, **row}, default=str) + "\n")
