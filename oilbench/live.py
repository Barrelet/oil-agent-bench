"""A live view for notebooks: one line per day while a season plays, so you can watch
every decision as it is made.

    2025-10-02  WTI  61.42  qwen3-8b_P1    SHORT  stop   64.10  p_up 0.35  P&L     -40  downtrend below the 20-day average

Position = what the account holds after that day's decision and the next session (a stop
hit that session shows FLAT, with a note). Stop = the stop actually in force (set when the
trade opened; it cannot be moved). P&L = running total at the next session's close, net of fees.
"""

from __future__ import annotations

import time


class LiveView:
    """Pass `view.day` as on_day and `view.decision` as on_decision to `replay.run`."""

    def __init__(self, n_days: int | None = None, show_thesis: bool = True, every: int = 1):
        self.n_days, self.show_thesis, self.every = n_days, show_thesis, every
        self.t0 = time.time()
        self.count = 0

    def decision(self, name: str, row: dict) -> None:
        self.count += 1
        if self.count % self.every:
            return
        held = row["held"].upper()
        active = row.get("active_stop")
        stop = f"stop {active:7.2f}" if active is not None and row["held"] != "flat" else " " * 12
        p = "  -  " if row.get("p_up") is None else f"{row['p_up']:.2f}"
        pnl = row["balance"] - 10_000
        note = ""
        if row.get("invalid"):
            note = "  [answer unusable: position kept]"
        elif row["wanted"] != row["held"] and row.get("closed_trades"):
            note = "  [disaster stop hit]"
        elif row["wanted"] != row["held"]:
            note = f"  [wanted {row['wanted']}, not filled]"
        thesis = f"  {row.get('thesis', '')[:70]}" if self.show_thesis and row.get("thesis") else ""
        print(f"{row['decision_date']}  WTI {row['close']:6.2f}  {name:<14} {held:<5}  {stop}  p_up {p}  "
              f"P&L {pnl:+7,.0f}{note}{thesis}", flush=True)

    def day(self, i: int, n: int, t, seconds: float) -> None:
        if i == n or i % 25 == 0:
            eta = seconds / i * (n - i) / 60
            print(f"  ---- day {i}/{n}  {seconds / 60:.1f} min so far, about {eta:.0f} min to go ----", flush=True)
