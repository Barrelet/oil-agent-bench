"""Plain-language pictures for the story notebook, in September's style.

The notebook reads only `data/summary/` (small CSVs, safe to publish: results, not
prices), so it runs in seconds and the story does not move when experiments rerun.
`export_summary()` writes that folder from the runs; `demo_summary()` writes one from
synthetic prices so the notebook can be built and checked before the real runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import beats, pnl_metrics, forecast_frame, forecast_scores, trade_metrics

# display names; categorical colours as in September, everything that is not a model is grey
LABELS = {"qwen3-8b_P1": "Qwen3 8B, Prompt 1", "qwen3-8b_P2": "Qwen3 8B, Prompt 2", "qwen3-8b_P3": "Qwen3 8B, Prompt 3",
          "gpt-5-1_P1": "GPT-5.1, Prompt 1", "gpt-5-1_P2": "GPT-5.1, Prompt 2", "gpt-5-1_P3": "GPT-5.1, Prompt 3",
          "timesfm_2p5": "TimesFM 2.5", "flat": "Do nothing (cash)", "buy_and_hold": "Hold oil all year"}
COLOR = {"Qwen3 8B, Prompt 1": "#2a78d6", "Qwen3 8B, Prompt 2": "#eb6834", "Qwen3 8B, Prompt 3": "#1baf7a",
         "GPT-5.1, Prompt 1": "#163f7a", "GPT-5.1, Prompt 2": "#9c3c14", "GPT-5.1, Prompt 3": "#0d6b48",
         "TimesFM 2.5": "#7b4fd6", "Do nothing (cash)": "#8a8987", "Hold oil all year": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 12, "figure.dpi": 110})


# ---- the summary folder ------------------------------------------------------------------------
def export_summary(results: dict[str, dict], random: pd.DataFrame, out: Path | str, meta: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    eq, tr, fc, rows = [], [], [], []
    for k, r in results.items():
        name = LABELS.get(k, k)
        e = r["balance"].rename("balance").rename_axis("date").reset_index().assign(player=name)
        eq.append(e)
        if not r["trades"].empty:
            tr.append(r["trades"].assign(player=name))
        f = forecast_frame(r["log"]).drop(columns=["close", "next_close"]).assign(player=name)   # no prices in the published summary
        fc.append(f)
        m = {**pnl_metrics(r["balance"]), **trade_metrics(r["trades"], r["log"]), **forecast_scores(r["log"])}
        m["beats_random"] = beats(random["total_return"], m["total_return"])
        rows.append({"key": k, "player": name, **m})
    pd.concat(eq).to_csv(out / "balance.csv", index=False)
    (pd.concat(tr) if tr else pd.DataFrame()).to_csv(out / "trades.csv", index=False)
    pd.concat(fc).to_csv(out / "forecasts.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "players.csv", index=False)
    random.to_csv(out / "random_traders.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def demo_summary(out: Path | str) -> None:
    """A full fake season on synthetic prices, with stand-in players, to build the notebook."""
    from .agents import BuyAndHold, Flat, RandomTrader, TimesFMAgent
    from .data import Market, synthetic
    from .metrics import random_traders
    from .replay import decision_dates, run

    md = Market.from_raw(synthetic(n_days=1150, start="2022-06-01", seed=11))

    def momentum_forecaster(closes):        # a stand-in for TimesFM: extrapolates 20-day drift
        c = closes[-1]
        drift = (closes[-1] / closes[-21] - 1) / 20
        return {q: c * (1 + drift) + (q - 0.5) * 0.05 * c for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}

    class Stand(RandomTrader):              # stand-ins for the LLM players, clearly fake
        def __init__(self, seed, key):
            super().__init__(seed=seed, p_open=0.15, p_close=0.04)
            self.name = key

    players = [Flat(), BuyAndHold(), Stand(101, "qwen3-8b_P1"), Stand(102, "qwen3-8b_P2"), Stand(103, "qwen3-8b_P3"),
               TimesFMAgent(momentum_forecaster)]
    from .replay import season_span
    res = run(md, players)
    null = random_traders(md, n=1000)
    export_summary(res, null, out, {"demo": True, **season_span(md)})


@dataclass
class Summary:
    balance: pd.DataFrame
    trades: pd.DataFrame
    forecasts: pd.DataFrame
    players: pd.DataFrame
    random: pd.DataFrame
    meta: dict

    @property
    def window(self) -> str:
        m = self.meta
        return (f"first trade {pd.Timestamp(m['first_fill']).strftime('%-d %B %Y')}, last mark "
                f"{pd.Timestamp(m['last_mark']).strftime('%-d %B %Y')}, {m['days']} trading decisions")

    def value(self, player: str, col: str):
        p = self.players.set_index("player")
        return p.loc[player, col]


def load_summary(root: Path | str, folder: str = "summary") -> Summary:
    """Reads data/<folder>/ (data/summary for the real season, data/summary_quick for a quick look)."""
    d = Path(root) / "data" / folder
    trades = pd.read_csv(d / "trades.csv") if (d / "trades.csv").stat().st_size > 2 else pd.DataFrame()
    return Summary(pd.read_csv(d / "balance.csv", parse_dates=["date"]), trades, pd.read_csv(d / "forecasts.csv"),
                   pd.read_csv(d / "players.csv"), pd.read_csv(d / "random_traders.csv"), json.loads((d / "meta.json").read_text()))


# ---- pictures ------------------------------------------------------------------------------------
def _tidy(ax, ylabel=""):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so end-of-line labels do not overlap (order kept)."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        out[i] = max(out[i], out[j] + gap)
    return list(out)


def pnl_chart(S: Summary, players: list[str], title: str = "Cumulative P&L, USD (each account starts with $10,000)"):
    """Cumulative P&L of each account through the year, net of fees."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ends = []
    for p in players:
        e = S.balance[S.balance["player"] == p]
        pnl = e["balance"] - 10_000.0
        grey = p in ("Do nothing (cash)", "Hold oil all year")
        ax.plot(e["date"], pnl, color=COLOR.get(p, INK2), lw=1.4 if grey else 2.0, ls="--" if grey else "-")
        ends.append((p, e["date"].iloc[-1], pnl.iloc[-1]))
    ax.axhline(0, color=INK2, lw=0.8)
    lo, hi = ax.get_ylim()
    ys = _spread([y for _, _, y in ends], gap=(hi - lo) * 0.055)
    for (p, x, y), yl in zip(ends, ys):
        ax.annotate(f" {p}  {y:+,.0f} ({y / 1e4:+.1%})", (x, y), xytext=(x, yl), color=COLOR.get(p, INK2), fontsize=9, va="center",
                    annotation_clip=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+,.0f}"))
    _tidy(ax, "P&L, USD")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def luck_chart(S: Summary, players: list[str]):
    """Where each player lands among the 1,000 random traders."""
    r = S.random["total_return"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(r, bins=40, color=GRID, edgecolor="white")
    top = ax.get_ylim()[1]
    ranked = sorted(players, key=lambda p: S.value(p, "total_return"))
    ax.set_ylim(0, top * (1.0 + 0.16 * len(players)))           # headroom: one label row per player
    for k, p in enumerate(ranked):
        v = S.value(p, "total_return")
        ax.axvline(v, color=COLOR.get(p, INK2), lw=2, ymax=(top * (1.08 + 0.16 * k)) / ax.get_ylim()[1])
        ax.annotate(f"{p}: beats {beats(r, v):.0f} in 100", (v, top * (1.1 + 0.16 * k)), color=COLOR.get(p, INK2), fontsize=8.5,
                    ha="left", va="center", xytext=(4, 0), textcoords="offset points")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
    _tidy(ax, "random traders")
    ax.set_title("1,000 random traders, same rules, same year: where the players land", loc="left")
    fig.tight_layout()
    return fig


def trades_chart(S: Summary, player: str):
    """Every trade of one player, in order, with its profit or loss."""
    t = S.trades[S.trades["player"] == player].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(t.index + 1, t["pnl"], color=np.where(t["pnl"] > 0, COLOR.get(player, INK), "#c9c8c4"))
    ax.axhline(0, color=INK2, lw=0.8)
    _tidy(ax, "profit or loss, USD")
    ax.set_xlabel("trade number")
    ax.set_title(f"{player}: every trade, {int((t['pnl'] > 0).sum())} of {len(t)} made money", loc="left")
    fig.tight_layout()
    return fig


def calibration_chart(S: Summary, players: list[str], edges=(0, 0.35, 0.45, 0.55, 0.65, 1.0001)):
    """Stated probability of a rise against how often oil actually rose."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2)
    for p in players:
        f = S.forecasts[S.forecasts["player"] == p].copy()
        f["bucket"] = pd.cut(f["p_up"], edges, right=False)
        g = f.groupby("bucket", observed=True).agg(n=("up", "size"), stated=("p_up", "mean"), realised=("up", "mean"))
        ax.scatter(g["stated"], g["realised"], s=12 + g["n"] * 1.5, color=COLOR.get(p, INK2), alpha=0.85, label=p)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("stated chance that oil closes higher tomorrow")
    ax.set_ylabel("share of days oil did close higher")
    _tidy(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Did the stated chances mean anything?", loc="left")
    fig.tight_layout()
    return fig


def scoreboard(S: Summary, players: list[str]) -> pd.DataFrame:
    p = S.players.set_index("player").loc[players]
    out = pd.DataFrame({
        "P&L, USD": p["total_return"].map(lambda x: f"{x * 10_000:+,.0f}"),
        "Return on $10,000": p["total_return"].map(lambda x: f"{x:+.1%}"),
        "Beats random traders (in 100)": p["beats_random"].round(0).astype(int),
        "Trades": p["n_trades"].astype(int),
        "Winning trades": p["win_rate"].map(lambda x: "-" if pd.isna(x) else f"{x:.0%}"),
        "Stopped out": p["stopped_out"].astype(int),
        "Days long / short / flat": [f"{int(a)} / {int(b)} / {int(c)}" for a, b, c in zip(p["days_long"], p["days_short"], p["days_flat"])],
    })
    return out
