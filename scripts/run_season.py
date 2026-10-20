"""Run the oil season on your Mac. Every answer is cached, so a stopped run resumes.

    # 0. once: prices (consumed from Yahoo, never committed) and a ten-day timing test
    python scripts/run_season.py --players baselines --download
    python scripts/run_season.py --players qwen --prompts P1 --days 10

    # post 1: the small open model (Ollama running, `ollama pull qwen3:8b` done)
    python scripts/run_season.py --players baselines,random,qwen

    # post 2: GPT-5.1 (OPENAI_API_KEY in local_settings.py, which is gitignored)
    python scripts/run_season.py --players gpt

    # post 3: TimesFM 2.5 (pip install "timesfm[torch]==3.0.2")
    python scripts/run_season.py --players timesfm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from oilbench.agents import BuyAndHold, Flat, LLMAgent, TimesFMAgent, load_timesfm  # noqa: E402
from oilbench.backends import OllamaBackend, OpenAICompatibleBackend, load_env_file  # noqa: E402
from oilbench.data import Market, download_yahoo, load_csv  # noqa: E402
from oilbench.metrics import forecast_scores, random_traders_cached, scoreboard  # noqa: E402
from oilbench.replay import decision_dates, end_after, run, season_span  # noqa: E402
from oilbench.rules import ADJUST_ROLLS, WINDOW_END, WINDOW_START  # noqa: E402

DATA = ROOT / "data"
PRICES = DATA / "wti_daily.csv"          # gitignored: Yahoo's prices are consumed, not republished
RESULTS = DATA / "results"
CACHE = DATA / "llm_cache"

LOCAL_MODEL = "qwen3:8b"
HOSTED_MODEL = "gpt-5.1"                 # deprecated by OpenAI, shutdown 1 Apr 2027: run before then
TIMESFM_REVISION = None                  # pin to a Hugging Face commit hash before the real run


def bar(i: int, n: int, note: str = "", width: int = 30) -> None:
    """One-line progress bar, redrawn in place: [#########.....]  40/251  note"""
    filled = int(width * i / n) if n else width
    sys.stdout.write(f"\r  [{'#' * filled}{'.' * (width - filled)}] {i:4d}/{n}  {note}")
    if i == n:
        sys.stdout.write("\n")
    sys.stdout.flush()


def players(names: list[str], prompts: list[str]):
    out = []
    if "baselines" in names:
        out += [Flat(), BuyAndHold()]
    if "qwen" in names:
        be = OllamaBackend(think=False)
        out += [LLMAgent(f"qwen3-8b_{p}", be, LOCAL_MODEL, prompt=p, cache_dir=CACHE) for p in prompts]
    if "gpt" in names:
        load_env_file(ROOT)
        be = OpenAICompatibleBackend("https://api.openai.com/v1", "OPENAI_API_KEY", reasoning_effort="low")   # as in September
        out += [LLMAgent(f"gpt-5-1_{p}", be, HOSTED_MODEL, prompt=p, cache_dir=CACHE) for p in prompts]
    if "timesfm" in names:
        out += [TimesFMAgent(load_timesfm(revision=TIMESFM_REVISION))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="baselines")
    ap.add_argument("--prompts", default="P1,P2,P3")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--days", type=int, default=None, help="only the first N days (timing test)")
    ap.add_argument("--anonymise", action="store_true")
    ap.add_argument("--start", default=WINDOW_START)
    ap.add_argument("--end", default=WINDOW_END)
    a = ap.parse_args()

    if a.download or not PRICES.exists():
        DATA.mkdir(exist_ok=True)
        download_yahoo(start="2015-01-01").reset_index().to_csv(PRICES, index=False)
    md = Market.from_raw(load_csv(PRICES), adjust_rolls=ADJUST_ROLLS)
    end = end_after(md, a.days, a.start) if a.days else a.end
    results_dir = RESULTS if not a.days else DATA / "results_quick"      # a quick test never overwrites the year
    span = season_span(md, a.start, end)
    print(f"{'QUICK TEST: ' if a.days else ''}{span['days']} decisions, first fill {span['first_fill']}, last mark {span['last_mark']}")
    names = a.players.split(",")
    tag = "_anon" if a.anonymise else ""

    agents = players(names, a.prompts.split(","))
    if agents:
        t0 = time.time()

        def progress(i, n, t, s):
            bar(i, n, f"{t.date()}  {s / 60:5.1f} min  eta {s / i * (n - i) / 60:5.1f} min")

        n_days = len(decision_dates(md, a.start, end))
        print(f"Playing {n_days} days with {len(agents)} player(s): {', '.join(x.name for x in agents)}")
        if any(isinstance(x, LLMAgent) for x in agents):
            print("  (the first answer is slow: the model is loading into memory)")
        bar(0, n_days, "starting")
        res = run(md, agents, a.start, end, anonymise=a.anonymise, on_day=progress,
                  log_dir=results_dir / f"{a.players.replace(',', '+')}{tag}")
        sb = scoreboard(res)
        for name, r in res.items():
            for k, v in forecast_scores(r["log"]).items():
                sb.loc[name, k] = v
        print(sb.round(3).to_string())
        for ag in agents:
            if isinstance(ag, LLMAgent) and ag.records:
                live = [r.latency_s for r in ag.records if not r.cached]
                if live:
                    print(f"{ag.name}: {sum(live) / len(live):.1f} s per live answer")
        print(f"done in {(time.time() - t0) / 60:.1f} min")

    if "random" in names:
        print("1,000 random traders")
        # saved under a file name that carries the window, so a short test can never stand in for the year
        null = random_traders_cached(md, RESULTS, a.start, end, on_batch=lambda done, n, s: bar(done, n, f"{s:5.0f} s"))
        print(null[["total_return", "sharpe", "n_trades"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
