"""The players: baselines, random traders, language models and TimesFM.

Every agent implements `decide(payload) -> Decision`. Agents that need the long
price history (TimesFM) set `needs_history = True` and receive payload["_closes"],
the closes up to and including the decision date, never later.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .prompt import MANDATES, render_text
from .schema import Decision, json_schema, parse_llm


class Agent:
    name = "agent"
    needs_history = False
    exempt_from_stop = False     # only the hold-oil benchmark has no disaster stop

    def reset(self) -> None:
        pass

    def decide(self, payload: dict) -> Decision:  # pragma: no cover
        raise NotImplementedError


def _current(payload: dict) -> str:
    return payload["position"]["position"]


# ---- baselines ----------------------------------------------------------------------------
class Flat(Agent):
    """Never trades. The 'do nothing' line: exactly $10,000 at the end."""
    name = "flat"

    def decide(self, payload):
        return Decision("flat", p_up=0.5)


class BuyAndHold(Agent):
    """Long from the first open to the last close, no stop: what holding oil made.
    The one player without the disaster stop (a benchmark, not a trader)."""
    name = "buy_and_hold"
    exempt_from_stop = True

    def decide(self, payload):
        return Decision("long", p_up=0.5)


class RandomTrader(Agent):
    """September's coin-flipper, on one market. When flat: 50% chance a day to open,
    long or short with equal odds. When in a trade: 10% chance a day to close it. Its
    trades carry the game's disaster stop like everyone else's. Its p_up is a random
    number. Seeded, so the 1,000 runs are reproducible."""

    def __init__(self, seed: int = 0, p_open: float = 0.5, p_close: float = 0.1):
        self.seed, self.p_open, self.p_close = seed, p_open, p_close
        self.name = f"random_s{seed}"
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def decide(self, payload):
        cur, c = _current(payload), payload["_close"]
        p_up = float(self.rng.uniform())
        if cur == "flat":
            if self.rng.random() < self.p_open:
                side = "long" if self.rng.random() < 0.5 else "short"
                return Decision(side, p_up=p_up)
            return Decision("flat", p_up=p_up)
        if self.rng.random() < self.p_close:
            return Decision("flat", p_up=p_up)
        return Decision(cur, p_up=p_up)


# ---- language models ------------------------------------------------------------------------
@dataclass
class CallRecord:
    date: str
    latency_s: float
    cached: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    repaired: bool
    invalid: str


class LLMAgent(Agent):
    """A language model playing the game. `backend` has September's interface:
    complete(system, user, model, temperature, schema, seed, max_tokens) -> object with
    .text, .prompt_tokens, .completion_tokens, .latency_s. September's OllamaBackend and
    OpenAICompatibleBackend plug in unchanged. Every answer is cached on disk by a hash
    of everything that could change it, so a rerun replays the season exactly."""

    def __init__(self, name: str, backend, model: str, prompt: str = "P3", temperature: float = 0.0,
                 seed: int | None = 7, max_tokens: int = 700, cache_dir: Path | str = "data/llm_cache"):
        self.name, self.backend, self.model, self.prompt = name, backend, model, prompt
        self.temperature, self.seed, self.max_tokens = temperature, seed, max_tokens
        self.system = MANDATES[prompt]
        self.schema = json_schema()
        self.cache_dir = Path(cache_dir) / name
        self.records: list[CallRecord] = []

    def _key(self, user: str) -> str:
        sig = getattr(self.backend, "signature", {})
        blob = json.dumps([getattr(self.backend, "name", "?"), sig, self.model, self.temperature, self.seed,
                           self.max_tokens, self.system, user], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _complete(self, user: str) -> tuple[str, dict]:
        key = self._key(user)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            d = json.loads(path.read_text())
            return d["text"], {**d["meta"], "cached": True}
        t0 = time.time()
        r = self.backend.complete(self.system, user, self.model, self.temperature, self.schema, self.seed, self.max_tokens)
        meta = {"latency_s": round(time.time() - t0, 2), "prompt_tokens": getattr(r, "prompt_tokens", None),
                "completion_tokens": getattr(r, "completion_tokens", None), "cached": False}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model, "system": self.system, "user": user, "text": r.text, "meta": meta}))
        return r.text, meta

    def _parse(self, text: str, payload: dict):
        return parse_llm(_strip_fences(text))

    def decide(self, payload):
        """Backend failures (Ollama down, network) are NOT caught here: they stop the run,
        and running again resumes from the saved answers. Only an answer the model actually
        gave that cannot be used becomes a hold, after one repair attempt."""
        user = render_text(payload)
        text, meta = self._complete(user)
        ans, err = self._parse(text, payload)
        repaired = False
        if ans is None:   # one repair attempt, as in September
            repaired = True
            text, meta2 = self._complete(user + f"\n\nYour previous answer was invalid: {err[:300]}\nReturn corrected JSON only.")
            meta = {**meta2, "latency_s": meta["latency_s"] + meta2["latency_s"]}
            ans, err = self._parse(text, payload)
        if ans is None:
            d = Decision.hold(_current(payload), f"invalid answer: {err[:200]}")
        else:
            d = Decision(ans.position, p_up=ans.p_up, thesis=ans.thesis, reasoning=ans.reasoning)
        self.records.append(CallRecord(payload["_date"], meta["latency_s"], meta["cached"], meta.get("prompt_tokens"),
                                       meta.get("completion_tokens"), repaired, d.invalid))
        return d


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


# ---- TimesFM ------------------------------------------------------------------------------------
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def p_up_from_quantiles(q: dict[float, float], close: float, clip: float = 0.05) -> float:
    """P(next close > today's close) read off the forecast quantiles by linear
    interpolation of the forecast's distribution at today's close. Outside the
    10th-90th range it is clipped to [0.05, 0.95]: the model gives no finer detail."""
    levels = np.array(sorted(q))
    values = np.array([q[k] for k in levels])
    cdf_at_close = float(np.interp(close, values, levels, left=clip, right=1 - clip))
    return float(np.clip(1.0 - cdf_at_close, clip, 1 - clip))


def tfm_series(closes: np.ndarray, kind: str, context: int) -> np.ndarray:
    """What TimesFM is shown: the last `context` closes ("price") or the last `context`
    daily returns, close to close ("returns")."""
    closes = np.asarray(closes, dtype=float)
    if kind == "price":
        return closes[-context:]
    if kind == "returns":
        return (closes[1:] / closes[:-1] - 1.0)[-context:]
    raise ValueError(f"unknown input kind {kind!r}")


def tfm_view(q: dict, close: float, kind: str) -> tuple[float, float]:
    """(expected move as a return, chance of a rise) from a forecast of the next value."""
    if kind == "price":
        return q[0.5] / close - 1.0, p_up_from_quantiles(q, close)
    return q[0.5], p_up_from_quantiles(q, 0.0)


class TimesFMAgent(Agent):
    """TimesFM 2.5 plays through a fixed rule:
      show it the last `context` days of either prices or daily returns (`kind`);
      expected move = its median forecast for the next close, as a return;
      long if it exceeds +band, short if below -band, otherwise flat
      (band = the round-trip cost, 2 x rules.FEE_RATE = 0.04%: a smaller expected move cannot pay for the trade);
      the game's disaster stop, like every other player;
      p_up read off the forecast quantiles (share of the forecast above today's close, or above a 0 return).
    The defaults (prices, 1,024 days) follow Google's example; notebook 04 chooses `kind` and
    `context` on the previous year (Oct 2024 to Sep 2025) by a rule fixed in advance.
    `forecaster(series) -> dict[quantile, value]` is injected, so the rule is testable without the model."""
    name = "timesfm_2p5"
    needs_history = True

    def __init__(self, forecaster: Callable[[np.ndarray], dict], context: int = 1024, band: float | None = None,
                 name: str | None = None, kind: str = "price"):
        from .rules import FEE_RATE
        band = 2 * FEE_RATE if band is None else band
        self.forecaster, self.context, self.band, self.kind = forecaster, context, band, kind
        if name:
            self.name = name
        self.log: list[dict] = []

    def precompute(self, md, start, end) -> None:
        """Forecast every decision day in batches (much faster than one call a day). Each day's
        forecast still uses only the prices up to that day: the same look-ahead guard as the game."""
        from .replay import decision_dates
        dds = decision_dates(md, start, end)
        series = [tfm_series(md.bars["close"].loc[:t].to_numpy(), self.kind, self.context) for t in dds]
        qs = self.forecaster.many(series) if hasattr(self.forecaster, "many") else [self.forecaster(s) for s in series]
        self._cache = {str(t.date()): q for t, q in zip(dds, qs)}

    def decide(self, payload):
        c = payload["_close"]
        cache = getattr(self, "_cache", {})
        q = cache.get(payload["_date"]) or self.forecaster(tfm_series(payload["_closes"], self.kind, self.context))
        move, p_up = tfm_view(q, c, self.kind)
        side = "long" if move > self.band else "short" if move < -self.band else "flat"
        lo, hi = (q[0.1], q[0.9]) if self.kind == "price" else (c * (1 + q[0.1]), c * (1 + q[0.9]))
        self.log.append({"date": payload["_date"], "close": c, "expected_move": move, "q10_price": lo, "q90_price": hi,
                         "p_up": p_up, "position": side})
        thesis = (f"forecast {c * (1 + move):.2f} ({move:+.2%}), 10-90% range {lo:.2f}-{hi:.2f}; "
                  f"{'above' if move > self.band else 'below' if move < -self.band else 'inside'} the {self.band:.2%} band")
        return Decision(side, p_up=p_up, thesis=thesis)


class TimesFMForecaster:
    """Real TimesFM 2.5 (runs on your Mac; weights download from Hugging Face at run time and are
    never committed). Call it with one series, or use `.many(list_of_series)` for a batch.
    Install: pip install "timesfm[torch]==3.0.2" (the PyPI package carries the 2.5 model).
    Series shorter than `max_context` are padded and masked by TimesFM itself, so one compiled
    model serves every history length up to `max_context`."""

    def __init__(self, checkpoint: str = "google/timesfm-2.5-200m-pytorch", revision: str | None = None,
                 max_context: int = 2560, batch: int = 32):
        import timesfm
        kw = {"revision": revision} if revision else {}
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint, **kw)
        self.model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=128, normalize_inputs=True,
                                                  per_core_batch_size=batch, use_continuous_quantile_head=True,
                                                  force_flip_invariance=True, infer_is_positive=True,
                                                  fix_quantile_crossing=True))
        self.max_context = max_context

    @staticmethod
    def _row(row) -> dict:
        return {lvl: float(row[i + 1]) for i, lvl in enumerate(QUANTILES)}   # row = [mean, q10, ..., q90]

    def many(self, series: list[np.ndarray]) -> list[dict]:
        _, qf = self.model.forecast(horizon=1, inputs=[np.asarray(s, dtype=np.float32) for s in series])
        return [self._row(qf[i, 0]) for i in range(len(series))]

    def __call__(self, series: np.ndarray) -> dict:
        return self.many([series])[0]


def load_timesfm(checkpoint: str = "google/timesfm-2.5-200m-pytorch", revision: str | None = None,
                 max_context: int = 2560) -> TimesFMForecaster:
    return TimesFMForecaster(checkpoint, revision, max_context)


def timesfm_version() -> str:
    """The installed TimesFM package version (the package has no __version__ attribute)."""
    try:
        from importlib.metadata import version
        return version("timesfm")
    except Exception:
        return "unknown"
