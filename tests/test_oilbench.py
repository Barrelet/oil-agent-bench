import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oilbench.agents import BuyAndHold, Flat, LLMAgent, RandomTrader, TimesFMAgent, p_up_from_quantiles
from oilbench.data import Market, expiry_dates, roll_adjusted, roll_days, synthetic
from oilbench.engine import Book
from oilbench.metrics import beats, calibration, forecast_scores, random_traders, scoreboard
from oilbench.prompt import MANDATES, build_payload, estimate_tokens, render_text
from oilbench.replay import decision_dates, run
from oilbench.rules import FEE_RATE, INITIAL_CAPITAL, STOP_ATR_MULT, STOP_MAX_PCT
from oilbench.schema import Decision

START, END = "2025-10-01", "2026-09-30"


@pytest.fixture(scope="module")
def md():
    return Market.from_raw(synthetic(n_days=1100, start="2022-06-01", seed=3))


# ---- data ------------------------------------------------------------------------------
def test_wti_expiry_rule_on_known_contracts():
    days = pd.bdate_range("2025-09-01", "2025-12-31")
    exp = expiry_dates(days)
    assert pd.Timestamp("2025-10-21") in exp      # CLX25: the 25th was a Saturday
    assert pd.Timestamp("2025-11-20") in exp      # CLZ25: the 25th was a Tuesday
    rolls = roll_days(days)
    assert pd.Timestamp("2025-10-22") in rolls and pd.Timestamp("2025-11-21") in rolls


def test_roll_day_return_is_zero_and_other_returns_exact():
    raw = synthetic(n_days=300, seed=1, roll_gap=0.05)
    md = Market.from_raw(raw, adjust_rolls=True)
    adj, r_raw = md.bars["close"].pct_change(), raw["close"].pct_change()
    roll = md.bars["is_roll"]
    assert roll.sum() >= 12
    assert np.allclose(adj[roll], 0.0)
    assert np.allclose(adj[~roll].iloc[1:], r_raw[~roll].iloc[1:])


def test_default_is_the_quoted_price():
    raw = synthetic(n_days=300, seed=1, roll_gap=0.05)
    md = Market.from_raw(raw)
    pd.testing.assert_frame_equal(md.bars[["open", "high", "low", "close"]], raw[["open", "high", "low", "close"]])
    assert md.bars["is_roll"].sum() >= 12 and (md.bars["factor"] == 1.0).all()


def test_adjustment_is_point_in_time():
    raw = synthetic(n_days=400, seed=2)
    full = Market.from_raw(raw, adjust_rolls=True).bars
    t = raw.index[250]
    part = Market.from_raw(raw.loc[:t], adjust_rolls=True).bars
    pd.testing.assert_frame_equal(full.loc[:t], part)


# ---- engine ----------------------------------------------------------------------------
def _bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def test_disaster_stop_is_three_atr_from_the_fill_and_gaps_fill_at_open():
    b = Book()
    b.step(Decision("long"), 100.0, pd.Timestamp("2026-01-05"), _bar(101, 102, 100, 101), atr=2.0)
    assert b.stop == pytest.approx(101 - STOP_ATR_MULT * 2.0)          # 95.0, from the fill price
    b.step(Decision("long"), 101.0, pd.Timestamp("2026-01-06"), _bar(90, 92, 89, 91), atr=2.5)
    t = b.trades[-1]
    assert t.reason == "stop" and t.exit_price == 90.0                  # gapped through: filled at the open


def test_disaster_stop_inside_range_fills_at_stop_for_a_short():
    b = Book()
    b.step(Decision("short"), 100.0, pd.Timestamp("2026-01-05"), _bar(100, 101, 99, 100), atr=1.5)
    assert b.stop == pytest.approx(104.5)
    b.step(Decision("short"), 100.0, pd.Timestamp("2026-01-06"), _bar(101, 106, 100, 102), atr=1.5)
    assert b.trades[-1].reason == "stop" and b.trades[-1].exit_price == pytest.approx(104.5)


def test_disaster_stop_is_capped_at_15_percent():
    b = Book()
    b.step(Decision("long"), 100.0, pd.Timestamp("2026-01-05"), _bar(100, 101, 99, 100), atr=8.0)    # 3 ATRs = 24%
    assert b.stop == pytest.approx(85.0)
    s = Book()
    s.step(Decision("short"), 100.0, pd.Timestamp("2026-01-05"), _bar(100, 101, 99, 100), atr=8.0)
    assert s.stop == pytest.approx(115.0)


def test_disaster_stop_never_moves():
    b = Book()
    b.step(Decision("long"), 100.0, pd.Timestamp("2026-01-05"), _bar(100, 101, 99, 100), atr=2.0)
    first = b.stop
    for i, px in enumerate([105, 110, 108]):
        b.step(Decision("long"), px, pd.Timestamp("2026-01-06") + pd.Timedelta(days=i), _bar(px, px + 1, px - 1, px), atr=4.0)
    assert b.stop == first


def test_reversal_is_two_fills_and_fees_add_up():
    assert FEE_RATE == pytest.approx(0.0002)
    b = Book()
    b.step(Decision("long"), 100.0, pd.Timestamp("2026-01-05"), _bar(100, 100, 100, 100), atr=2.0)
    b.step(Decision("short"), 100.0, pd.Timestamp("2026-01-06"), _bar(100, 100, 100, 100), atr=2.0)
    assert b.stop == pytest.approx(106.0)                                # a new stop for the new position
    b.close_all(pd.Timestamp("2026-01-06"), 100.0)
    assert len(b.trades) == 2
    # four fills of roughly the whole account (open long, close long, open short, close short)
    assert b.fees_paid == pytest.approx(4 * FEE_RATE * INITIAL_CAPITAL, rel=0.01)
    assert b.balance(100.0) == pytest.approx(INITIAL_CAPITAL - b.fees_paid)


def test_baselines(md):
    res = run(md, [Flat(), BuyAndHold()], START, END)
    assert res["flat"]["balance"].iloc[-1] == INITIAL_CAPITAL
    dds = decision_dates(md, START, END)
    first_open = md.bar(md.next_date(dds[0]))["open"]
    last_close = md.bar(md.next_date(dds[-1]))["close"]
    expected = INITIAL_CAPITAL / (1 + FEE_RATE) * (last_close / first_open) * (1 - FEE_RATE)
    assert res["buy_and_hold"]["balance"].iloc[-1] == pytest.approx(expected, rel=1e-9)


# ---- look-ahead --------------------------------------------------------------------------
class Recorder(Flat):
    name = "recorder"
    needs_history = True

    def __init__(self):
        self.seen = []

    def decide(self, payload):
        self.seen.append(render_text(payload) + str(payload["_closes"][-5:]))
        return super().decide(payload)


def test_scrambling_the_future_changes_nothing_before_it():
    raw = synthetic(n_days=1100, start="2022-06-01", seed=4)
    cut = pd.Timestamp("2026-03-02")
    scrambled = raw.copy()
    after = scrambled.index > cut
    scrambled.loc[after, ["open", "high", "low", "close"]] *= 1.5
    a, b = Recorder(), Recorder()
    run(Market.from_raw(raw), [a], START, END)
    run(Market.from_raw(scrambled), [b], START, END)
    n = int((decision_dates(Market.from_raw(raw), START, END) <= cut).sum())
    assert a.seen[:n] == b.seen[:n]
    assert a.seen[n:] != b.seen[n:]


# ---- random traders --------------------------------------------------------------------
def test_random_traders_reproducible_and_trade(md):
    r1 = random_traders(md, n=20, batch=10, start=START, end=END)
    r2 = random_traders(md, n=20, batch=10, start=START, end=END)
    pd.testing.assert_frame_equal(r1, r2)
    assert r1["n_trades"].median() > 5
    assert beats(r1["total_return"], r1["total_return"].median()) == pytest.approx(50, abs=5)


# ---- language model ----------------------------------------------------------------------
class MockBackend:
    name = "mock"
    signature = {}

    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, system, user, model, temperature, schema, seed=None, max_tokens=700):
        self.calls += 1
        a = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        return SimpleNamespace(text=a if isinstance(a, str) else json.dumps(a), prompt_tokens=10, completion_tokens=5)


def test_anonymised_payload_hides_dates(md, tmp_path):
    t = decision_dates(md, START, END)[10]
    p = build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"]), anonymise=True)
    assert p["price"] == 100.0 and "decision_date" not in p and "first_date" not in p["daily"]
    agent = LLMAgent("mock", MockBackend([{"reasoning": "x", "p_up": 0.6, "position": "long", "thesis": "t"}]), "m", cache_dir=tmp_path)
    assert agent.decide(p).position == "long"


def test_shown_price_is_the_real_contract_price(tmp_path):
    raw = synthetic(n_days=1100, start="2022-06-01", seed=5, roll_gap=0.03)
    md = Market.from_raw(raw)
    for t in decision_dates(md, START, END)[::25]:
        p = build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"]))
        assert p["price"] == pytest.approx(raw.loc[t, "close"], abs=0.006)
        assert p["_atr"] > 0


def test_llm_invalid_answer_is_repaired_once_then_held(md, tmp_path):
    t = decision_dates(md, START, END)[0]
    p = build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"]))
    good = {"reasoning": "x", "p_up": 0.4, "position": "flat"}
    agent = LLMAgent("mock", MockBackend(["not json", good]), "m", cache_dir=tmp_path)
    assert agent.decide(p).position == "flat" and agent.records[-1].repaired
    agent2 = LLMAgent("mock2", MockBackend(['{"position": "long"}']), "m", cache_dir=tmp_path)
    d = agent2.decide(p)
    assert d.invalid and d.position == "flat"


def test_llm_answers_are_cached(md, tmp_path):
    t = decision_dates(md, START, END)[0]
    p = build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"]))
    be = MockBackend([{"reasoning": "x", "p_up": 0.5, "position": "flat"}])
    LLMAgent("c", be, "m", cache_dir=tmp_path).decide(p)
    LLMAgent("c", be, "m", cache_dir=tmp_path).decide(p)
    assert be.calls == 1


def test_prompt_is_small(md):
    t = decision_dates(md, START, END)[0]
    text = MANDATES["P3"] + render_text(build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"])))
    assert estimate_tokens(text) < 4000


# ---- TimesFM rule --------------------------------------------------------------------------
def _fake_forecaster(shift):
    def f(closes):
        c = closes[-1]
        return {q: c * (1 + shift) + (q - 0.5) * 0.04 * c for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    return f


def test_timesfm_rule(md):
    t = decision_dates(md, START, END)[5]
    base = build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"]))
    base["_closes"] = md.asof(t).bars["close"].to_numpy()
    up = TimesFMAgent(_fake_forecaster(+0.01)).decide(base)
    assert up.position == "long" and up.p_up > 0.5
    down = TimesFMAgent(_fake_forecaster(-0.01)).decide(base)
    assert down.position == "short" and down.p_up < 0.5
    assert TimesFMAgent(_fake_forecaster(0.0001)).decide(base).position == "flat"     # inside the 0.04% cost band
    assert TimesFMAgent(_fake_forecaster(0.0)).band == pytest.approx(2 * FEE_RATE)


def test_p_up_from_quantiles_symmetric():
    q = {k: 100 + (k - 0.5) * 10 for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    assert p_up_from_quantiles(q, 100.0) == pytest.approx(0.5)
    assert p_up_from_quantiles(q, 150.0) == pytest.approx(0.05)


def test_timesfm_runs_through_a_season(md):
    res = run(md, [TimesFMAgent(_fake_forecaster(0.0)), Flat()], START, END)
    assert len(res["timesfm_2p5"]["log"]) == len(decision_dates(md, START, END))


# ---- forecast scoring ------------------------------------------------------------------------
def test_forecast_scores():
    log = pd.DataFrame({"decision_date": range(6), "p_up": [0.9, 0.9, 0.1, 0.1, 0.5, 0.5],
                        "close": [100] * 6, "next_close": [101, 102, 99, 98, 101, 99]})
    s = forecast_scores(log)
    assert s["hit_rate"] == 1.0 and s["brier"] < s["brier_no_view"] and s["brier_skill"] > 0
    assert calibration(log)["days"].sum() == 6


def test_scoreboard(md):
    res = run(md, [Flat(), BuyAndHold(), RandomTrader(1)], START, END)
    sb = scoreboard(res)
    assert set(sb.index) == {"flat", "buy_and_hold", "random_s1"}


# ---- fixes from the 19 Oct code review ------------------------------------------------------
def test_window_ends_inside_the_window(md):
    from oilbench.replay import season_span
    s = season_span(md, START, END)
    assert pd.Timestamp(s["last_mark"]) <= pd.Timestamp(END)
    assert pd.Timestamp(s["first_fill"]) > pd.Timestamp(START) - pd.Timedelta(days=1)


def test_end_after_gives_exact_days(md):
    from oilbench.replay import end_after
    assert len(decision_dates(md, START, end_after(md, 10, START))) == 10


class Broken(Flat):
    name = "broken"

    def __init__(self):
        self.n = 0

    def decide(self, payload):
        self.n += 1
        if self.n > 5:
            raise ConnectionError("model server down")
        return super().decide(payload)


def test_backend_failure_stops_the_run(md):
    with pytest.raises(RuntimeError, match="could not answer"):
        run(md, [Broken()], START, END)


def test_random_traders_file_is_tied_to_its_window(md, tmp_path):
    from oilbench.metrics import random_traders_cached
    from oilbench.replay import end_after
    short = random_traders_cached(md, tmp_path, START, end_after(md, 10, START), n=20)
    full = random_traders_cached(md, tmp_path, START, END, n=20)
    assert short["window"].iloc[0] != full["window"].iloc[0]
    assert full["n_trades"].median() > short["n_trades"].median()
    assert len(list(tmp_path.glob("random_traders_*.csv"))) == 2








def test_every_player_gets_the_same_disaster_stop_except_hold_oil(md):
    from oilbench.agents import RandomTrader
    res = run(md, [BuyAndHold(), RandomTrader(3), TimesFMAgent(_fake_forecaster(0.01))], START, END)
    assert res["buy_and_hold"]["log"]["active_stop"].isna().all()
    for name in ("random_s3", "timesfm_2p5"):
        log = res[name]["log"]
        before = log["held"].shift(fill_value="flat")
        opened = log[(log["held"] != "flat") & (log["held"] != before)]
        assert len(opened) > 0
        for _, r in opened.iterrows():
            fill = md.bar(pd.Timestamp(r["fill_date"]))["open"]
            side = 1 if r["held"] == "long" else -1
            dist = min(STOP_ATR_MULT * r["atr"], STOP_MAX_PCT * fill)
            assert r["active_stop"] == pytest.approx(fill - side * dist)


def test_an_answer_with_a_stop_field_is_still_accepted_and_the_stop_ignored(md, tmp_path):
    t = decision_dates(md, START, END)[3]
    p = build_payload(md.asof(t), Book().snapshot(float(md.bar(t)["close"])))
    a = LLMAgent("s", MockBackend([{"reasoning": "x", "p_up": 0.6, "position": "long", "stop_price": 1.0}]), "m", cache_dir=tmp_path)
    d = a.decide(p)
    assert d.position == "long" and not d.invalid and not hasattr(d, "stop_price")


def test_prompts_carry_the_new_rules():
    for k, m in MANDATES.items():
        assert "0.02%" in m and "disaster stop" in m and "15%" in m and "0.1%" not in m, k
    from oilbench.prompt import OUTPUT_CONTRACT
    assert "stop_price" not in OUTPUT_CONTRACT


def test_forecasts_scored_on_the_same_days():
    log = pd.DataFrame({"decision_date": range(4), "p_up": [0.9, None, 0.2, None],
                        "close": [100] * 4, "next_close": [101, 99, 99, 101]})
    s = forecast_scores(log)
    assert s["n_forecasts"] == 4 and s["n_missing"] == 2


def test_saved_runs_reload_identically(md, tmp_path):
    from oilbench.replay import load_saved, collect_saved
    from oilbench.agents import RandomTrader
    res = run(md, [RandomTrader(7)], START, END, log_dir=tmp_path / "timesfm_2p5")
    back = load_saved(tmp_path / "timesfm_2p5")["random_s7"]
    assert back["balance"].iloc[-1] == pytest.approx(res["random_s7"]["balance"].iloc[-1])
    assert len(back["trades"]) == len(res["random_s7"]["trades"]) and len(back["log"]) == len(res["random_s7"]["log"])
    assert forecast_scores(back["log"]) == pytest.approx(forecast_scores(res["random_s7"]["log"]))
    assert "random_s7" in collect_saved(tmp_path)


# ---- TimesFM: prices or returns, and the choice on the previous year ------------------------------
def test_tfm_series_and_view():
    from oilbench.agents import tfm_series, tfm_view
    closes = np.array([100.0, 101.0, 99.99, 102.0])
    assert list(tfm_series(closes, "price", 2)) == [99.99, 102.0]
    r = tfm_series(closes, "returns", 2)
    assert r == pytest.approx([99.99 / 101 - 1, 102 / 99.99 - 1])
    q = {k: (k - 0.5) * 0.02 + 0.001 for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    move, p_up = tfm_view(q, 102.0, "returns")
    assert move == pytest.approx(0.001) and 0.5 < p_up < 0.6


def test_timesfm_agent_on_returns(md):
    t = decision_dates(md, START, END)[5]
    base = build_payload(md.asof(t), Book().snapshot(md.bar(t)["close"]))
    base["_closes"] = md.asof(t).bars["close"].to_numpy()
    up = lambda s: {k: 0.003 + (k - 0.5) * 0.02 for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    a = TimesFMAgent(up, context=63, kind="returns")
    d = a.decide(base)
    assert d.position == "long" and d.p_up > 0.5 and a.log[-1]["expected_move"] == pytest.approx(0.003)


class _Fake:
    def __init__(self):
        self.lengths = []

    def __call__(self, series):
        return self.many([series])[0]

    def many(self, series):
        self.lengths += [len(s) for s in series]
        out = []
        for s in series:
            centre, spread = (s[-1], 0.02 * s[-1]) if abs(s[-1]) > 1 else (0.0, 0.02)   # prices: unchanged; returns: zero
            out.append({k: centre + (k - 0.5) * spread for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})
        return out


def test_dev_forecasts_use_only_the_past_and_the_rule_falls_back_to_default():
    from oilbench.tfm_select import dev_forecasts, score_versions, choose, DEFAULT
    raw = synthetic(n_days=1300, start="2020-06-01", seed=9)
    md = Market.from_raw(raw)
    f = dev_forecasts(md, _Fake(), "2024-10-01", "2025-01-31", contexts={"3 months": 63, "1 year": 252})
    scr = raw.copy(); scr.loc[scr.index > pd.Timestamp("2024-12-01"), ["open", "high", "low", "close"]] *= 2
    g = dev_forecasts(Market.from_raw(scr), _Fake(), "2024-10-01", "2025-01-31", contexts={"3 months": 63, "1 year": 252})
    early = f["decision_date"] < "2024-12-01"
    pd.testing.assert_series_equal(f.loc[early, "expected_move"], g.loc[early, "expected_move"])
    s = score_versions(f)
    assert len(s) == 4 and s["brier_skill"].abs().max() < 0.05
    kind, ctx, why = choose(s)                     # a no-skill forecaster: the default is kept
    assert (kind, ctx) == DEFAULT and "default kept" in why


def test_rule_picks_the_best_version_when_it_clears_the_bar():
    from oilbench.tfm_select import choose
    s = pd.DataFrame({"brier_skill": [0.002, 0.03, -0.01]},
                     index=pd.MultiIndex.from_tuples([("price", "4 years"), ("returns", "6 months"), ("price", "3 months")],
                                                     names=["input", "history"]))
    assert choose(s)[:2] == ("returns", 126)


def test_timesfm_precompute_gives_the_same_decisions(md):
    a, b = TimesFMAgent(_Fake(), context=63), TimesFMAgent(_Fake(), context=63)
    b.precompute(md, START, END)
    ra, rb = run(md, [a], START, END), run(md, [b], START, END)
    pd.testing.assert_frame_equal(ra["timesfm_2p5"]["log"], rb["timesfm_2p5"]["log"])
