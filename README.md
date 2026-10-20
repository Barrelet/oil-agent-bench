# oil-agent-bench

AI models trade one year of oil. The [agent-alpha-bench](https://github.com/Barrelet/agent-alpha-bench) game from September, narrowed to one market: front-month WTI crude oil, one decision a day, long, short or flat with the whole account, a disaster stop set by the game, 0.02% per fill. The idea comes from public AI trading leaderboards such as Alpha Arena and TradeRank, applied to oil futures and replayed over a past year so the result is known in full.

Players: Qwen3 8B (a small open model on a laptop), GPT-5.1 (through OpenAI's API), and TimesFM 2.5 (Google Research's time-series foundation model). Each language model plays three times, once with each of three prompts. Every player is measured against doing nothing, holding oil, and 1,000 random traders playing the same rules. Each player also states a daily probability that oil closes higher, scored on its own.

Test year: 1 October 2025 to 30 September 2026, after every model's training cutoff. Not investment advice.

## Results

| Player | Return | Beats random traders (in 100) | Trades | Brier skill of its daily forecast |
|---|---|---|---|---|
| Qwen3 8B, Prompt 1 (basic) | −7.3% | 56 | 109 | −0.070 |
| Qwen3 8B, Prompt 2 (plus risk rules) | −35.7% | 27 | 97 | −0.093 |
| Qwen3 8B, Prompt 3 (structured) | +11.4% | 70 | 30 | −0.036 |
| GPT-5.1, Prompt 1 | +14.1% | 72 | 21 | +0.010 |
| GPT-5.1, Prompt 2 | +35.2% | 82 | 11 | −0.001 |
| GPT-5.1, Prompt 3 | −0.1% | 61 | 26 | −0.012 |
| TimesFM 2.5 | −34.5% | 28 | 52 | −0.047 |
| Do nothing | 0.0% | 61 | 0 | |
| Hold oil all year | +46.3% | 86 | 1 | |

Brier skill above zero means the stated chances beat always saying 50%.

What the year showed, in four lines:
1. **The prompt moved the result more than the model.** Same small model, same prices, three prompts: a 47-point spread. The best prompt for one model was the worst for the other.
2. **The bigger model's edge was patience.** GPT-5.1 traded a fraction as often as the small model and sat in cash nearly half the time on its best run.
3. **No player could call the next day.** Every stated chance of a rise was a coin or worse; GPT-5.1 with Prompt 1 came out a hair above.
4. **Two or three trades decided every year.** Every profitable run was long through the March 2026 rally.

TimesFM, shown prices with four years of history, forecast a return to the long-run average and was short on 175 of 250 days. Twelve input and history-length versions were scored on the previous year before the test year, by a rule fixed in advance; none had meaningful forecasting skill, so Google's default stayed. The best of the twelve, played once as a check, lost 61%.

## How an AI player decides

Each day the model reads one of the three prompts and that day's data, and writes its answer: a position (long, short or flat), its chance that oil closes higher tomorrow (`p_up`), a one-line thesis and a short reasoning. It never sets a stop: the game places the same disaster stop on every position (three ATRs from entry, at most 15%). Nothing in the code makes the choice for it. The decision is the model's own judgement, shaped by the prompt:

- **P1, the basic brief**: the rules only, no method. The model decides however it sees fit.
- **P2, plus risk rules**: the cost of changing position, flat when there is no clear view, 0.5 means no view; still no entry method.
- **P3, a structured brief**: read trend (20 and 60-day returns, distance to the 20 and 50-day averages), momentum (RSI, 5-day return) and location (distance to the 20-day high and low), weigh them yourself, and take a position when, on balance, the expected move beats the 0.04% round-trip cost; flat only with no view; keep a position while its reason holds.

**No threshold on `p_up`.** No level of `p_up` triggers a long or a short. The position and the chance of a rise are two separate answers, scored separately. This is deliberate: in September the chance was tied to a rule (no trade below 0.80) and both models simply wrote the smallest number the rule accepted.

**Every day is a new decision.** When a trade is open, the model is asked again each day, seeing its position, entry, the game's stop and unrealised P&L. Answering the same position keeps the trade; flat closes it at the next open; the opposite side reverses it. The disaster stop can also close it during any session. The stop protects the entry, not the profits: taking profits is the model's decision.

**The reasoning is a window, not proof.** The written reasoning shows what the model says it checked. A language model's explanation does not always match how it reached its answer.

TimesFM is different: it forecasts the next close, and a fixed rule written in advance turns the forecast into a position (long if the median forecast is more than 0.04% above today's close, short if more than 0.04% below, otherwise flat).

## Notebooks, in order

| Notebook | What it does | Time |
|---|---|---|
| `01_data_and_game.ipynb` | Prices, rules, one trade by hand, do nothing, hold oil, 1,000 random traders, look-ahead checks. No AI calls | about 1 min |
| `02_small_model.ipynb` | Qwen3 8B on the laptop plays the year with P1, P2, P3, one cell per prompt; one line per day; results | about 2.5 h |
| `03_large_model.ipynb` | GPT-5.1 plays the same year with the same prompts; a 10-day look with projected cost before the full year | about 1 h, about $4 |
| `04_timesfm.ipynb` | TimesFM 2.5: chooses prices or returns and the history length on the previous year (12 versions, rule fixed in advance), plays the test year, checks the best version too | 10 to 20 min |
| `oil_story.ipynb` | The plain-language story, every chart drawn from `data/summary/`. No model calls | seconds |

Each notebook is generated by `scripts/make_notebook_0N.py`, as in September; the executed notebooks in `notebooks/` carry the outputs of the runs reported above.

## Code

| File | What it does |
|---|---|
| `oilbench/rules.py` | The rules in one place |
| `oilbench/data.py` | Prices, contract roll dates, roll-adjusted series (point-in-time) |
| `oilbench/engine.py` | The paper account: fills, the disaster stop, fees, marks |
| `oilbench/prompt.py` | What the players see; the three prompts |
| `oilbench/schema.py` | The daily answer format |
| `oilbench/agents.py` | Baselines, random traders, LLM player, TimesFM player |
| `oilbench/backends.py` | Ollama and OpenAI backends, unchanged from September |
| `oilbench/replay.py` | Walks the year day by day |
| `oilbench/metrics.py` | Returns, trades, random-trader percentile, forecast scores |
| `oilbench/report.py` | Charts and tables for the story notebook |
| `oilbench/live.py` | The one-line-per-day view while a season plays |
| `oilbench/tfm_select.py` | The 12 TimesFM versions and the rule that chooses one on the previous year |

## Running it

```bash
pip install -e ".[dev]"            # add ".[timesfm]" for notebook 04
pytest                             # 36 checks, about 40 seconds
```

Notebook 02 needs [Ollama](https://ollama.com) with `qwen3:8b` pulled. Notebook 03 needs an OpenAI key in `local_settings.py` (`OPENAI_API_KEY = "sk-..."`), which is gitignored; GPT-5.1 is deprecated by OpenAI with shutdown on 1 April 2027, and because OpenAI fixes its temperature a fresh run can differ from the one reported here. Every model answer is cached on disk, so a run can be stopped and resumed, and the reported runs replay exactly from the cache.

Prices are the front-month WTI future as quoted (Yahoo CL=F), downloaded at run time with no adjustment for contract rolls (see `oilbench/data.py`); they are not redistributed here. TimesFM 2.5 (Google Research, Apache-2.0) is downloaded from Hugging Face at run time; it is not an official Google product and this project is not endorsed by Google. Qwen3 is released under Apache-2.0. Not investment advice.
