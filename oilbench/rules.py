"""The rules of the game, in one place. Every agent plays under exactly these.

Adapted from the September game on US stocks (agent-alpha-bench, TradeRank-style rules):
same account, same costs, same timing, same stop mechanics; one market instead of
fifty, and three positions instead of free sizing.
"""

INITIAL_CAPITAL = 10_000.0      # paper account, USD
FEE_RATE = 0.0002               # 0.02% of notional per fill (each open, each close): about $18 per WTI contract
                                # at $90 oil, i.e. exchange and broker fees plus about one tick of slippage.
                                # (September's stock game used 0.1%; decision of 19 Oct 2026 for realism in oil.)
POSITIONS = ("long", "short", "flat")   # long = the whole account, short = the whole account, flat = no position
SIZE = 1.0                      # share of the account put on when long or short; no leverage

# Disaster stop (decision of 19 Oct 2026): set BY THE GAME when a position opens, the same for every
# player except the hold-oil benchmark, at STOP_ATR_MULT x the 14-day ATR from the entry (fill) price,
# but never further than STOP_MAX_PCT from it (decision of 19 Oct 2026: 3 ATRs reached 28% in March 2026).
# The ATR is the one known at the decision close. Players do not choose or move it; they exit by
# choosing flat. Wide on purpose: a safety net for violent moves, rarely the reason a trade ends.
STOP_ATR_MULT = 3.0
STOP_MAX_PCT = 0.15
ATR_DAYS = 14

# Timing (the anti-look-ahead contract, unchanged from September):
#   decision at the close of day t, using prices up to and including t
#   fill at the open of the next trading day t+1
#   the disaster stop is checked against the low (long) or high (short) of each day from the fill day on
#   a stop that is gapped through fills at the open (the worse price), else at the stop
#   the account is marked at the close of t+1
# A position is held until the agent asks for a different one or the stop trades.

# Prices: the front-month WTI price as quoted (Yahoo CL=F), NOT adjusted for contract
# rolls. Decision of 19 Oct 2026; reasons in data.py. Every player faces the same prices.
ADJUST_ROLLS = False

# Replay window (to confirm): one year after both LLMs' training cutoffs.
WINDOW_START = "2025-10-01"
WINDOW_END = "2026-09-30"

# The forecast every agent states each day, scored separately from its trades:
# the probability that the next close is above today's close.
FORECAST_HORIZON_DAYS = 1
