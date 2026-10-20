"""oilbench: the September agent-alpha-bench game, narrowed to one market (front-month WTI).

Data flow: data (quoted front-month prices, roll dates) -> prompt (payload) -> agent
(decision) -> engine (fills, stop, marks) -> metrics (returns, baselines, random
traders, forecast scoring).
"""

__version__ = "0.1.0"
