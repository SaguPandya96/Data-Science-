# Model Portfolio Rebalancer

I hold a mix of US equities, crypto and cash, and I want each account to stay on its target
weights at any hour: during a weekend crypto crash, or when stocks reopen Sunday night after
the weekend's news. Nobody should have to wait for Monday 9:30. This project is the engine for
that. It holds a few model portfolios, watches prices, and trades only the parts of the
portfolio whose market is open, in small limit orders, inside hard risk limits.

This is phase 1, a simulator. It doesn't connect to a broker or exchange and uses no API keys.
Orders go to simulated venues that follow the real session rules.

## What it does

- **Models** are YAML files in [`models/`](models). Each one splits the portfolio into sleeves
  (US large cap, international, BTC, ETH, cash) with a target weight, a drift band, and the
  instruments that can hold the sleeve in each session.
- **Sessions** follow the NYSE calendar in Eastern time: regular 9:30–16:00, pre-market from
  4:00, after-hours to 20:00, and the broker's overnight session from 20:00 to 4:00. Holidays
  and half-days come from `exchange_calendars` and can be overridden by hand. Equities are
  frozen from Friday 20:00 to Sunday night and on holidays, including the night before. From
  Sunday December 6, 2026 the overnight session opens at 21:00, with a daily 20:00–21:00 pause.
- **The engine** runs a cycle on every price update (every 60 seconds once it's live). Each
  cycle settles last cycle's orders, reconciles against the venues, and marks the portfolio.
  Then it finds sleeves outside their bands, drops any whose market is closed or whose quote
  is stale, and sizes trades back to the band edge rather than all the way to target. Cash is
  used before anything gets sold. A sleeve in a closed market is marked at its last print,
  flagged stale, and never used to trigger a trade.
- **The risk gate** checks every order before it leaves:

  | Rule | Limit | On breach |
  |---|---|---|
  | Single order | 5% of account | reject |
  | Daily turnover | 15% of account | clip, then halt until midnight ET |
  | Off-hours turnover | 5% of account per night or weekend | defer to the regular session |
  | Limit price | within 2% of the quote mid | reject |
  | Quote age | 60 seconds | skip the sleeve this cycle |
  | BTC or ETH drawdown | 15% below its 24-hour high | pause crypto buys, alert |
  | Reconciliation | engine vs venue off by more than $10 | halt everything, alert |
  | Venue rejects | 3 in 10 minutes | halt that venue, alert |
  | Order type | no market orders outside the regular session | reject |

  Equity trades outside the regular session are also capped at 25% of the rebalance needed
  that night. The rest waits for the open. Trades under $25 are skipped, and buys never take
  cash below the 2% floor.
- **The replay harness** feeds a price history through the engine, one cycle per timestamp.
  It writes every order, decision and alert to CSV.

## Running it

Python 3.11 or later.

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
python -m rebalancer.replay
```

The last command replays every model in `models/` over November 2 to December 18, 2026. It
writes the orders, decisions (why each sleeve traded, held or skipped) and alerts to
[`reports/sample_replay/`](reports/sample_replay). To replay your own prices, pass a CSV with
columns `ts,instrument,price`, with timezone-aware timestamps:

```bash
python -m rebalancer.replay --model models/growth-247.yaml --prices data/my_prices.csv --out reports/my_run
```

Equity rows should only appear while that market is open. The engine sees the gaps the same
way it would live.

## What the sample replay shows

The sample prices are scripted, not recorded: random walks with a few events added. There's a
20% crypto crash on Saturday November 14, an equity gap at the Sunday night reopen, and a
crypto rally in the first week of the new 23/5 schedule. They check that the engine behaves;
they say nothing about real returns. [`scenarios.py`](src/rebalancer/scenarios.py) has the
details.

Each model was run three ways from $100,000: the 24/7 engine, the same engine limited to the
regular session, and no rebalancing at all.

| Model | Run | Time any sleeve out of band | Orders filled | Turnover | Fees |
|---|---|---:|---:|---:|---:|
| Core 24/7 | 24/7 engine | 0.9% | 12 | 0.5% | $0.48 |
| Core 24/7 | regular hours only | 0.9% | 1 | 0.4% | $0.43 |
| Core 24/7 | no rebalancing | 12.6% | 0 | 0% | $0 |
| Growth 24/7 | 24/7 engine | 1.8% | 27 | 3.8% | $2.21 |
| Growth 24/7 | regular hours only | 3.9% | 6 | 3.5% | $1.96 |
| Growth 24/7 | no rebalancing | 15.5% | 0 | 0% | $0 |
| Crypto-tilt | 24/7 engine | 11.5% | 88 | 9.0% | $7.61 |
| Crypto-tilt | regular hours only | 14.8% | 24 | 6.0% | $4.87 |
| Crypto-tilt | no rebalancing | 32.3% | 0 | 0% | $0 |

The full table, with end values and worst drift, is in
[`comparison.md`](reports/sample_replay/comparison.md).

- **Trading around the clock cuts time out of band,** by about half for Growth and a fifth
  for Crypto-tilt compared with waiting for the regular session. The cost is more orders and
  a bit more turnover. For Core, with wide bands and little crypto, it makes no difference.
- **The crash weekend worked as intended.** Crypto buys paused at 06:30 Saturday, when ETH
  was 16% below its 24-hour high. They resumed Sunday 06:00, and Crypto-tilt then bought
  crypto back within the weekend's 5% budget. No equity order was sent while equities were
  closed. No run hit a halt, and cash never went below 9% (the floor is 2%).
- **Trading only to the band edge makes a lot of small trades in a trend.** During the
  overnight rally into December 10, Growth sold BTC 14 times between midnight and the 9:30 open,
  $34 to $278 each. After each trade the sleeve sits right on the edge, so the next move
  pushes it back out. It's cheap here but noisy. A wider no-trade zone or trading partway to
  target would cut it, and I haven't changed anything yet.
- **Worst drift barely moves between runs** (4.9 against 4.8 points for Crypto-tilt, 4.6
  against 4.6 for Growth). In both it's US large cap at the 9:30 open on December 10. The
  overnight crypto rally left it underweight, and the overnight equity trade was capped at
  25% of the need and then rounded down to zero whole VOO shares, so it waited for the open.
  24/7 trading only helps equities as much as the overnight cap and share size allow.

## Adding a model

Copy one of the files in `models/` and edit it. Each file has one sleeve per exposure.
Targets must sum to 1, and exactly one sleeve must have the id `cash`.

```yaml
model: my-model            # lowercase slug, unique across files
version: 1                 # bump on every change; the engine never edits a model
cash_floor: 0.02
sleeves:
  - id: us_large_cap
    target: 0.50
    band: {abs: 0.03, rel: 0.20}   # out of band when it leaves either one
    instruments:
      regular: [VOO]
      overnight: [VOO]             # also used pre-market and after-hours unless `extended` is set
      weekend: []                  # empty = frozen; add a tokenized fund here once one is allowed
    session_policy: {max_off_hours_pct: 0.25}
  - id: btc
    target: 0.40
    band: {abs: 0.03, rel: 0.15}
    instruments: {any: [BTC-USD]}  # `any` applies in every session
  - id: cash
    target: 0.10
    band: {abs: 0.03}              # optional; outside it, cash is deployed or raised
    instruments: {any: [USD, USDC]}
```

Instrument lists can be keyed by `regular`, `extended`, `overnight`, `weekend` or `any`. The
loader rejects files with targets that don't sum to 1, unknown keys or sessions, an instrument
in two sleeves, a cash target below the floor, or an equity sleeve without a band. Each new
instrument also needs a venue and lot size in `DEFAULT_INSTRUMENTS` in
[`replay.py`](src/rebalancer/replay.py). The starter models are examples, not
recommendations.

## Layout

```text
models/                  model portfolio files
src/rebalancer/
  models.py              schema, loader, validation
  sessions.py            session calendar
  engine.py              the rebalancing loop
  risk.py                risk gate and halts
  venue.py               adapter interface shared by every venue
  sim.py                 simulated venues
  audit.py               decisions, orders and alerts
  replay.py              replay harness and command line
  scenarios.py           scripted price paths
tests/
reports/sample_replay/   output of the sample replay
```

## Limits of this version

- The simulated venues fill any limit order that crosses the quote, in full. There's no
  order book depth, no partial fills and no queue. Real overnight and weekend books are
  thin, so fills there will be worse.
- One cash balance is shared across venues, as if equities and crypto sat with one broker.
  With a separate crypto exchange, cash would be split and moving it takes days.
- Equities trade in whole shares. On a $100,000 account, a quarter of a small overnight
  shortfall is often less than one VOO share, so the overnight trade rounds to nothing and
  waits for the open.
- Everything is in memory. Positions, orders and decisions aren't stored anywhere yet.
- Not built yet: the weekly calendar check, moving between model versions over several
  sessions, spread limits per session, slicing large orders, netting trades across accounts,
  tax lots and wash sales, the watchdog, and the dashboard.
