# Model Portfolio Rebalancer

I hold a mix of US equities, crypto and cash, and I want each account to stay on its target
weights at any hour: during a weekend crypto crash, or when stocks reopen Sunday night after
the weekend's news. Nobody should have to wait for Monday 9:30. This project is the engine for
that. It holds a few model portfolios, watches prices, and trades only the parts of the
portfolio whose market is open, in small limit orders, inside hard risk limits.

Everything runs through one Alpaca account, equities and crypto together against one cash
balance. Phase 1 is a simulator of that account, with the real session and order rules. Phase 2
connects to an Alpaca **paper** account: an adapter for Alpaca, a runner that drives the engine
against it once a minute, a watchdog that cancels orders if the engine goes quiet, and a
dashboard with a halt button.

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
  | Quote age | 60 seconds for equities, 5 minutes for crypto | skip the sleeve this cycle |
  | BTC or ETH drawdown | 15% below its 24-hour high | pause crypto buys, alert |
  | Reconciliation | engine vs venue off by more than $10 | halt everything, alert |
  | Broker rejects | 3 in 10 minutes for one asset class | halt that asset class, alert |
  | Order type | no market orders outside the regular session | reject |

  Equity trades outside the regular session are also capped at 25% of the rebalance needed
  that night. The rest waits for the open. Trades under $25 are skipped, and buys never take
  cash below the 2% floor.
- **Settlement.** The accounts are cash accounts, so buys only spend settled cash.
  - Equity sale proceeds settle one trading day after the trade date. A trade in the
    overnight session counts toward the next trading day, so a Tuesday 23:00 sale settles
    Thursday.
  - Until then the proceeds count toward the cash weight but can't be spent. Money from
    selling VOO on Tuesday reaches BTC on Wednesday.
  - Never spending unsettled cash also rules out good-faith violations: nothing is ever
    bought with money that could be sold before it settles.
  - Crypto sale proceeds are treated as available at once. `EngineConfig.settlement_days`
    changes either rule.
- **The replay harness** feeds a price history through the engine, one cycle per timestamp.
  It writes every order, decision and alert to CSV.

## Running it

Python 3.11 to 3.14. CI tests the oldest and the newest.

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
| Core 24/7 | 24/7 engine | 0.9% | 12 | 0.5% | $1.20 |
| Core 24/7 | regular hours only | 0.9% | 1 | 0.4% | $1.08 |
| Core 24/7 | no rebalancing | 12.6% | 0 | 0% | $0 |
| Growth 24/7 | 24/7 engine | 1.7% | 34 | 3.6% | $5.54 |
| Growth 24/7 | regular hours only | 3.8% | 6 | 3.4% | $4.92 |
| Growth 24/7 | no rebalancing | 15.1% | 0 | 0% | $0 |
| Crypto-tilt | 24/7 engine | 5.7% | 94 | 8.7% | $18.28 |
| Crypto-tilt | regular hours only | 12.0% | 18 | 5.8% | $11.71 |
| Crypto-tilt | no rebalancing | 32.3% | 0 | 0% | $0 |

Fees are Alpaca's crypto taker fee at its lowest volume tier (25 bps, from my reading of the
schedule, and what the paper account charged). Equity trades are commission-free. The simulator
takes the fee in dollars; Alpaca takes it out of the coins a buy receives. The cost is the same.

The full table, with end values and worst drift, is in
[`comparison.md`](reports/sample_replay/comparison.md).

- **Trading around the clock cuts time out of band by about half** for Growth and
  Crypto-tilt compared with waiting for the regular session. The cost is more orders, more
  turnover, and higher crypto fees ($18 against $12 for Crypto-tilt over seven weeks). For
  Core, with wide bands and little crypto, it makes no difference.
- **The crash weekend worked as intended.** Crypto buys paused at 06:30 Saturday, when ETH
  was 16% below its 24-hour high. They resumed Sunday 06:00, and Crypto-tilt then bought
  crypto back within the weekend's 5% budget. No equity order was sent while equities were
  closed. No run hit a halt, and cash never went below 8.9% (the floor is 2%).
- **Trading only to the band edge makes a lot of small trades in a trend.** During the
  overnight rally into December 10, Growth sold BTC 14 times between midnight and the 9:30
  open, $37 to $278 each. After each trade the sleeve sits right on the edge, so the next move
  pushes it back out. It's cheap here but noisy. A wider no-trade zone or trading partway to
  target would cut it, and I haven't changed anything yet.
- **The overnight cap limits how much equities can catch up.** The same rally left US large
  cap underweight. Fractional shares let the engine buy VOO overnight and pre-market, but
  only $31 to $55 at a time under the 25% cap. Most of the gap waited for a $1,142 buy at the
  9:30 open. Worst drift for Growth was 4.1 points, against 4.6 when trading only in regular
  hours. For Crypto-tilt it was 4.5 against 4.6.

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

## Database

Postgres 16 is the system of record. The schema is in
[`migrations/001_initial.sql`](src/rebalancer/migrations/001_initial.sql). It covers model
versions, accounts, every cycle's prices and marks, decisions, orders and their status
changes, fills, cash flows, reconciliations, halts, alerts and tax lots. Later changes go in
new numbered files. Never edit one that has been applied: the runner stores a checksum for
each file and refuses one that changed.

```bash
python -m pip install -e ".[dev,db]"
export DATABASE_URL=postgresql://user@host/rebalancer
python -m rebalancer.migrate            # apply pending migrations
python -m rebalancer.migrate status
```

- **Roles.** The migration creates three roles; each login user is granted one of them.
  - `rebalancer_engine` inserts everywhere. It can update only rows whose state really
    changes, like order status, open lots and halts. It can't rewrite decisions, fills or
    alerts.
  - `rebalancer_watchdog` reads everything, can raise halts and alerts, and can mark an
    alert as delivered.
  - `rebalancer_dashboard` reads everything, can insert a halt (the halt button) and can
    acknowledge alerts.
- **Models.** Git is the source of truth. The database keeps a copy of each version as the
  engine loaded it, with its commit, checksum and sleeves.
- **Retention.** `select prune_cycle_detail()` keeps prices and marks at full resolution
  for 90 days, then thins them to one cycle every 5 minutes. Cycles, decisions and orders
  are never pruned.
- **Tax lots.** The plan is to sell the highest-cost lot first, and to prefer lots held
  over a year when selling at a gain. Whether the wash-sale rule applies to crypto is a
  question for a tax professional; the schema can record wash-sale adjustments either way.

**Recording.** Give the engine an audit log with a store and it writes as it goes:

```python
conn = psycopg.connect(dsn, autocommit=True)
store = PostgresStore.open(conn, model, instruments, account=Account("main", "PA123"))
engine = Engine(model, venues, instruments, calendar, audit=AuditLog(store=store))
```

- **Every cycle** gets a row, with the prices it used, the sleeve marks, a reconciliation
  against the broker, and the decisions it made.
- **Orders are written as `pending` before they go to the broker.** The broker never holds
  an order the database doesn't know about. Each status change then becomes an order
  event, and fills link back to their order.
- **Orders can fill in pieces.** Each fill updates positions straight away and moves the
  order to `partially_filled`. When the rest is cancelled to re-quote, only its unfilled
  share of the turnover and overnight budgets is given back. The next cycle sizes a new
  order for whatever is still needed.
- **Halts** are written when they start and when they clear. Alerts and cash flows are
  recorded too.
- **If a write fails,** the engine stops writing and halts trading. It never sends an
  order it couldn't record.
- **Opening a store** records the model version and refuses a model file that changed
  without a version bump.

**Restarts.** With a store attached, `engine.start()` resumes the account from the database.
Positions still come from the broker.

- **Open orders** go back on the books, and the next cycle cancels and re-quotes them as
  usual.
- **An order still `pending`** means the engine died between writing it and hearing back.
  The engine asks the broker by client order id: if the broker has it, the order is picked
  up; if not, it's closed as "never reached the broker".
- **Partly filled orders** come back with the quantity already filled and its average
  price, rebuilt from the stored fills.
- **Fills made while the engine was down** are recorded and linked to their orders. They
  aren't added to the ledger a second time, since the broker's positions already include
  them.
- **Budgets:** today's turnover, each night's off-hours budget, and each sleeve's overnight
  shortfall and what's been bought against it are all restored.
- **The crypto drawdown pause** is rebuilt from the last day of stored BTC and ETH prices,
  without raising the alert again.
- **Halts are read from the database every cycle,** not only at start. A halt inserted from
  the dashboard or the watchdog stops trading on the next cycle and cancels resting orders.
  Setting `cleared_at` on it lets trading resume.

`python -m rebalancer.replay --dsn ...` records each model's 24/7 run as its own paper
account. Recording all three sample runs, 2,230 cycles each, takes about 20 seconds.

The database tests create a throwaway database for each test. Point `TEST_DATABASE_URL` at a
server where the user can create databases and roles; without it those tests are skipped.

## Alpaca paper account

`alpaca.py` implements the same adapter interface as the simulated broker, against Alpaca's
Trading and Market Data APIs.

- **Keys** come only from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in the environment,
  never from a file in the repo.
- **Paper only.** The adapter refuses any trading URL other than
  `https://paper-api.alpaca.markets`.
- **Order rules.** Equities use the same session rules as the simulator. Orders outside the
  regular session go out as limit orders with `extended_hours` set. Crypto orders use Alpaca's
  `BTC/USD` symbols and GTC.
- **Unclear sends are never guessed.** If a send times out, the adapter looks the order up by
  its client order id. If the order isn't found it counts as not sent, and if the lookup fails
  too the adapter raises an error.
- **Missing quotes don't stop the engine.** A missing or failed quote just means that sleeve
  waits a cycle.
- **Position symbols.** Positions come back under the engine's own symbols (Alpaca reports
  `BTCUSD`). Anything else in the account is reported under its own name, so reconciliation
  flags it. Use a paper account that holds nothing else.

To look at the paper account:

```bash
python -m pip install -e ".[dev,alpaca]"
export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
python -m rebalancer.alpaca check              # read-only
python -m rebalancer.alpaca check --order-test # also places and cancels one VOO limit at half the price
python -m rebalancer.alpaca cash-only          # set the margin multiplier to 1
```

`check` prints the market clock, the account's cash and buying power, its margin multiplier,
positions, a quote and the session for each instrument, and the latest fill and crypto-fee
activities.

**What the paper account showed.** I ran `check` and `check --order-test` against a new paper
account on September 28, 2026, during regular hours.

- The clock, cash, positions, quotes and session names all read correctly, and the crypto
  symbols mapped both ways.
- The order test placed a VOO limit, found it again by its client id, cancelled it, and read
  back `canceled`.
- **Paper accounts start as 4x margin accounts:** $100,000 of cash showed $400,000 of buying
  power. The engine only spends settled cash, but `cash-only` makes the broker enforce that too.
- **ETH quotes were 53 and 58 seconds old** on two runs, against 0 to 3 seconds for the other
  three. Alpaca only sends a crypto quote when the book changes, so with a 60-second limit ETH
  would have been skipped most cycles. Crypto quotes may now be up to 5 minutes old. That also
  means a frozen crypto feed takes up to 5 minutes to notice.
- **The first engine cycle** ran with `runner --once` after hours. It bought about $2,500 each
  of BTC and ETH, which is the 5% off-hours budget split between them.
- **Crypto fees come out of the coins bought.** The account held exactly 0.25% less ETH and BTC
  than it had bought, while its cash matched the fills to the cent. For BTC, filled in three
  pieces, 0.25% of each piece rounded up to 9 decimals adds up to the shortfall exactly. No
  `CFEE` activity had appeared. Without accounting for this, the engine's balances would have
  been about $12 off the broker's after those two buys, and reconciliation would have halted
  trading. The adapter now charges crypto buys 0.25% in coins, rounded up the same way.
  `ALPACA_CRYPTO_FEE_BPS` changes the rate, which Alpaca lowers as monthly volume grows.

**What I still haven't been able to verify.** Everything else runs against `tests/fake_alpaca.py`,
a model of Alpaca built from its documentation:

- **Crypto sale fees.** A sale is assumed to pay 0.25% out of the dollars received. The paper
  account hasn't sold any crypto yet. The first sale will show whether that's right, and a wrong
  guess shows up as a reconciliation gap.
- **`CFEE` activities.** None appeared after the first buys. The engine doesn't read them, so if
  they do turn up later they can't be counted twice.
- **Overnight quotes.** The default stock feed is IEX, which has no overnight quotes. Equities
  then read as stale overnight and aren't traded. Set `ALPACA_STOCK_FEED` to a feed that covers
  the overnight session if your data plan includes one.
- **Settled cash.** `check` prints `non_marginable_buying_power`. Compare it with the engine's
  own settlement tracking before relying on either.

## Running on a schedule

`runner.py` runs the engine against the Alpaca paper account, one cycle a minute on the minute.

```bash
python -m pip install -e ".[db,alpaca]"
export DATABASE_URL=... APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
python -m rebalancer.migrate
python -m rebalancer.runner --model models/growth-247.yaml --account paper-main --broker-account PA...
python -m rebalancer.runner ... --once   # a single cycle, then stop
```

- **Postgres is required.** The runner won't start without it, because the audit trail,
  restart recovery and halts all live there.
- **One runner per account.** At startup the runner claims the account with a Postgres
  lock, so a second runner on the same account refuses to start. The lock goes when the
  connection does, so a crashed runner doesn't leave the account locked.
- **Failures.** A failed cycle, like a broker timeout, is logged and alerted, and the next
  cycle tries again. Three failures in a row halt trading, and the halt is recorded like any
  other.
- **Slow cycles.** A cycle that overruns makes the loop skip ahead to the next minute rather
  than run the missed cycles back to back.
- **Stopping.** Ctrl-C or SIGTERM finishes the current cycle, cancels whatever is resting at
  the broker, and exits. Starting again resumes from the database.
- **Output.** Each cycle logs one line with the session, account value, orders placed and
  rejected, and any halt.

## Watchdog

`watchdog.py` is the dead-man's switch. It runs as its own process, next to the runner, and
checks every 30 seconds.

```bash
export DATABASE_URL=... APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
export WATCHDOG_WEBHOOK_URL=https://ntfy.sh/<topic>   # optional: alerts to my phone
python -m rebalancer.watchdog --account paper-main
```

- **Heartbeat.** The engine's newest cycle is its heartbeat. After five minutes without one,
  the watchdog cancels every order resting at the broker and sends a critical alert. It keeps
  cancelling anything new until the heartbeat comes back, then says so.
- **No halt for a quiet engine.** Stopping the runner on purpose stops the heartbeat too, so
  five minutes later I get the alert, and there's nothing left to cancel. A restarted engine
  carries on from the database without waiting for me to clear a halt.
- **Reconciliation.** The engine halts itself on a reconciliation gap before it records one.
  If the latest reconciliation failed and no halt is in force, something is wrong with the
  engine as well as the positions, so the watchdog inserts a global halt.
- **Alerts.** It sends each new alert, from the engine or its own, to the webhook, oldest
  first, and marks it delivered. The webhook gets plain text with `Title` and `Priority`
  headers, which is what ntfy expects. Without a webhook, alerts go to the log. A failed send
  is retried on the next check, and alerts more than a day old are not sent.
- **When the database is down.** It sends an alert straight to the webhook. It still counts
  five minutes from the last heartbeat it saw and cancels orders after that: not knowing
  whether the engine is alive is treated as the engine being dead.
- **Its own login.** Connect as a user in the `rebalancer_watchdog` role.
- **Orders cancelled behind the engine's back.** When the engine comes back it finds those
  orders gone. It asks the broker where each one stands. It closes an order the broker says
  was cancelled or expired, once every fill that order got has arrived, and gives the unfilled
  part back to the turnover budgets. The same handles an order I cancel by hand in Alpaca.

## Dashboard

`dashboard.py` serves one page per account: where each sleeve stands against its target and
band, orders resting at the broker, today's trades, open halts and unacknowledged alerts.

```bash
export DATABASE_URL=...
python -m rebalancer.dashboard --account paper-main   # http://127.0.0.1:8050
```

- **Read from Postgres only,** so it works whether or not the engine is running. It shows how
  long ago the last cycle was, and flags the engine as stopped after five minutes, the same
  limit the watchdog uses. The page refreshes every 30 seconds.
- **Halt button.** It inserts a global halt with my reason. The engine stops trading and
  cancels resting orders on its next cycle.
- **It can't resume trading.** Connect it as a user in the `rebalancer_dashboard` role, which
  can insert a halt and acknowledge an alert but can't clear a halt. Clearing one is done by
  hand in the database, so a stray tap can only stop trading.
- **Forms carry a token** made when the server starts, so another page open in the same
  browser can't post a halt or an acknowledgement.
- **Localhost only.** It has no login. To use it from my phone I reach it through an SSH tunnel
  or a private network rather than opening the port.
- **No new dependencies.** It uses Python's built-in HTTP server and plain HTML.

## Layout

```text
models/                  model portfolio files
src/rebalancer/
  models.py              schema, loader, validation
  sessions.py            session calendar
  engine.py              the rebalancing loop
  risk.py                risk gate and halts
  venue.py               adapter interface shared by every venue
  sim.py                 simulated broker account
  audit.py               decisions, orders and alerts
  replay.py              replay harness and command line
  scenarios.py           scripted price paths
  migrate.py             migration runner
  store.py               writes the audit trail to Postgres
  alpaca.py              Alpaca paper adapter and check command
  runner.py              scheduled loop against Alpaca
  watchdog.py            dead-man's switch and alert delivery
  dashboard.py           status page and halt button
  broker_rules.py        order types allowed in each session
  migrations/            numbered SQL schema files
tests/
reports/sample_replay/   output of the sample replay
```

## Limits of this version

- In the sample replay, the simulated broker fills any limit order that crosses the quote,
  in full: its book has no depth limit and no queue. Tests use `set_depth` to fill orders in
  pieces, but the sample doesn't. Real overnight and weekend books are thin, so fills there
  will be worse and more of them will be partial.
- Settlement is modelled by the engine from its own fills, not read from the broker.
  - The live adapter should check the engine's figure against the settled cash Alpaca
    reports.
  - I haven't confirmed that Alpaca makes crypto sale proceeds available at once.
  - Proceeds count as settled from midnight ET on the settlement date.
  - The sample replay never sells equities, so settlement never binds there. It's covered by
    the tests in `test_settlement.py`.
- Fractional equity orders assume VOO and VXUS are fractionable at Alpaca, and crypto lot
  sizes are placeholders. The live version should read both, with minimum order sizes, from
  the broker's asset list.
- At startup the engine takes positions from the broker as they are. It doesn't yet compare
  them with what the database expected, so a change made by hand while it was down passes
  without comment.
- The runner cycles on a timer only. The spec also asks for a cycle when prices move, which
  needs Alpaca's streaming feed.
- The watchdog only protects anything if it runs somewhere the engine's failure doesn't
  reach. On the same machine, a dead machine takes both down. Nothing watches the watchdog
  yet, and a watchdog started while the database is down has no heartbeat to count from.
- The watchdog uses the same Alpaca keys as the engine, and cancels every open order in the
  account, not only the engine's. Use an account the engine has to itself.
- Not built yet: the weekly calendar check, moving between model versions over several
  sessions, spread limits per session, slicing large orders, netting trades across accounts,
  tax-lot selection and wash-sale checks.
