-- Initial schema. Applied once by `python -m rebalancer.migrate`; later changes go in new
-- numbered files, never edits to this one.
--
--   * Money and quantities are numeric, never float. Timestamps are timestamptz (stored UTC).
--   * The broker is the truth for positions and cash; these tables are what the engine did,
--     why, and what it believed, so every trade can be explained and every restart resumes.
--   * Audit tables (decisions, order_events, fills, alerts, lot_closures) are append-only:
--     the engine role can insert but not update or delete them.
--   * Models come from git. The engine copies each committed version in when it loads it and
--     never writes one otherwise.

create type asset_class as enum ('equity', 'crypto');
create type order_side as enum ('buy', 'sell');
create type order_status as enum (
    'pending', 'accepted', 'partially_filled', 'filled', 'canceled', 'expired', 'rejected'
);
create type halt_scope as enum ('global', 'daily', 'lane');

-- ---------------------------------------------------------------------------------------------
-- Models and accounts
-- ---------------------------------------------------------------------------------------------

create table model_versions (
    id          bigint generated always as identity primary key,
    name        text not null,
    version     integer not null check (version > 0),
    yaml        text not null,
    sha256      text not null,
    git_commit  text not null,
    cash_floor  numeric(6, 5) not null check (cash_floor between 0 and 1),
    loaded_at   timestamptz not null default now(),
    unique (name, version)
);

create table model_sleeves (
    model_version_id   bigint not null references model_versions,
    sleeve             text not null,
    target             numeric(6, 5) not null check (target between 0 and 1),
    band_abs           numeric(6, 5),
    band_rel           numeric(6, 5),
    max_off_hours_pct  numeric(6, 5) not null,
    instruments        jsonb not null,  -- {"regular": ["VOO"], "overnight": ["VOO"], "weekend": []}
    primary key (model_version_id, sleeve)
);

create table accounts (
    id                 bigint generated always as identity primary key,
    name               text not null unique,
    broker             text not null default 'alpaca',
    broker_account_id  text not null,
    mode               text not null check (mode in ('paper', 'live')),
    tax_treatment      text not null check (tax_treatment in ('taxable', 'traditional_ira', 'roth_ira')),
    created_at         timestamptz not null default now(),
    unique (broker, broker_account_id)
);

-- Which model an account follows, with history. A new row starts a move to a new version.
create table account_models (
    account_id        bigint not null references accounts,
    model_version_id  bigint not null references model_versions,
    effective_from    timestamptz not null,
    primary key (account_id, effective_from)
);

create table instruments (
    symbol              text primary key,      -- name used in models, e.g. BTC-USD
    broker_symbol       text not null,         -- e.g. BTC/USD at Alpaca
    asset_class         asset_class not null,
    venue               text not null,
    lot_size            numeric(20, 10) not null check (lot_size > 0),
    min_order_qty       numeric(28, 10),
    min_order_notional  numeric(18, 2),
    fractionable        boolean not null,
    active              boolean not null default true,
    refreshed_at        timestamptz not null   -- last read from the broker's asset list
);

-- ---------------------------------------------------------------------------------------------
-- Engine cycles: what it saw
-- ---------------------------------------------------------------------------------------------

-- One row per cycle per account. The newest row is also the heartbeat the watchdog checks.
create table cycles (
    id                bigint generated always as identity primary key,
    account_id        bigint not null references accounts,
    model_version_id  bigint not null references model_versions,
    ts                timestamptz not null,
    session           text not null,
    account_value     numeric(18, 2),
    halted            text,
    engine_build      text not null            -- git commit of the running code
);
create index cycles_account_ts on cycles (account_id, ts desc);

-- Prices the cycle used. Also rebuilds the 24h BTC/ETH history for the drawdown pause on restart.
create table cycle_prices (
    cycle_id    bigint not null references cycles on delete cascade,
    instrument  text not null references instruments,
    mid         numeric(20, 8) not null check (mid > 0),
    quote_ts    timestamptz not null,
    fresh       boolean not null,
    primary key (cycle_id, instrument)
);
create index cycle_prices_instrument_ts on cycle_prices (instrument, quote_ts desc);

create table sleeve_marks (
    cycle_id  bigint not null references cycles on delete cascade,
    sleeve    text not null,
    value     numeric(18, 2) not null,
    weight    numeric(8, 6) not null,
    stale     boolean not null,
    primary key (cycle_id, sleeve)
);

-- ---------------------------------------------------------------------------------------------
-- Decisions and orders: what it did and why
-- ---------------------------------------------------------------------------------------------

create table decisions (
    id        bigint generated always as identity primary key,
    cycle_id  bigint not null references cycles,
    sleeve    text not null,
    action    text not null check (action in ('ok', 'hold', 'skip', 'trade', 'reject', 'halt', 'cash_flow')),
    detail    text not null,
    data      jsonb         -- the numbers behind detail: weight, band edges, need, caps applied
);
create index decisions_cycle on decisions (cycle_id);

create table orders (
    id                bigint generated always as identity primary key,
    client_order_id   text not null unique,    -- sent to the broker, so a retry can't double-place
    broker_order_id   text unique,
    account_id        bigint not null references accounts,
    cycle_id          bigint not null references cycles,
    decision_id       bigint references decisions,
    sleeve            text not null,
    instrument        text not null references instruments,
    side              order_side not null,
    qty               numeric(28, 10) not null check (qty > 0),
    order_type        text not null check (order_type in ('limit', 'market')),
    time_in_force     text not null check (time_in_force in ('day', 'gtc', 'ioc')),
    limit_price       numeric(20, 8) check (limit_price > 0),
    reference_price   numeric(20, 8) not null check (reference_price > 0),
    notional          numeric(18, 2) not null,
    session           text not null,
    off_hours_window  timestamptz,             -- last regular close, when placed outside it
    reason            text not null,
    status            order_status not null,
    reject_reason     text,
    created_at        timestamptz not null,
    updated_at        timestamptz not null,
    check (order_type = 'market' or limit_price is not null),
    check (order_type = 'limit' or session = 'regular')
);
create index orders_open on orders (account_id)
    where status in ('pending', 'accepted', 'partially_filled');
create index orders_account_created on orders (account_id, created_at);
create index orders_off_hours_window on orders (account_id, off_hours_window)
    where off_hours_window is not null;

-- Every status change, as reported by the broker's order stream.
create table order_events (
    id        bigint generated always as identity primary key,
    order_id  bigint not null references orders,
    ts        timestamptz not null,
    status    order_status not null,
    detail    text,
    raw       jsonb
);
create index order_events_order on order_events (order_id, ts);

create table fills (
    id              bigint generated always as identity primary key,
    broker_fill_id  text not null unique,
    order_id        bigint references orders,  -- null for fills the engine didn't place
    account_id      bigint not null references accounts,
    instrument      text not null references instruments,
    side            order_side not null,
    qty             numeric(28, 10) not null check (qty > 0),
    price           numeric(20, 8) not null check (price > 0),
    fee             numeric(18, 8) not null default 0 check (fee >= 0),
    ts              timestamptz not null,
    raw             jsonb
);
create index fills_account_ts on fills (account_id, ts);

create table cash_flows (
    id                   bigint generated always as identity primary key,
    account_id           bigint not null references accounts,
    broker_activity_id   text not null unique,
    ts                   timestamptz not null,
    amount               numeric(18, 2) not null,  -- positive in, negative out
    kind                 text not null check (kind in ('deposit', 'withdrawal', 'dividend', 'interest', 'fee', 'other')),
    raw                  jsonb
);

-- ---------------------------------------------------------------------------------------------
-- Risk state: survives a restart
-- ---------------------------------------------------------------------------------------------

-- Engine vs broker, whenever the engine checks. Cash is the line with instrument 'USD'.
create table reconciliations (
    id          bigint generated always as identity primary key,
    account_id  bigint not null references accounts,
    cycle_id    bigint references cycles,
    ts          timestamptz not null,
    gap_usd     numeric(18, 2) not null,
    ok          boolean not null
);

create table reconciliation_lines (
    reconciliation_id  bigint not null references reconciliations on delete cascade,
    instrument         text not null,
    engine_qty         numeric(28, 10) not null,
    broker_qty         numeric(28, 10) not null,
    price              numeric(20, 8),
    gap_usd            numeric(18, 2) not null,
    primary key (reconciliation_id, instrument)
);

-- Open halts are the rows with cleared_at null. The dashboard's halt button inserts one.
create table halts (
    id          bigint generated always as identity primary key,
    account_id  bigint references accounts,     -- null halts every account
    scope       halt_scope not null,
    lane        text,                           -- e.g. alpaca/equity, for scope = 'lane'
    reason      text not null,
    source      text not null check (source in ('engine', 'watchdog', 'dashboard', 'manual')),
    started_at  timestamptz not null default now(),
    until       timestamptz,                    -- set for daily halts
    cleared_at  timestamptz,
    cleared_by  text,
    check ((scope = 'lane') = (lane is not null)),
    check ((scope = 'daily') = (until is not null))
);
create index halts_open on halts (account_id) where cleared_at is null;

-- The first shortfall seen for a sleeve in an off-hours window, which the 25% cap is taken
-- from. Turnover itself is summed from orders.
create table off_hours_needs (
    account_id    bigint not null references accounts,
    window_start  timestamptz not null,
    sleeve        text not null,
    need          numeric(18, 2) not null check (need >= 0),
    primary key (account_id, window_start, sleeve)
);

create table alerts (
    id               bigint generated always as identity primary key,
    account_id       bigint references accounts,
    ts               timestamptz not null,
    level            text not null check (level in ('info', 'warning', 'critical')),
    message          text not null,
    delivered_at     timestamptz,
    acknowledged_at  timestamptz
);
create index alerts_unacknowledged on alerts (ts) where acknowledged_at is null;

-- ---------------------------------------------------------------------------------------------
-- Tax lots
-- ---------------------------------------------------------------------------------------------

create table tax_lots (
    id                    bigint generated always as identity primary key,
    account_id            bigint not null references accounts,
    instrument            text not null references instruments,
    open_fill_id          bigint not null references fills,
    acquired_at           timestamptz not null,
    holding_period_start  timestamptz not null,  -- moves back when a wash sale carries over
    qty                   numeric(28, 10) not null check (qty > 0),
    qty_open              numeric(28, 10) not null,
    cost_per_unit         numeric(20, 8) not null,  -- fee included
    wash_sale_adjustment  numeric(18, 2) not null default 0,  -- disallowed loss added to basis
    check (qty_open between 0 and qty)
);
create index tax_lots_open on tax_lots (account_id, instrument) where qty_open > 0;

create table lot_closures (
    id                    bigint generated always as identity primary key,
    lot_id                bigint not null references tax_lots,
    close_fill_id         bigint not null references fills,
    qty                   numeric(28, 10) not null check (qty > 0),
    proceeds              numeric(18, 2) not null,   -- fee deducted
    cost                  numeric(18, 2) not null,
    realized              numeric(18, 2) not null,
    term                  text not null check (term in ('short', 'long')),
    wash_sale_disallowed  numeric(18, 2) not null default 0
);
create index lot_closures_fill on lot_closures (close_fill_id);

-- ---------------------------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------------------------

-- Daily closes for marking frozen sleeves, and anything else pulled in bulk.
create table reference_prices (
    instrument  text not null references instruments,
    kind        text not null check (kind in ('close', 'last')),
    ts          timestamptz not null,
    price       numeric(20, 8) not null check (price > 0),
    source      text not null,
    primary key (instrument, kind, ts)
);

create table corporate_actions (
    id          bigint generated always as identity primary key,
    instrument  text not null references instruments,
    ex_date     date not null,
    kind        text not null check (kind in ('split', 'dividend', 'symbol_change', 'other')),
    ratio       numeric(20, 10),
    amount      numeric(18, 8),
    raw         jsonb,
    applied_at  timestamptz,
    unique (instrument, ex_date, kind)
);

-- ---------------------------------------------------------------------------------------------
-- Retention
-- ---------------------------------------------------------------------------------------------

-- Cycle prices and marks are kept at full resolution for `keep_full`, then thinned to the first
-- cycle in each `resolution` bucket. Cycles, decisions and orders are never pruned.
create function prune_cycle_detail(
    keep_full   interval    default interval '90 days',
    resolution  interval    default interval '5 minutes',
    as_of       timestamptz default now()
) returns bigint
language sql
as $$
    with ranked as (
        select id,
               row_number() over (
                   partition by account_id, date_bin(resolution, ts, timestamptz '2000-01-01')
                   order by ts, id
               ) as n
        from cycles
        where ts < as_of - keep_full
    ),
    thinned as (
        select id from ranked where n > 1
    ),
    prices as (
        delete from cycle_prices where cycle_id in (select id from thinned) returning 1
    ),
    marks as (
        delete from sleeve_marks where cycle_id in (select id from thinned) returning 1
    )
    select (select count(*) from prices) + (select count(*) from marks);
$$;

-- ---------------------------------------------------------------------------------------------
-- Roles
-- ---------------------------------------------------------------------------------------------

-- Roles belong to the whole cluster, so another database may already have created them.
-- Login users get one of these with `grant rebalancer_engine to <user>`.
do $$
declare
    name text;
begin
    foreach name in array array['rebalancer_engine', 'rebalancer_watchdog', 'rebalancer_dashboard'] loop
        if not exists (select from pg_roles where rolname = name) then
            execute format('create role %I nologin', name);
        end if;
    end loop;
end
$$;

grant usage on schema public to rebalancer_engine, rebalancer_watchdog, rebalancer_dashboard;
grant select on all tables in schema public
    to rebalancer_engine, rebalancer_watchdog, rebalancer_dashboard;
grant usage on all sequences in schema public
    to rebalancer_engine, rebalancer_watchdog, rebalancer_dashboard;

-- Tables added by later migrations are readable by all three without repeating this.
alter default privileges in schema public
    grant select on tables to rebalancer_engine, rebalancer_watchdog, rebalancer_dashboard;
alter default privileges in schema public
    grant usage on sequences to rebalancer_engine, rebalancer_watchdog, rebalancer_dashboard;

-- Engine: inserts everywhere; updates only where state really changes.
grant insert on all tables in schema public to rebalancer_engine;
grant update on orders, tax_lots, halts, instruments, corporate_actions, alerts
    to rebalancer_engine;
grant delete on cycle_prices, sleeve_marks to rebalancer_engine;  -- retention pruning only
revoke execute on function prune_cycle_detail(interval, interval, timestamptz) from public;
grant execute on function prune_cycle_detail(interval, interval, timestamptz) to rebalancer_engine;

-- Watchdog: can raise halts and alerts. It cancels orders through the broker, not here.
grant insert on halts, alerts to rebalancer_watchdog;

-- Dashboard: read-only apart from the halt button and acknowledging alerts.
grant insert on halts to rebalancer_dashboard;
grant update (acknowledged_at) on alerts to rebalancer_dashboard;
