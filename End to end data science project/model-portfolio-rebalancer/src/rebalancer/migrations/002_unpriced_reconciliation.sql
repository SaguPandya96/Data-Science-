-- A mismatch in an instrument the engine has no price for yet can't be valued in dollars.
-- Record it with a null gap instead of a made-up number; the row's ok flag is false either way.
alter table reconciliations alter column gap_usd drop not null;
alter table reconciliation_lines alter column gap_usd drop not null;
