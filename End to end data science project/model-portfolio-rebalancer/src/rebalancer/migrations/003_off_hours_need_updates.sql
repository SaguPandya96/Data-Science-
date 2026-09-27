-- The engine records the first shortfall it sees for a sleeve in each off-hours window and
-- raises it if the shortfall grows during the night, so it needs to update those rows.
grant update on off_hours_needs to rebalancer_engine;
