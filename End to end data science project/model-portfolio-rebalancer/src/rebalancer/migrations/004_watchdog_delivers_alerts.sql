-- The watchdog sends warnings and critical alerts to my phone, and marks each one when it has
-- gone out so it isn't sent twice.
grant update (delivered_at) on alerts to rebalancer_watchdog;
