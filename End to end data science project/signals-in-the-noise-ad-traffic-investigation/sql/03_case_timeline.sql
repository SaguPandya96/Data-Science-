-- Replace :campaign_id with a campaign from the review queue.
SELECT
    timestamp,
    uid,
    source_id,
    click,
    conversion,
    ROUND(cost, 6) AS transformed_cost_units,
    time_since_last_click,
    event_origin
FROM events
WHERE campaign = :campaign_id
ORDER BY timestamp;
