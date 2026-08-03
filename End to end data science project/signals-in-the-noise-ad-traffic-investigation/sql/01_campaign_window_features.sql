WITH event_base AS (
    SELECT
        timestamp,
        CAST(timestamp / 1800 AS INTEGER) * 1800 AS window_start,
        campaign,
        uid,
        source_id,
        click,
        conversion,
        cost,
        time_since_last_click,
        event_origin,
        is_simulated_abuse,
        scenario
    FROM events
),
user_counts AS (
    SELECT window_start, campaign, uid, COUNT(*) AS user_events
    FROM event_base
    GROUP BY window_start, campaign, uid
),
user_rollup AS (
    SELECT
        window_start,
        campaign,
        COUNT(*) AS unique_users,
        SUM(CASE WHEN user_events > 1 THEN user_events ELSE 0 END) AS repeated_events
    FROM user_counts
    GROUP BY window_start, campaign
),
source_counts AS (
    SELECT window_start, campaign, source_id, COUNT(*) AS source_events
    FROM event_base
    GROUP BY window_start, campaign, source_id
),
source_rollup AS (
    SELECT
        window_start,
        campaign,
        COUNT(*) AS unique_sources,
        MAX(source_events) AS top_source_events
    FROM source_counts
    GROUP BY window_start, campaign
),
window_rollup AS (
    SELECT
        window_start,
        campaign,
        COUNT(*) AS impressions,
        SUM(click) AS clicks,
        SUM(conversion) AS conversions,
        SUM(cost) AS total_cost_units,
        AVG(cost) AS avg_cost_units,
        AVG(CASE WHEN time_since_last_click >= 0 THEN time_since_last_click END) AS avg_seconds_since_click,
        COUNT(DISTINCT CASE WHEN click = 1 THEN CAST(timestamp / 60 AS INTEGER) END) AS active_click_minutes,
        SUM(CASE WHEN event_origin = 'observed' THEN 1 ELSE 0 END) AS observed_events,
        SUM(CASE WHEN event_origin = 'simulation' THEN 1 ELSE 0 END) AS simulated_events,
        MAX(is_simulated_abuse) AS abuse_label,
        GROUP_CONCAT(DISTINCT CASE WHEN event_origin = 'simulation' THEN scenario END) AS evaluation_scenario
    FROM event_base
    GROUP BY window_start, campaign
)
SELECT
    w.window_start,
    w.campaign,
    w.impressions,
    w.clicks,
    w.conversions,
    w.total_cost_units,
    w.avg_cost_units,
    COALESCE(w.avg_seconds_since_click, -1) AS avg_seconds_since_click,
    w.active_click_minutes,
    u.unique_users,
    s.unique_sources,
    CAST(u.repeated_events AS REAL) / w.impressions AS repeat_event_share,
    CAST(s.top_source_events AS REAL) / w.impressions AS top_source_share,
    CAST(w.clicks AS REAL) / w.impressions AS ctr,
    CAST(w.conversions AS REAL) / w.impressions AS conversion_rate,
    CASE WHEN w.clicks = 0 THEN 0 ELSE CAST(w.conversions AS REAL) / w.clicks END AS conversion_per_click,
    CAST(w.active_click_minutes AS REAL) / 30.0 AS active_click_minute_share,
    CAST(w.clicks AS REAL) / MAX(u.unique_users, 1) AS clicks_per_unique_user,
    w.observed_events,
    w.simulated_events,
    w.abuse_label,
    COALESCE(w.evaluation_scenario, 'observed') AS evaluation_scenario
FROM window_rollup w
JOIN user_rollup u USING (window_start, campaign)
JOIN source_rollup s USING (window_start, campaign)
WHERE w.impressions >= 5
ORDER BY w.window_start, w.campaign;
