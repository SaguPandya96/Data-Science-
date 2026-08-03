-- This table deliberately excludes every window touched by the simulation layer.
-- Rolling baselines use only earlier windows from the same campaign, so the
-- features could have existed at scoring time.
WITH observed_only AS (
    SELECT
        window_start,
        campaign,
        impressions,
        clicks,
        conversions,
        total_cost_units,
        avg_cost_units,
        active_click_minutes,
        unique_users,
        unique_sources,
        repeat_event_share,
        top_source_share,
        ctr,
        conversion_rate,
        active_click_minute_share,
        clicks_per_unique_user
    FROM campaign_window_features
    WHERE simulated_events = 0
      AND observed_events = impressions
),
history_raw AS (
    SELECT
        *,
        CAST(window_start / 3600 AS INTEGER) % 24 AS hour_of_day,
        COUNT(*) OVER campaign_history AS history_windows,
        AVG(impressions) OVER campaign_history AS baseline_impressions,
        AVG(ctr) OVER campaign_history AS baseline_ctr,
        AVG(conversion_rate) OVER campaign_history AS baseline_conversion_rate,
        AVG(total_cost_units) OVER campaign_history AS baseline_cost_units
    FROM observed_only
    WINDOW campaign_history AS (
        PARTITION BY campaign
        ORDER BY window_start
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    )
)
SELECT
    *,
    CASE
        WHEN baseline_impressions IS NULL OR baseline_impressions = 0 THEN NULL
        ELSE CAST(impressions AS REAL) / baseline_impressions
    END AS volume_change_ratio,
    CASE
        WHEN baseline_ctr IS NULL THEN NULL
        ELSE ctr - baseline_ctr
    END AS ctr_change_from_baseline,
    CASE
        WHEN baseline_conversion_rate IS NULL THEN NULL
        ELSE conversion_rate - baseline_conversion_rate
    END AS conversion_change_from_baseline
FROM history_raw
ORDER BY window_start, campaign;
