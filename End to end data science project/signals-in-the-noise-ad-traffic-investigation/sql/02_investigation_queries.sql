-- Questions I used while moving from a score to an investigation.

-- 1. Which campaign windows need attention first?
SELECT
    window_start,
    campaign,
    ROUND(risk_score, 3) AS risk_score,
    recommended_action,
    impressions,
    clicks,
    ROUND(ctr, 3) AS ctr,
    ROUND(conversion_per_click, 3) AS conversion_per_click,
    ROUND(repeat_event_share, 3) AS repeat_event_share,
    ROUND(top_source_share, 3) AS top_source_share,
    evidence_summary
FROM risk_scores
ORDER BY risk_score DESC
LIMIT 25;

-- 2. Are alerts concentrated in a small number of campaigns?
SELECT
    campaign,
    COUNT(*) AS reviewed_windows,
    ROUND(AVG(risk_score), 3) AS average_risk,
    ROUND(MAX(risk_score), 3) AS highest_risk,
    SUM(impressions) AS impressions_in_review
FROM risk_scores
WHERE recommended_action <> 'monitor'
GROUP BY campaign
ORDER BY highest_risk DESC, reviewed_windows DESC;

-- 3. What did the experimental scenarios look like after aggregation?
-- evaluation_scenario exists only for offline validation and would not be
-- available in a real review queue.
SELECT
    evaluation_scenario,
    COUNT(*) AS campaign_windows,
    ROUND(AVG(risk_score), 3) AS average_risk,
    SUM(CASE WHEN recommended_action <> 'monitor' THEN 1 ELSE 0 END) AS queued_windows
FROM risk_scores
GROUP BY evaluation_scenario
ORDER BY average_risk DESC;
