-- GDELT 2.0 hourly news metadata for RUB and the five recipient countries.
-- Run in BigQuery Standard SQL and download the result as CSV.
-- The output contains aggregates, not copyrighted article text.

DECLARE date_from DATE DEFAULT DATE('2018-01-01');
DECLARE date_to DATE DEFAULT DATE('2026-09-02');

WITH corridors AS (
  SELECT * FROM UNNEST([
    STRUCT('AMD' AS corridor, 'ARM' AS actor_code, 'AM' AS geo_code),
    STRUCT('KGS' AS corridor, 'KGZ' AS actor_code, 'KG' AS geo_code),
    STRUCT('KZT' AS corridor, 'KAZ' AS actor_code, 'KZ' AS geo_code),
    STRUCT('TJS' AS corridor, 'TJK' AS actor_code, 'TI' AS geo_code),
    STRUCT('UZS' AS corridor, 'UZB' AS actor_code, 'UZ' AS geo_code)
  ])
),
events AS (
  SELECT
    GlobalEventID,
    Actor1CountryCode,
    Actor2CountryCode,
    Actor1Geo_CountryCode,
    Actor2Geo_CountryCode,
    ActionGeo_CountryCode,
    EventRootCode,
    QuadClass,
    GoldsteinScale
  FROM `gdelt-bq.gdeltv2.events_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP(date_from)
    AND _PARTITIONTIME < TIMESTAMP(DATE_ADD(date_to, INTERVAL 1 DAY))
    AND (
      Actor1CountryCode IN ('RUS', 'ARM', 'KGZ', 'KAZ', 'TJK', 'UZB')
      OR Actor2CountryCode IN ('RUS', 'ARM', 'KGZ', 'KAZ', 'TJK', 'UZB')
      OR Actor1Geo_CountryCode IN ('RS', 'AM', 'KG', 'KZ', 'TI', 'UZ')
      OR Actor2Geo_CountryCode IN ('RS', 'AM', 'KG', 'KZ', 'TI', 'UZ')
      OR ActionGeo_CountryCode IN ('RS', 'AM', 'KG', 'KZ', 'TI', 'UZ')
    )
),
mentions AS (
  SELECT
    GlobalEventID,
    MentionTimeDate,
    MentionIdentifier,
    MentionSourceName,
    SAFE_CAST(MentionDocTone AS FLOAT64) AS mention_tone
  FROM `gdelt-bq.gdeltv2.eventmentions_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP(date_from)
    AND _PARTITIONTIME < TIMESTAMP(DATE_ADD(date_to, INTERVAL 1 DAY))
    AND MentionType = 1
    AND Confidence >= 40
),
mention_events AS (
  SELECT
    TIMESTAMP_TRUNC(
      PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(m.MentionTimeDate AS STRING)),
      HOUR
    ) AS timestamp_utc,
    c.corridor,
    m.MentionIdentifier AS article_id,
    m.MentionSourceName AS source_name,
    m.mention_tone,
    (
      e.Actor1CountryCode = 'RUS' OR e.Actor2CountryCode = 'RUS'
      OR e.Actor1Geo_CountryCode = 'RS' OR e.Actor2Geo_CountryCode = 'RS'
      OR e.ActionGeo_CountryCode = 'RS'
    ) AS has_russia,
    (
      e.Actor1CountryCode = c.actor_code OR e.Actor2CountryCode = c.actor_code
      OR e.Actor1Geo_CountryCode = c.geo_code OR e.Actor2Geo_CountryCode = c.geo_code
      OR e.ActionGeo_CountryCode = c.geo_code
    ) AS has_recipient,
    SAFE_CAST(e.QuadClass AS INT64) AS quad_class,
    SAFE_CAST(e.EventRootCode AS INT64) AS root_code,
    SAFE_CAST(e.GoldsteinScale AS FLOAT64) AS goldstein_scale
  FROM mentions AS m
  JOIN events AS e USING (GlobalEventID)
  CROSS JOIN corridors AS c
  WHERE
    e.Actor1CountryCode = 'RUS' OR e.Actor2CountryCode = 'RUS'
    OR e.Actor1Geo_CountryCode = 'RS' OR e.Actor2Geo_CountryCode = 'RS'
    OR e.ActionGeo_CountryCode = 'RS'
    OR e.Actor1CountryCode = c.actor_code OR e.Actor2CountryCode = c.actor_code
    OR e.Actor1Geo_CountryCode = c.geo_code OR e.Actor2Geo_CountryCode = c.geo_code
    OR e.ActionGeo_CountryCode = c.geo_code
),
-- One article can describe several extracted events. Collapse it before counting,
-- otherwise a long article would look like many independent news stories.
article_scope AS (
  SELECT
    timestamp_utc,
    corridor,
    article_id,
    ANY_VALUE(source_name) AS source_name,
    AVG(mention_tone) AS tone,
    LOGICAL_OR(has_russia) AS has_russia,
    LOGICAL_OR(has_recipient) AS has_recipient,
    MAX(IF(quad_class >= 3, 1, 0)) AS is_conflict,
    MAX(IF(quad_class = 4, 1, 0)) AS is_material_conflict,
    MAX(IF(quad_class <= 2, 1, 0)) AS is_cooperation,
    MAX(IF(root_code IN (16, 17), 1, 0)) AS is_coercion,
    MAX(IF(root_code = 14, 1, 0)) AS is_protest,
    AVG(goldstein_scale) AS goldstein_scale
  FROM mention_events
  GROUP BY timestamp_utc, corridor, article_id
)
SELECT
  timestamp_utc,
  corridor,
  COUNT(*) AS article_count,
  COUNTIF(has_russia) AS russia_article_count,
  COUNTIF(has_recipient) AS recipient_article_count,
  COUNTIF(has_russia AND has_recipient) AS bilateral_article_count,
  COUNT(DISTINCT source_name) AS source_count,
  SUM(tone) AS tone_sum,
  COUNT(tone) AS tone_count,
  SUM(IF(has_russia, tone, 0.0)) AS russia_tone_sum,
  COUNTIF(has_russia AND tone IS NOT NULL) AS russia_tone_count,
  SUM(IF(has_recipient, tone, 0.0)) AS recipient_tone_sum,
  COUNTIF(has_recipient AND tone IS NOT NULL) AS recipient_tone_count,
  COUNTIF(tone <= -2.0) AS negative_count,
  SUM(is_conflict) AS conflict_count,
  SUM(is_material_conflict) AS material_conflict_count,
  SUM(is_cooperation) AS cooperation_count,
  SUM(is_coercion) AS coercion_count,
  SUM(is_protest) AS protest_count,
  SUM(goldstein_scale) AS goldstein_sum,
  COUNT(goldstein_scale) AS goldstein_count
FROM article_scope
GROUP BY timestamp_utc, corridor
ORDER BY timestamp_utc, corridor;
