-- 基础表保持第三范式，时间属性通过视图派生。
CREATE TABLE dim_user (
    user_id TEXT PRIMARY KEY NOT NULL
) WITHOUT ROWID;

CREATE TABLE dim_item (
    item_id TEXT PRIMARY KEY NOT NULL,
    category_id TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE dim_time (
    event_time TEXT PRIMARY KEY NOT NULL,
    CHECK(length(event_time) = 19),
    CHECK(substr(event_time, 15, 5) = '00:00')
) WITHOUT ROWID;

CREATE TABLE user_behavior (
    event_id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES dim_user(user_id),
    item_id TEXT NOT NULL REFERENCES dim_item(item_id),
    behavior_type INTEGER NOT NULL CHECK(behavior_type IN (1, 2, 3, 4)),
    event_time TEXT NOT NULL REFERENCES dim_time(event_time)
);

CREATE VIEW v_time_attributes AS
SELECT
    event_time,
    date(event_time) AS event_date,
    CAST(strftime('%Y', event_time) AS INTEGER) AS year,
    CAST(strftime('%m', event_time) AS INTEGER) AS month,
    CAST(strftime('%H', event_time) AS INTEGER) AS hour,
    (CAST(strftime('%w', event_time) AS INTEGER) + 6) % 7 + 1 AS weekday,
    CAST(strftime('%w', event_time) IN ('0', '6') AS INTEGER) AS is_weekend
FROM dim_time;
