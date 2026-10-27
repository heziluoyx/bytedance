"""新版清洗的聚合中间表、日级诊断标记和重复分布导出。"""

from pathlib import Path

import duckdb

AGGREGATES_SQL = """
CREATE TABLE user_summary AS
SELECT user_id, COUNT(*) AS event_count, COUNT(DISTINCT item_id) AS distinct_items,
       COUNT(DISTINCT CAST(timestamp AS DATE)) AS active_days,
       COUNT(*) FILTER(WHERE behavior_type=1) AS pv_count,
       COUNT(*) FILTER(WHERE behavior_type=2) AS fav_count,
       COUNT(*) FILTER(WHERE behavior_type=3) AS cart_count,
       COUNT(*) FILTER(WHERE behavior_type=4) AS buy_count,
       MIN(timestamp) AS first_time, MAX(timestamp) AS last_time
FROM retained_events GROUP BY user_id;
CREATE TABLE item_summary AS
SELECT item_id, COUNT(*) AS event_count, COUNT(DISTINCT user_id) AS distinct_users,
       COUNT(DISTINCT category_id) AS category_count,
       COUNT(*) FILTER(WHERE behavior_type=1) AS pv_count,
       COUNT(*) FILTER(WHERE behavior_type=2) AS fav_count,
       COUNT(*) FILTER(WHERE behavior_type=3) AS cart_count,
       COUNT(*) FILTER(WHERE behavior_type=4) AS buy_count
FROM retained_events GROUP BY item_id;
CREATE TABLE time_summary AS
SELECT timestamp, COUNT(*) AS event_count, COUNT(DISTINCT user_id) AS distinct_users,
       COUNT(*) FILTER(WHERE behavior_type=1) AS pv_count,
       COUNT(*) FILTER(WHERE behavior_type=2) AS fav_count,
       COUNT(*) FILTER(WHERE behavior_type=3) AS cart_count,
       COUNT(*) FILTER(WHERE behavior_type=4) AS buy_count
FROM retained_events GROUP BY timestamp;
CREATE TABLE item_category_conflicts AS
SELECT item_id,COUNT(DISTINCT category_id) AS category_count FROM events
GROUP BY item_id HAVING COUNT(DISTINCT category_id)>1;
CREATE TABLE user_daily AS
SELECT user_id,CAST(timestamp AS DATE) AS date,COUNT(*) AS event_count
FROM retained_events GROUP BY user_id,CAST(timestamp AS DATE);
"""
EXPORTS = {
    "user_summary": "SELECT * FROM user_summary",
    "item_summary": "SELECT * FROM item_summary",
    "time_summary": "SELECT * FROM time_summary",
    "item_category_conflicts": "SELECT * FROM item_category_conflicts",
    "user_daily_flags": "SELECT * FROM user_daily_flags",
    "hourly_profile": "SELECT * FROM assessed_hours",
    "removed_anomalies": "SELECT * FROM removed_anomalies",
    "repeated_operations": "SELECT * FROM repeat_keys WHERE repeat_count>1",
    "repeat_gap_hours": "SELECT * FROM repeat_intervals WHERE observed_gap_hours IS NOT NULL",
    "repetition_frequencies": (
        "SELECT repeat_count,COUNT(*) AS operation_groups,SUM(repeat_count) AS raw_rows, "
        + "SUM(repeat_count-1) AS excess_rows FROM repeat_keys GROUP BY repeat_count"
    ),
}


def export_outputs(connection: duckdb.DuckDBPyConnection, output: Path) -> dict:
    """重新聚合保留数据，并导出 Parquet；日级异常只标记不删除。

    Args:
        connection: 已含 retained_events 等诊断表的连接。
        output: 输出目录。

    Returns:
        日级标记阈值与三个维度的核对总量。
    """
    connection.execute(AGGREGATES_SQL)
    quartiles = connection.execute(
        "SELECT quantile_cont(event_count,.25),quantile_cont(event_count,.75), "
        + "AVG(event_count),STDDEV_POP(event_count) FROM user_daily"
    ).fetchone()
    iqr_upper = quartiles[1] + 1.5 * (quartiles[1] - quartiles[0])
    sigma_upper = quartiles[2] + 3 * quartiles[3]
    connection.execute(
        "CREATE TABLE user_daily_flags AS SELECT *,event_count > ? AS is_iqr_outlier, "
        + "event_count > ? AS is_sigma_outlier FROM user_daily",
        [iqr_upper, sigma_upper],
    )
    connection.execute(
        "CREATE VIEW cleaned_events AS SELECT e.*, d.is_iqr_outlier,d.is_sigma_outlier, "
        + "CASE behavior_type WHEN 1 THEN 'pv' WHEN 2 THEN 'fav' "
        + "WHEN 3 THEN 'cart' WHEN 4 THEN 'buy' END AS behavior_name "
        + "FROM retained_events e JOIN user_daily_flags d "
        + "ON e.user_id=d.user_id AND CAST(e.timestamp AS DATE)=d.date"
    )
    queries = {"cleaned_events": "SELECT * FROM cleaned_events", **EXPORTS}
    for name, query in queries.items():
        connection.sql(query).write_parquet(
            str(output / f"{name}.parquet"), compression="zstd"
        )
    aggregate_totals = {
        "user": connection.execute(
            "SELECT SUM(event_count) FROM user_summary"
        ).fetchone()[0],
        "item": connection.execute(
            "SELECT SUM(event_count) FROM item_summary"
        ).fetchone()[0],
        "time": connection.execute(
            "SELECT SUM(event_count) FROM time_summary"
        ).fetchone()[0],
    }
    return {
        "daily_flags": {
            "iqr_upper": iqr_upper,
            "sigma_upper": sigma_upper,
            "action": "flag_only",
        },
        "aggregate_totals": aggregate_totals,
    }
