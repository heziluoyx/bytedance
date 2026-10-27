"""原始行为的重复频数、用户小时量与跨小时重复间隔诊断。"""

import duckdb

PROFILE_SQL = """
CREATE TABLE repeat_keys AS
SELECT user_id, item_id, behavior_type, timestamp,
       COUNT(*)::BIGINT AS repeat_count
FROM events GROUP BY user_id, item_id, behavior_type, timestamp;
CREATE TABLE hourly_profile AS
SELECT user_id, timestamp, SUM(repeat_count)::BIGINT AS event_count,
       COUNT(*)::BIGINT AS unique_operations,
       MAX(repeat_count)::BIGINT AS max_same_operation,
       SUM(repeat_count - 1)::BIGINT AS repeat_excess,
       SUM(repeat_count - 1) * 1.0 / SUM(repeat_count) AS repeat_ratio
FROM repeat_keys GROUP BY user_id, timestamp;
CREATE TABLE repeat_intervals AS
SELECT user_id, item_id, behavior_type, timestamp, repeat_count,
       date_diff('hour', LAG(timestamp) OVER (
           PARTITION BY user_id, item_id, behavior_type ORDER BY timestamp
       ), timestamp)::BIGINT AS observed_gap_hours
FROM repeat_keys;
"""


def create_profiles(connection: duckdb.DuckDBPyConnection) -> dict:
    """构建全局诊断表，不删除任何事件。

    Args:
        connection: 已含 events 表的 DuckDB 连接，时间列为 TIMESTAMP。

    Returns:
        全量频数分布与重复间隔摘要。小时桶相同不代表实际时间差为零。
    """
    connection.execute(PROFILE_SQL)
    hourly = connection.execute(
        "SELECT COUNT(*), SUM(event_count), MIN(event_count), MAX(event_count), "
        + "AVG(event_count), STDDEV_POP(event_count), "
        + "quantile_cont(event_count, [0.25,0.5,0.75,0.9,0.95,0.99,0.995,0.999,0.9999]), "
        + "quantile_cont(max_same_operation, [0.25,0.5,0.75,0.99,0.999,0.9999]), "
        + "MAX(max_same_operation) FROM hourly_profile"
    ).fetchone()
    repetitions = connection.execute(
        "SELECT COUNT(*), SUM(repeat_count - 1), MAX(repeat_count), "
        + "COUNT(*) FILTER(WHERE repeat_count > 1) FROM repeat_keys"
    ).fetchone()
    gaps = connection.execute(
        "SELECT COUNT(*), MIN(observed_gap_hours), MAX(observed_gap_hours), "
        + "quantile_cont(observed_gap_hours,[0.25,0.5,0.75,0.9,0.99]) "
        + "FROM repeat_intervals WHERE observed_gap_hours IS NOT NULL"
    ).fetchone()
    candidates = connection.execute(
        "SELECT user_id, CAST(timestamp AS VARCHAR), event_count, unique_operations, "
        + "max_same_operation, repeat_ratio FROM hourly_profile "
        + "ORDER BY event_count DESC, user_id, timestamp LIMIT 30"
    ).fetchall()
    return {
        "hourly": {
            "user_hours": hourly[0],
            "raw_events": hourly[1],
            "min": hourly[2],
            "max": hourly[3],
            "mean": hourly[4],
            "std": hourly[5],
            "quantile_probabilities": [
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
                0.995,
                0.999,
                0.9999,
            ],
            "quantiles": hourly[6],
            "max_same_operation_quantile_probabilities": [
                0.25,
                0.5,
                0.75,
                0.99,
                0.999,
                0.9999,
            ],
            "max_same_operation_quantiles": hourly[7],
            "max_same_operation": hourly[8],
        },
        "repetitions": {
            "unique_quadruples": repetitions[0],
            "excess_records": repetitions[1],
            "max_frequency": repetitions[2],
            "repeated_quadruples": repetitions[3],
        },
        "different_hour_gaps": {
            "pairs": gaps[0],
            "min_hours": gaps[1],
            "max_hours": gaps[2],
            "quantiles": gaps[3],
            "note": "Distinct observed hour bins; same-hour actual gaps are unknown.",
        },
        "largest_user_hours": candidates,
    }
