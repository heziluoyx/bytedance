"""磁盘去重、质量审计与 Parquet 导出。SQL 语句仅使用固定结构。"""

import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_SQL = """
CREATE TABLE events (
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    category_id TEXT NOT NULL,
    behavior_type INTEGER NOT NULL CHECK(behavior_type BETWEEN 1 AND 4),
    timestamp TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    PRIMARY KEY(user_id, item_id, behavior_type, timestamp)
) WITHOUT ROWID;
CREATE TABLE duplicates (
    user_id TEXT, item_id TEXT, category_id TEXT, behavior_type INTEGER,
    timestamp TEXT, source_file TEXT, source_row INTEGER,
    kept_category_id TEXT, kept_source_file TEXT, kept_source_row INTEGER
);
"""
INSERT_SQL = """
INSERT OR IGNORE INTO events
SELECT user_id, item_id, category_id, behavior_type, timestamp,
       source_file, source_row FROM incoming ORDER BY source_row
"""
DUPLICATES_SQL = """
INSERT INTO duplicates
SELECT i.*, e.category_id, e.source_file, e.source_row
FROM incoming AS i JOIN events AS e
USING(user_id, item_id, behavior_type, timestamp)
WHERE i.source_file != e.source_file OR i.source_row != e.source_row
"""
AGGREGATE_SQL = """
CREATE TABLE user_daily AS
SELECT user_id, substr(timestamp, 1, 10) AS date, COUNT(*) AS event_count
FROM events GROUP BY user_id, substr(timestamp, 1, 10);
CREATE TABLE user_summary AS
SELECT user_id, COUNT(*) AS event_count,
       COUNT(DISTINCT item_id) AS distinct_items,
       COUNT(DISTINCT substr(timestamp, 1, 10)) AS active_days,
       SUM(behavior_type = 1) AS pv_count,
       SUM(behavior_type = 2) AS fav_count,
       SUM(behavior_type = 3) AS cart_count,
       SUM(behavior_type = 4) AS buy_count,
       MIN(timestamp) AS first_time, MAX(timestamp) AS last_time
FROM events GROUP BY user_id;
CREATE TABLE item_summary AS
SELECT item_id, COUNT(*) AS event_count,
       COUNT(DISTINCT user_id) AS distinct_users,
       COUNT(DISTINCT category_id) AS category_count,
       SUM(behavior_type = 1) AS pv_count,
       SUM(behavior_type = 2) AS fav_count,
       SUM(behavior_type = 3) AS cart_count,
       SUM(behavior_type = 4) AS buy_count
FROM events GROUP BY item_id;
CREATE TABLE item_category_conflicts AS
SELECT item_id, COUNT(DISTINCT category_id) AS category_count
FROM (
    SELECT item_id, category_id FROM events
    UNION ALL
    SELECT item_id, category_id FROM duplicates
)
GROUP BY item_id HAVING COUNT(DISTINCT category_id) > 1;
CREATE TABLE time_summary AS
SELECT timestamp, COUNT(*) AS event_count,
       COUNT(DISTINCT user_id) AS distinct_users,
       SUM(behavior_type = 1) AS pv_count,
       SUM(behavior_type = 2) AS fav_count,
       SUM(behavior_type = 3) AS cart_count,
       SUM(behavior_type = 4) AS buy_count
FROM events GROUP BY timestamp;
"""
EXPORT_QUERIES = {
    "user_summary": "SELECT * FROM user_summary",
    "item_summary": "SELECT * FROM item_summary",
    "time_summary": "SELECT * FROM time_summary",
    "item_category_conflicts": "SELECT * FROM item_category_conflicts",
    "duplicates": "SELECT * FROM duplicates",
}
EVENT_QUERY = """
SELECT e.user_id, e.item_id, e.category_id, e.behavior_type, e.timestamp,
       e.source_file, e.source_row,
       d.is_iqr_outlier, d.is_sigma_outlier
FROM events AS e JOIN daily_flags AS d
ON e.user_id = d.user_id AND substr(e.timestamp, 1, 10) = d.date
"""


def insert_chunk(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
) -> int:
    """将一个有效块写入磁盘数据库并全局去重。

    Args:
        connection: 已初始化的 SQLite 连接。
        frame: 按标准列顺序排列的有效数据，不会被修改。

    Returns:
        本块移除的重复行数。稳定保留输入顺序的第一条。
    """
    frame.to_sql("incoming", connection, if_exists="replace", index=False)
    with connection:
        before = connection.total_changes
        connection.execute(INSERT_SQL)
        inserted = connection.total_changes - before
        connection.execute(DUPLICATES_SQL)
        connection.execute("DROP TABLE incoming")
    return len(frame) - inserted


def build_daily_flags(
    connection: sqlite3.Connection,
) -> dict[str, float | int | str]:
    """计算用户日行为量的上尾 IQR 和 3σ 标记，不删除记录。

    Args:
        connection: 已构建 user_daily 表的连接。

    Returns:
        全量频数分布计算的阈值、均值和总体标准差。
    """
    distribution = pd.read_sql_query(
        "SELECT event_count, COUNT(*) AS frequency FROM user_daily "
        + "GROUP BY event_count ORDER BY event_count",
        connection,
    )
    values = distribution["event_count"].to_numpy(dtype="float64")
    weights = distribution["frequency"].to_numpy(dtype="int64")
    cumulative = weights.cumsum()
    total = int(weights.sum())

    def _quantile(probability: float) -> float:
        position = (total - 1) * probability
        lower = int(position)
        upper = min(lower + 1, total - 1)
        left = values[cumulative.searchsorted(lower + 1)]
        right = values[cumulative.searchsorted(upper + 1)]
        return float(left + (right - left) * (position - lower))

    first_quartile = _quantile(0.25)
    third_quartile = _quantile(0.75)
    mean = float((values * weights).sum() / total)
    variance = float((((values - mean) ** 2) * weights).sum() / total)
    sigma = variance**0.5
    iqr_upper = third_quartile + 1.5 * (third_quartile - first_quartile)
    sigma_upper = mean + 3 * sigma
    connection.execute(
        "CREATE TABLE daily_flags AS SELECT *, "
        + "event_count > ? AS is_iqr_outlier, "
        + "event_count > ? AS is_sigma_outlier FROM user_daily",
        (iqr_upper, sigma_upper),
    )
    connection.execute("CREATE UNIQUE INDEX daily_key ON daily_flags(user_id, date)")
    connection.commit()
    return {
        "metric": "deduplicated_user_daily_event_count",
        "sample_count": total,
        "q1": first_quartile,
        "q3": third_quartile,
        "mean": mean,
        "population_std": sigma,
        "iqr_upper": iqr_upper,
        "sigma_upper": sigma_upper,
        "action": "flag_only",
    }


def export_query(
    connection: sqlite3.Connection,
    query: str,
    path: Path,
    chunk_size: int,
) -> int:
    """分批导出内部固定 SQL 查询为压缩 Parquet。

    Args:
        connection: 数据库连接。
        query: 仅接受包内定义的查询，不传入外部输入。
        path: 输出文件路径。
        chunk_size: 每批行数。

    Returns:
        导出行数，包括空结果时生成的零行文件。
    """
    writer = None
    exported = 0
    try:
        for frame in pd.read_sql_query(query, connection, chunksize=chunk_size):
            for column in frame.columns:
                if column in {"timestamp", "first_time", "last_time"}:
                    frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d %H")
                elif column.startswith("is_"):
                    frame[column] = frame[column].astype("bool")
                elif column == "behavior_type":
                    frame[column] = frame[column].astype("int8")
                elif (
                    column.endswith("_id")
                    or column.endswith("source_file")
                    or column == "date"
                ):
                    frame[column] = frame[column].astype("string")
                else:
                    frame[column] = frame[column].astype("int64")
            if "behavior_type" in frame:
                frame["behavior_name"] = pd.Categorical(
                    frame["behavior_type"].map(
                        {1: "pv", 2: "fav", 3: "cart", 4: "buy"}
                    ),
                    categories=["pv", "fav", "cart", "buy"],
                )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    path, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table)
            exported += len(frame)
    finally:
        if writer is not None:
            writer.close()
    return exported
