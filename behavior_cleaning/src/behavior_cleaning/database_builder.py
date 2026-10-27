"""将清洗 Parquet 分批导入规范化 SQLite，公开 BuildConfig 和 build_database。"""

import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
INPUT_COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
INDEX_NAMES = {
    "uq_behavior_quadruple",
    "idx_behavior_user_time",
    "idx_behavior_item_type_user",
    "idx_behavior_type_time",
    "idx_behavior_time_user",
    "idx_item_category",
}
ITEM_CONFLICT_SQL = """
SELECT i.item_id FROM incoming_items AS i JOIN dim_item AS d USING(item_id)
WHERE i.category_id != d.category_id LIMIT 1
"""
EVENT_INSERT_SQL = """
INSERT INTO user_behavior(user_id, item_id, behavior_type, event_time)
VALUES (?, ?, ?, ?)
"""


@dataclass(frozen=True)
class BuildConfig:
    """数据库构建配置。

    Attributes:
        input_path: 清洗后的 Parquet 文件。
        database_path: 新 SQLite 文件，已有文件禁止覆盖。
        chunk_size: 每批读取的记录数。
    """

    input_path: Path
    database_path: Path
    chunk_size: int = 100_000

    def __post_init__(self) -> None:
        """拒绝非法分块大小。

        Raises:
            ValueError: chunk_size 不是正整数。
        """
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size 必须是正整数")


def _normalize_batch(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame[INPUT_COLUMNS].isna().any().any():
        raise ValueError("输入包含缺失字段，不是合格的清洗结果")
    for name in ("user_id", "item_id", "category_id"):
        # 维度 ID 必须已经是规范字符串，禁止通过转换悄悄修复脏输入。
        if not frame[name].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{name} 必须为字符串")
        valid = frame[name].str.fullmatch(r"[0-9]+", na=False)
        valid &= frame[name].str.contains(r"[1-9]", na=False)
        if not valid.all():
            raise ValueError(f"{name} 包含非法 ID")
    if not frame["behavior_type"].isin([1, 2, 3, 4]).all():
        raise ValueError("behavior_type 必须为整数 1、2、3、4")
    parsed = pd.to_datetime(frame["timestamp"], errors="raise")
    if parsed.dt.tz is not None:
        raise ValueError("输入时间必须为无时区的源数据本地时间")
    if not parsed.eq(parsed.dt.floor("h")).all():
        raise ValueError("输入必须是小时精度，禁止自动截断分钟和秒")
    frame["event_time"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["behavior_type"] = frame["behavior_type"].astype("int8")
    return frame


def _insert_batch(connection: sqlite3.Connection, frame: pd.DataFrame) -> None:
    items = frame[["item_id", "category_id"]].drop_duplicates()
    if items["item_id"].duplicated().any():
        raise ValueError("同一批次存在一个商品对应多个类目的冲突")
    with connection:
        connection.execute("DELETE FROM incoming_items")
        connection.executemany(
            "INSERT INTO incoming_items VALUES (?, ?)",
            items.itertuples(index=False, name=None),
        )
        conflict = connection.execute(ITEM_CONFLICT_SQL).fetchone()
        if conflict:
            raise ValueError(f"跨批次商品类目冲突，item_id={conflict[0]}")
        connection.executemany(
            "INSERT OR IGNORE INTO dim_user VALUES (?)",
            ((value,) for value in frame["user_id"].drop_duplicates()),
        )
        connection.execute(
            "INSERT OR IGNORE INTO dim_item SELECT * FROM incoming_items"
        )
        connection.executemany(
            "INSERT OR IGNORE INTO dim_time VALUES (?)",
            ((value,) for value in frame["event_time"].drop_duplicates()),
        )
        connection.executemany(
            EVENT_INSERT_SQL,
            frame[["user_id", "item_id", "behavior_type", "event_time"]].itertuples(
                index=False, name=None
            ),
        )


def _verify(
    connection: sqlite3.Connection,
    source: Path,
    imported_rows: int,
    expected_behaviors: Counter[int],
    preserve_repetitions: bool = False,
) -> dict:
    LOGGER.info("核对四张表的记录数量")
    counts = {
        "user_behavior": connection.execute(
            "SELECT COUNT(*) FROM user_behavior"
        ).fetchone()[0],
        "dim_user": connection.execute("SELECT COUNT(*) FROM dim_user").fetchone()[0],
        "dim_item": connection.execute("SELECT COUNT(*) FROM dim_item").fetchone()[0],
        "dim_time": connection.execute("SELECT COUNT(*) FROM dim_time").fetchone()[0],
    }
    if counts["user_behavior"] != imported_rows:
        raise ValueError("数据库行数与 Parquet 行数不一致")
    LOGGER.info("核对四类行为数量")
    actual_behaviors = dict(
        connection.execute(
            "SELECT behavior_type, COUNT(*) FROM user_behavior GROUP BY behavior_type"
        ).fetchall()
    )
    if actual_behaviors != dict(expected_behaviors):
        raise ValueError("行为类型数量与输入不一致")
    LOGGER.info("执行外键完整性检查")
    foreign_key_issue = connection.execute("PRAGMA foreign_key_check").fetchone()
    if foreign_key_issue is not None:
        raise ValueError(f"外键校验失败: {foreign_key_issue}")
    LOGGER.info("执行数据库全量 integrity_check")
    integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
    if integrity != ["ok"]:
        raise ValueError(f"数据库完整性校验失败: {integrity[:5]}")
    indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    expected_indexes = set(INDEX_NAMES)
    if preserve_repetitions:
        expected_indexes.remove("uq_behavior_quadruple")
        expected_indexes.add("idx_behavior_quadruple")
    if not expected_indexes.issubset(indexes):
        raise ValueError("索引缺失")
    LOGGER.info("交叉核对清洗报告、聚合表与索引")
    # 若同目录存在清洗报告和聚合表，则与前一阶段结果独立交叉核对。
    cross_checks: dict[str, bool] = {}
    report_path = source.parent / "quality_report.json"
    if report_path.exists():
        clean_report = json.loads(report_path.read_text(encoding="utf-8"))
        cross_checks["cleaning_report_rows"] = (
            clean_report["rows"]["clean"] == imported_rows
        )
        cross_checks["cleaning_report_behaviors"] = {
            int(key): value for key, value in clean_report["behavior_counts"].items()
        } == actual_behaviors
    for filename, table in (
        ("user_summary.parquet", "dim_user"),
        ("item_summary.parquet", "dim_item"),
        ("time_summary.parquet", "dim_time"),
    ):
        path = source.parent / filename
        if path.exists():
            cross_checks[filename] = (
                pq.ParquetFile(path).metadata.num_rows == counts[table]
            )
    if not all(cross_checks.values()):
        raise ValueError(f"与清洗阶段交叉核对失败: {cross_checks}")
    return {
        "table_rows": counts,
        "behavior_counts": actual_behaviors,
        "foreign_key_check": "passed",
        "integrity_check": "ok",
        "unique_quadruple": (
            "not_required_preserve_regular_repetitions"
            if preserve_repetitions
            else "enforced_by_unique_index"
        ),
        "indexes": sorted(expected_indexes),
        "cross_checks": cross_checks,
        "sqlite_version": sqlite3.sqlite_version,
    }


def _create_indexes(connection: sqlite3.Connection, sql_text: str) -> None:
    # 仅分解项目内可信的 SQL 模板，不接受用户输入的 SQL。
    pending: list[str] = []
    statements: list[str] = []
    for line in sql_text.splitlines(keepends=True):
        pending.append(line)
        statement = "".join(pending)
        if sqlite3.complete_statement(statement):
            statements.append(statement)
            pending.clear()
    if "".join(pending).strip():
        raise ValueError("索引模板中存在未结束的 SQL 语句")
    for statement in statements:
        match = re.search(r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)", statement, re.I)
        label = match.group(1) if match else "ANALYZE"
        if (
            match
            and connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (label,)
            ).fetchone()
        ):
            LOGGER.info("索引已存在，继续下一项：%s", label)
            continue
        LOGGER.info("开始：%s", label)
        started = perf_counter()
        connection.execute(statement)
        connection.commit()
        LOGGER.info("完成：%s，耗时 %.1f 秒", label, perf_counter() - started)


def build_database(config: BuildConfig) -> Path:
    """分批建库、建立索引、校验后发布数据库，已有数据库不覆盖。

    Args:
        config: 输入文件、输出数据库和分块配置。

    Returns:
        校验成功后的 SQLite 文件绝对路径。

    Raises:
        FileNotFoundError: 输入文件或 SQL 模板不存在。
        FileExistsError: 数据库、验证报告或构建锁已经存在。
        ValueError: 输入数据不合法或校验失败。
        sqlite3.Error: 数据库写入、约束或索引创建失败。
        OSError: 文件操作失败。
    """
    source = config.input_path.absolute()
    target = config.database_path.absolute()
    if not source.is_file():
        raise FileNotFoundError(source)
    report_path = target.with_suffix(".validation.json")
    if target.exists() or report_path.exists():
        raise FileExistsError(f"目标已存在，禁止覆盖：{target}")
    parquet = pq.ParquetFile(source)
    if not set(INPUT_COLUMNS).issubset(parquet.schema_arrow.names):
        raise ValueError("Parquet 缺少必要字段")
    if parquet.metadata.num_rows == 0:
        raise ValueError("输入 Parquet 为空")
    sql_dir = Path(__file__).absolute().parents[2] / "sql"
    schema_sql = (sql_dir / "01_schema.sql").read_text("utf-8")
    clean_report_path = source.parent / "quality_report.json"
    clean_report = (
        json.loads(clean_report_path.read_text("utf-8"))
        if clean_report_path.exists()
        else {}
    )
    preserve_repetitions = (
        clean_report.get("configuration", {}).get("duplicate_policy")
        == "preserve_regular_repetitions"
    )
    index_file = "02_indexes_preserve.sql" if preserve_repetitions else "02_indexes.sql"
    indexes_sql = (sql_dir / index_file).read_text("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(".build.lock")
    # 排他创建锁，避免两次构建竞争同一输出；失败的临时库保留供排查。
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(datetime.now().astimezone().isoformat())
    temporary = target.with_name(f"{target.stem}.building_{uuid4().hex[:8]}.sqlite")
    connection = None
    started = perf_counter()
    try:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("PRAGMA mmap_size=536870912")
        connection.execute("PRAGMA threads=2")
        connection.executescript(schema_sql)
        connection.execute(
            "CREATE TEMP TABLE incoming_items(item_id TEXT, category_id TEXT)"
        )
        imported_rows = 0
        behaviors: Counter[int] = Counter()
        for batch in parquet.iter_batches(
            batch_size=config.chunk_size, columns=INPUT_COLUMNS
        ):
            frame = _normalize_batch(batch.to_pandas())
            _insert_batch(connection, frame)
            behaviors.update(
                {
                    int(key): int(value)
                    for key, value in frame.behavior_type.value_counts().items()
                }
            )
            imported_rows += len(frame)
            LOGGER.info("已导入 %s / %s 行", imported_rows, parquet.metadata.num_rows)
        if imported_rows != parquet.metadata.num_rows:
            raise ValueError("实际读取行数与 Parquet 元数据不一致")
        LOGGER.info("导入完成，开始创建索引和 ANALYZE")
        _create_indexes(connection, indexes_sql)
        LOGGER.info("索引完成，开始核验外键、完整性和数量")
        report = _verify(
            connection, source, imported_rows, behaviors, preserve_repetitions
        )
        report.update(
            {
                "status": "success",
                "source": str(source),
                "source_sha256": digest.hexdigest(),
                "database": str(target),
                "chunk_size": config.chunk_size,
                "created_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": round(perf_counter() - started, 2),
                "timestamp_format": "YYYY-MM-DD HH:00:00, source local time",
                "scope": "steps_1_to_5_no_basic_analysis",
            }
        )
        connection.close()
        connection = None
        # 使用硬链接排他发布，已有同名文件时不会被覆盖。
        target.hardlink_to(temporary)
        temporary.unlink()
        report["database_bytes"] = target.stat().st_size
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        LOGGER.info("数据库与校验报告已生成：%s", target)
        return target
    finally:
        if connection is not None:
            connection.close()
        lock_path.unlink()
