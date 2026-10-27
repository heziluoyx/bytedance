"""对已完整导入、在索引阶段中断的临时 SQLite 续建并核验。"""

import hashlib
import json
import logging
import os
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

from behavior_cleaning.database_builder import BuildConfig, _create_indexes, _verify

LOGGER = logging.getLogger(__name__)


def finish_database(config: BuildConfig, temporary: Path) -> Path:
    """续建已完全导入的临时库，校验后发布，禁止重写现有正式库。

    Args:
        config: 与原构建相同的输入和输出配置。
        temporary: 原构建生成的同目录 building 临时库。

    Returns:
        核验成功的正式数据库路径。

    Raises:
        ValueError: 临时路径不合法或导入行数未完整匹配。
        FileExistsError: 正式库、报告或锁已存在。
        sqlite3.Error: 索引或完整性核验失败。
        OSError: 文件操作失败。
    """
    source = config.input_path.absolute()
    target = config.database_path.absolute()
    temporary = temporary.absolute()
    if (
        temporary.parent != target.parent
        or not temporary.name.startswith(f"{target.stem}.building_")
        or temporary.suffix != ".sqlite"
    ):
        raise ValueError("仅允许续建目标目录中的指定 building 临时库")
    if not temporary.is_file() or not source.is_file():
        raise FileNotFoundError("输入或临时库不存在")
    report_path = target.with_suffix(".validation.json")
    if target.exists() or report_path.exists():
        raise FileExistsError("正式库或校验报告已存在，禁止覆盖")
    quality_path = source.parent / "quality_report.json"
    quality = (
        json.loads(quality_path.read_text("utf-8")) if quality_path.exists() else {}
    )
    preserve = (
        quality.get("configuration", {}).get("duplicate_policy")
        == "preserve_regular_repetitions"
    )
    expected = Counter(
        {int(key): value for key, value in quality.get("behavior_counts", {}).items()}
    )
    parquet = pq.ParquetFile(source)
    if not expected:
        for batch in parquet.iter_batches(
            columns=["behavior_type"], batch_size=config.chunk_size
        ):
            expected.update(int(value) for value in batch.column(0).to_pylist())
    lock_path = target.with_suffix(".build.lock")
    with lock_path.open("x", encoding="utf-8") as stream:
        stream.write(datetime.now().astimezone().isoformat())
    connection = None
    started = perf_counter()
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("PRAGMA mmap_size=536870912")
        connection.execute("PRAGMA threads=2")
        count = connection.execute("SELECT COUNT(*) FROM user_behavior").fetchone()[0]
        if count != parquet.metadata.num_rows:
            raise ValueError("临时库未完整导入，不允许跳过导入继续建库")
        actual = dict(
            connection.execute(
                "SELECT behavior_type,COUNT(*) FROM user_behavior GROUP BY behavior_type"
            ).fetchall()
        )
        if actual != dict(expected):
            raise ValueError("临时库行为数量与指定输入不一致")
        LOGGER.info("已确认完整导入 %s 行，从剩余索引继续", count)
        sql_dir = Path(__file__).absolute().parents[2] / "sql"
        index_name = "02_indexes_preserve.sql" if preserve else "02_indexes.sql"
        _create_indexes(connection, (sql_dir / index_name).read_text("utf-8"))
        report = _verify(connection, source, count, expected, preserve)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        report.update(
            {
                "status": "success",
                "source": str(source),
                "source_sha256": digest.hexdigest(),
                "database": str(target),
                "resumed_from": str(temporary),
                "created_at": datetime.now().astimezone().isoformat(),
                "resume_elapsed_seconds": round(perf_counter() - started, 2),
                "scope": "steps_1_to_5_no_basic_analysis",
                "temporary_directory": os.environ.get("TMP"),
                "chunk_size": config.chunk_size,
                "timestamp_format": "YYYY-MM-DD HH:00:00, source local time",
            }
        )
        connection.close()
        connection = None
        target.hardlink_to(temporary)
        temporary.unlink()
        report["database_bytes"] = target.stat().st_size
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        LOGGER.info("续建和核验完成：%s", target)
        return target
    finally:
        if connection is not None:
            connection.close()
        lock_path.unlink()
