"""执行分块、并行校验、磁盘去重和质量报告，导出 run_pipeline。"""

import csv
import hashlib
import json
import logging
import sqlite3
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import pandas as pd

from behavior_cleaning.cleaning import (
    ALIASES,
    REQUIRED_COLUMNS,
    CleanResult,
    Config,
    clean_chunk,
)
from behavior_cleaning.storage import (
    AGGREGATE_SQL,
    EVENT_QUERY,
    EXPORT_QUERIES,
    SCHEMA_SQL,
    build_daily_flags,
    export_query,
    insert_chunk,
)

LOGGER = logging.getLogger(__name__)


class SourceInfo(TypedDict):
    """输入文件的可追溯元数据。"""

    path: str
    bytes: int
    sha256: str
    columns: list[str]
    ignored_columns: list[str]


def _discover_files(config: Config) -> list[Path]:
    source = config.input_path.resolve(strict=True)
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError("输入文件必须是 CSV")
        return [source]
    output_root = config.output_root.resolve()
    files = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
        and output_root not in path.resolve().parents
    )
    if not files:
        raise FileNotFoundError(f"未找到 CSV: {source}")
    return files


def _inspect_source(path: Path, encoding: str) -> SourceInfo:
    with path.open("r", encoding=encoding, newline="") as stream:
        header = next(csv.reader(stream), [])
    normalized = [ALIASES.get(name.strip(), name.strip()) for name in header]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path}: 表头重名或别名冲突")
    if not set(REQUIRED_COLUMNS).issubset(normalized):
        raise ValueError(f"{path}: 表头缺少标准字段")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "columns": header,
        "ignored_columns": sorted(set(normalized) - set(REQUIRED_COLUMNS)),
    }


def _iter_clean_results(files: list[Path], config: Config) -> Iterator[CleanResult]:
    # 限制在途块数，避免并行调度提前读取整个 CSV。
    pending: deque[Future[CleanResult]] = deque()
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for path in files:
            offset = 0
            with pd.read_csv(
                path,
                dtype="string",
                chunksize=config.chunk_size,
                encoding=config.encoding,
                keep_default_na=False,
                skip_blank_lines=False,
                on_bad_lines="error",
            ) as reader:
                for chunk in reader:
                    if not isinstance(chunk.index, pd.RangeIndex):
                        raise ValueError(f"{path}: CSV 数据列多于表头，禁止隐式索引")
                    pending.append(
                        executor.submit(clean_chunk, chunk, str(path), offset, config)
                    )
                    offset += len(chunk)
                    if len(pending) >= config.workers:
                        yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def run_pipeline(config: Config) -> Path:
    """运行完整清洗流程，原始数据不会被修改。

    Args:
        config: 输入、输出、资源及时间规则配置。

    Returns:
        成功运行的独立输出目录。存在 SUCCESS 文件才代表全部完成。

    Raises:
        FileNotFoundError: 找不到输入。
        ValueError: 数据结构非法、没有有效数据或质量检查失败。
        pandas.errors.ParserError: CSV 结构破损，停止而不静默跳行。
        OSError: 读写失败。
    """
    files = _discover_files(config)
    sources = [_inspect_source(path, config.encoding) for path in files]
    run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output = config.output_root.resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "cleaning.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        _execute(config, sources, files, output)
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    return output


def _execute(
    config: Config,
    sources: list[SourceInfo],
    files: list[Path],
    output: Path,
) -> None:
    input_rows = 0
    invalid_rows = 0
    duplicate_rows = 0
    missing_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    connection = sqlite3.connect(output / "audit.sqlite")
    try:
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        connection.executescript(SCHEMA_SQL)
        with (output / "rejected.jsonl").open("w", encoding="utf-8") as rejected_file:
            for result in _iter_clean_results(files, config):
                input_rows += result.input_rows
                invalid_rows += len(result.rejected)
                missing_counts.update(result.missing_counts)
                if not result.rejected.empty:
                    result.rejected.to_json(
                        rejected_file, orient="records", lines=True, force_ascii=False
                    )
                    for reason in result.rejected["reject_reason"]:
                        reason_counts.update(reason.split(";"))
                if not result.accepted.empty:
                    duplicate_rows += insert_chunk(connection, result.accepted)
                LOGGER.info(
                    "已读取 %s 行，规则拒绝 %s 行，重复 %s 行",
                    input_rows,
                    invalid_rows,
                    duplicate_rows,
                )
        clean_rows = int(
            connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )
        if clean_rows == 0:
            raise ValueError("没有有效记录，详见 rejected.jsonl 和 cleaning.log")
        if input_rows != clean_rows + invalid_rows + duplicate_rows:
            raise ValueError("输入与输出行数无法核对")
        connection.executescript(AGGREGATE_SQL)
        outlier_rules = build_daily_flags(connection)
        exported_rows = export_query(
            connection,
            EVENT_QUERY,
            output / "cleaned_events.parquet",
            config.chunk_size,
        )
        for name, query in EXPORT_QUERIES.items():
            export_query(
                connection, query, output / f"{name}.parquet", config.chunk_size
            )
        export_query(
            connection,
            "SELECT * FROM daily_flags",
            output / "user_daily_flags.parquet",
            config.chunk_size,
        )
        totals = connection.execute(
            "SELECT SUM(event_count) FROM user_summary"
        ).fetchone()[0]
        if exported_rows != clean_rows or totals != clean_rows:
            raise ValueError("Parquet 导出行数或用户聚合总数校验失败")
        conflicts = connection.execute(
            "SELECT COUNT(*) FROM item_category_conflicts"
        ).fetchone()[0]
        category_disagreements = connection.execute(
            "SELECT COUNT(*) FROM duplicates WHERE category_id != kept_category_id"
        ).fetchone()[0]
        time_range = connection.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM events"
        ).fetchone()
        behavior_counts = dict(
            connection.execute(
                "SELECT behavior_type, COUNT(*) FROM events GROUP BY behavior_type"
            ).fetchall()
        )
        flagged_days = connection.execute(
            "SELECT SUM(is_iqr_outlier), SUM(is_sigma_outlier) FROM daily_flags"
        ).fetchone()
        report = {
            "status": "success",
            "created_at": datetime.now().astimezone().isoformat(),
            "sources": sources,
            "configuration": {
                "chunk_size": config.chunk_size,
                "workers": config.workers,
                "encoding": config.encoding,
                "start_inclusive": config.start,
                "end_inclusive": config.end,
                "timestamp_semantics": "source_local_hour_no_timezone_conversion",
                "dedup_key": ["user_id", "item_id", "behavior_type", "timestamp"],
                "dedup_policy": "first_in_sorted_file_order_then_source_row",
                "id_policy": "positive_digit_string_preserve_leading_zeros",
            },
            "rows": {
                "input": input_rows,
                "invalid": invalid_rows,
                "duplicates": duplicate_rows,
                "clean": clean_rows,
                "retained_ratio": clean_rows / input_rows,
            },
            "missing_by_field_before_cleaning": dict(missing_counts),
            "rejected_by_reason_nonexclusive": dict(reason_counts),
            "category_conflicting_items": conflicts,
            "duplicate_category_disagreements": category_disagreements,
            "time_min": time_range[0],
            "time_max": time_range[1],
            "behavior_counts": behavior_counts,
            "outlier_rules": outlier_rules,
            "outlier_user_days": {"iqr": flagged_days[0], "sigma": flagged_days[1]},
            "validation": {
                "row_balance": input_rows == clean_rows + invalid_rows + duplicate_rows,
                "unique_quadruple": True,
                "required_values_valid": True,
                "parquet_rows_match": exported_rows == clean_rows,
                "user_aggregate_matches": totals == clean_rows,
                "category_consistency": conflicts == 0,
            },
            "warnings": [
                "类别冲突只报告，不臆造正确类别；重复记录保留第一条并保存审计明细。",
                "时间仅精确到小时；同小时的真实重复交互也可能被指定四元组规则合并。",
                "IQR/3σ 仅标记高行为量用户日，不自动删除高活跃用户。",
            ],
        }
        with (output / "quality_report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        (output / "SUCCESS").write_text("完成所有导出和质量核对\n", encoding="utf-8")
        LOGGER.info("清洗完成：%s", output)
    except (OSError, ValueError, sqlite3.Error, pd.errors.ParserError):
        LOGGER.exception("清洗未完成，当前目录不能作为成功结果使用")
        raise
    finally:
        connection.close()
