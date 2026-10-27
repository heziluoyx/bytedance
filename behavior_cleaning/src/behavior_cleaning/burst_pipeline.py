"""保留正常重复的新版清洗流程，导出 run_burst_pipeline。"""

import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq

from behavior_cleaning.burst_outputs import export_outputs
from behavior_cleaning.burst_profile import create_profiles
from behavior_cleaning.burst_rules import (
    BurstRules,
    assess_hours,
    retain_regular_repetitions,
)
from behavior_cleaning.cleaning import Config
from behavior_cleaning.pipeline import (
    _discover_files,
    _inspect_source,
    _iter_clean_results,
)

LOGGER = logging.getLogger(__name__)
EVENT_SCHEMA = """
CREATE TABLE events (
    user_id VARCHAR, item_id VARCHAR, category_id VARCHAR, behavior_type TINYINT,
    timestamp TIMESTAMP, source_file VARCHAR, source_row BIGINT
)
"""
INSERT_EVENTS = """
INSERT INTO events SELECT user_id,item_id,category_id,behavior_type,
strptime(timestamp,'%Y-%m-%d %H'),source_file,source_row FROM incoming_chunk
"""


def run_burst_pipeline(config: Config, rules: BurstRules | None = None) -> Path:
    """从原始 CSV 重跑清洗，仅压缩异常小时的重复冗余，原结果不覆盖。

    Args:
        config: 输入、输出、分块及时间规则配置。
        rules: 统计异常参数，省略时使用默认稳健规则。

    Returns:
        包含 SUCCESS 的独立输出目录。

    Raises:
        ValueError: 没有有效数据或输出核对失败。
        FileNotFoundError: 输入不存在。
        duckdb.Error: 分析引擎发生错误。
        OSError: 文件读写失败。
    """
    rules = rules or BurstRules()
    files = _discover_files(config)
    sources = [_inspect_source(path, config.encoding) for path in files]
    output = config.output_root.absolute() / (
        f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "cleaning.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    connection = duckdb.connect(str(output / "audit.duckdb"))
    try:
        connection.execute("SET memory_limit='1GB'")
        connection.execute("SET threads=2")
        connection.execute(EVENT_SCHEMA)
        input_rows = 0
        invalid_rows = 0
        missing: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        with (output / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for result in _iter_clean_results(files, config):
                input_rows += result.input_rows
                invalid_rows += len(result.rejected)
                missing.update(result.missing_counts)
                if not result.rejected.empty:
                    result.rejected.to_json(
                        rejected, orient="records", lines=True, force_ascii=False
                    )
                    for reason in result.rejected["reject_reason"]:
                        reasons.update(reason.split(";"))
                if not result.accepted.empty:
                    connection.register("incoming_chunk", result.accepted)
                    try:
                        connection.execute(INSERT_EVENTS)
                    finally:
                        connection.unregister("incoming_chunk")
                LOGGER.info("已校验并保留原始频次 %s 行，规则拒绝 %s 行", input_rows, invalid_rows)
        if input_rows == invalid_rows:
            raise ValueError("没有有效记录，详见 rejected.jsonl")
        LOGGER.info("开始统计全量重复频数、用户小时量与跨小时重复间隔")
        profile = create_profiles(connection)
        limits = assess_hours(connection, rules)
        LOGGER.info(
            "小时候选阈值 > %s；重复强度阈值 > %s", limits["hour_limit"], limits["repeat_limit"]
        )
        retain_regular_repetitions(connection)
        removed = connection.execute(
            "SELECT COUNT(*) FROM removed_anomalies"
        ).fetchone()[0]
        clean = connection.execute("SELECT COUNT(*) FROM retained_events").fetchone()[0]
        LOGGER.info("仅隔离异常小时的重复冗余 %s 行，保留 %s 行", removed, clean)
        aggregates = export_outputs(connection, output)
        sensitivity = []
        hour_quantiles = profile["hourly"]["quantiles"]
        for threshold in sorted(
            {
                100,
                math.ceil(hour_quantiles[5]),
                math.ceil(hour_quantiles[6]),
                limits["hour_limit"],
                math.ceil(hour_quantiles[-1]),
            }
        ):
            counts = connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(event_count),0),COALESCE(SUM(repeat_excess),0) "
                + "FROM assessed_hours WHERE event_count > ?",
                [threshold],
            ).fetchone()
            sensitivity.append(
                {
                    "reference_threshold": threshold,
                    "candidate_hours": counts[0],
                    "raw_rows_in_candidates": counts[1],
                    "repeat_excess_in_candidates": counts[2],
                }
            )
        hours = connection.execute(
            "SELECT COUNT(*) FILTER(WHERE is_high_volume_candidate), "
            + "COUNT(*) FILTER(WHERE is_burst_anomaly), "
            + "COUNT(DISTINCT user_id) FILTER(WHERE is_burst_anomaly) FROM assessed_hours"
        ).fetchone()
        behavior_counts = dict(
            connection.execute(
                "SELECT behavior_type,COUNT(*) FROM retained_events GROUP BY behavior_type"
            ).fetchall()
        )
        removed_behaviors = dict(
            connection.execute(
                "SELECT behavior_type,COUNT(*) FROM removed_anomalies GROUP BY behavior_type"
            ).fetchall()
        )
        conflicts = connection.execute(
            "SELECT COUNT(*) FROM item_category_conflicts"
        ).fetchone()[0]
        retained_keys = connection.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT user_id,item_id,behavior_type,timestamp FROM retained_events)"
        ).fetchone()[0]
        parquet_rows = pq.ParquetFile(
            output / "cleaned_events.parquet"
        ).metadata.num_rows
        validations = {
            "row_balance": input_rows == invalid_rows + removed + clean,
            "parquet_rows_match": clean == parquet_rows,
            "aggregate_totals_match": all(
                total == clean for total in aggregates["aggregate_totals"].values()
            ),
            "all_distinct_operations_preserved": retained_keys
            == profile["repetitions"]["unique_quadruples"],
            "removed_only_from_flagged_hours": connection.execute(
                "SELECT COUNT(*) FROM removed_anomalies r JOIN assessed_hours h USING(user_id,timestamp) "
                + "WHERE NOT h.is_burst_anomaly"
            ).fetchone()[0]
            == 0,
            "category_consistency": conflicts == 0,
        }
        if not all(
            value
            for name, value in validations.items()
            if name != "category_consistency"
        ):
            raise ValueError(f"输出核验失败: {validations}")
        time_range = connection.execute(
            "SELECT MIN(timestamp),MAX(timestamp) FROM retained_events"
        ).fetchone()
        report = {
            "schema_version": 2,
            "status": "success",
            "created_at": datetime.now().astimezone().isoformat(),
            "sources": sources,
            "configuration": {
                "duplicate_policy": "preserve_regular_repetitions",
                "chunk_size": config.chunk_size,
                "workers": config.workers,
                "encoding": config.encoding,
                "start_inclusive": config.start,
                "end_inclusive": config.end,
                "timestamp_semantics": "source_local_hour_no_timezone_conversion",
                "calibration_scope": "full_observed_dataset_for_descriptive_cleaning_not_predictive_training",
            },
            "rows": {
                "input": input_rows,
                "invalid": invalid_rows,
                "clean": clean,
                "removed_anomaly_redundancies": removed,
                "retained_ratio": clean / input_rows,
                "observed_duplicate_excess": profile["repetitions"]["excess_records"],
                "remaining_duplicate_excess": clean - retained_keys,
            },
            "missing_by_field_before_cleaning": dict(missing),
            "rejected_by_reason_nonexclusive": dict(reasons),
            "time_min": time_range[0].isoformat(),
            "time_max": time_range[1].isoformat(),
            "behavior_counts": behavior_counts,
            "removed_behavior_counts": removed_behaviors,
            "category_conflicting_items": conflicts,
            "profile": profile,
            "burst_rules": limits,
            "hour_decisions": {
                "high_volume_candidates": hours[0],
                "flagged_hours": hours[1],
                "flagged_users": hours[2],
            },
            "threshold_sensitivity": sensitivity,
            "daily_diagnostics": aggregates["daily_flags"],
            "validation": validations,
            "warnings": [
                "正常重复保留，四元组不再要求唯一；异常小时内每个四元组至少保留一条。",
                "小时桶不是滚动60分钟窗口；同小时实际时间差无法从现有数据推断。",
                "统计异常不等于已证实的机器行为；高频但无二次证据的操作只标记。",
                "个人基线与阈值使用全观察期数据，预测建模时必须仅在训练历史期拟合。",
                "旧版SQLite唯一四元组约束不适用于本版本；旧数据库和旧报告仍属于旧口径。",
            ],
        }
        for name, value in [
            ("quality_report.json", report),
            ("repetition_analysis.json", profile),
        ]:
            (output / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        connection.execute("CHECKPOINT")
        (output / "SUCCESS").write_text("新版异常重复规则及全部输出核验通过\n", encoding="utf-8")
        LOGGER.info("新版清洗完成：%s", output)
    except (OSError, ValueError, duckdb.Error):
        LOGGER.exception("新版清洗未完成，当前目录不可作为成功结果")
        raise
    finally:
        connection.close()
        LOGGER.removeHandler(handler)
        handler.close()
    return output
