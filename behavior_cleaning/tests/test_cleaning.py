"""标准化规则和全流程关键边界的独立测试。"""

import json
from pathlib import Path

import pandas as pd
import pytest

from behavior_cleaning.cleaning import Config, clean_chunk
from behavior_cleaning.pipeline import run_pipeline


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """为每个案例构造独立配置。"""
    return Config(tmp_path / "input.csv", tmp_path / "outputs", chunk_size=1)


def _frame(timestamp: str = "2025-12-01 02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [timestamp],
            "user_id": [" 001 "],
            "item_id": ["002"],
            "item_category": ["003"],
            "behavior_type": ["1"],
        },
        dtype="string",
    )


def test_aliases_and_leading_zero(config: Config) -> None:
    """字段别名规范化时保留 ID 的前导零。"""
    result = clean_chunk(_frame(), "source.csv", 0, config)
    assert result.accepted.iloc[0]["user_id"] == "001"
    assert result.accepted.iloc[0]["category_id"] == "003"


@pytest.mark.parametrize(
    "timestamp", ["2025-02-30 02", "2025-12-01 24", "2025-12-01 02:00:00"]
)
def test_invalid_timestamp(config: Config, timestamp: str) -> None:
    """无效日期及超出原始精度的格式必须被拒绝。"""
    result = clean_chunk(_frame(timestamp), "source.csv", 7, config)
    assert result.accepted.empty
    assert result.rejected.iloc[0]["source_row"] == 8


def test_missing_id_is_not_filled(config: Config) -> None:
    """缺失主键不能通过任意填充制造用户。"""
    raw = _frame()
    raw.loc[0, "user_id"] = "NULL"
    result = clean_chunk(raw, "source.csv", 0, config)
    assert result.rejected.iloc[0]["reject_reason"] == "missing_user_id"


def test_invalid_behavior(config: Config) -> None:
    """拒绝未定义的行为编码。"""
    raw = _frame()
    raw.loc[0, "behavior_type"] = "5"
    result = clean_chunk(raw, "source.csv", 0, config)
    assert result.accepted.empty


def test_duplicate_alias_rejected(config: Config) -> None:
    """同时出现标准时间列和别名时不静默选列。"""
    raw = _frame()
    raw["timestamp"] = raw["time"]
    with pytest.raises(ValueError, match="重名"):
        clean_chunk(raw, "source.csv", 0, config)


def test_time_bounds_are_inclusive(tmp_path: Path) -> None:
    """采样起止端点属于有效范围。"""
    config = Config(
        tmp_path / "input.csv",
        tmp_path / "out",
        start="2025-12-01 02",
        end="2025-12-01 02",
    )
    result = clean_chunk(_frame(), "source.csv", 0, config)
    assert len(result.accepted) == 1


def test_global_dedup_and_category_audit(config: Config) -> None:
    """跨块重复按四元组消除，但类目分歧保留审计。"""
    raw = pd.concat([_frame(), _frame(), _frame("2025-12-01 03")])
    raw.iloc[1, raw.columns.get_loc("item_category")] = "004"
    raw.to_csv(config.input_path, index=False)

    output = run_pipeline(config)
    report = json.loads((output / "quality_report.json").read_text("utf-8"))
    cleaned = pd.read_parquet(output / "cleaned_events.parquet")
    duplicates = pd.read_parquet(output / "duplicates.parquet")

    assert report["rows"] == {
        "input": 3,
        "invalid": 0,
        "duplicates": 1,
        "clean": 2,
        "retained_ratio": 2 / 3,
    }
    assert len(cleaned) == 2
    assert duplicates.iloc[0]["kept_category_id"] == "003"
    assert report["category_conflicting_items"] == 1
    assert (output / "SUCCESS").exists()


def test_cross_file_duplicate(tmp_path: Path) -> None:
    """全局去重覆盖多个文件。"""
    source = tmp_path / "input"
    source.mkdir()
    _frame().to_csv(source / "a.csv", index=False)
    _frame().to_csv(source / "b.csv", index=False)

    output = run_pipeline(Config(source, tmp_path / "out", chunk_size=1))
    report = json.loads((output / "quality_report.json").read_text("utf-8"))

    assert report["rows"]["clean"] == 1
    assert report["rows"]["duplicates"] == 1


def test_reject_and_row_balance(config: Config) -> None:
    """无效记录单独保存且行数核对包含拒绝行。"""
    raw = pd.concat([_frame(), _frame("2025-02-30 00")])
    raw.to_csv(config.input_path, index=False)

    output = run_pipeline(config)
    report = json.loads((output / "quality_report.json").read_text("utf-8"))
    rejected = pd.read_json(output / "rejected.jsonl", lines=True)

    assert report["rows"]["invalid"] == 1
    assert report["validation"]["row_balance"]
    assert rejected.iloc[0]["source_row"] == 2


def test_outliers_flagged_without_deletion(config: Config) -> None:
    """高活跃尾部仅标记，清洗明细仍完整保留。"""
    raw = pd.concat([_frame()] * 29, ignore_index=True)
    raw["item_id"] = pd.Series(range(1, 30), dtype="string")
    raw.loc[:8, "user_id"] = pd.Series(range(10, 19), dtype="string")
    raw.to_csv(config.input_path, index=False)

    output = run_pipeline(config)
    flags = pd.read_parquet(output / "user_daily_flags.parquet")
    cleaned = pd.read_parquet(output / "cleaned_events.parquet")

    assert flags["is_iqr_outlier"].sum() == 1
    assert len(cleaned) == 29


def test_extra_csv_column_is_not_silently_dropped(config: Config) -> None:
    """首条数据的多余列不能被 Pandas 当作隐式索引丢弃。"""
    config.input_path.write_text(
        "time,user_id,item_id,item_category,behavior_type\n"
        + "2025-12-01 02,1,2,3,1,extra\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="隐式索引"):
        run_pipeline(config)
