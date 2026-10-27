"""正常重复保留及异常小时有限压缩的业务测试。"""

import json
from pathlib import Path

import pandas as pd
import pytest

from behavior_cleaning.burst_pipeline import run_burst_pipeline
from behavior_cleaning.burst_rules import BurstRules
from behavior_cleaning.cleaning import Config


def _raw(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": times,
            "user_id": "001",
            "item_id": "002",
            "item_category": "003",
            "behavior_type": "1",
        },
        dtype="string",
    )


def test_normal_same_hour_repetitions_preserved(tmp_path: Path) -> None:
    """同小时同一操作重复两次不再自动去重。"""
    source = tmp_path / "input.csv"
    _raw(["2025-12-01 02"] * 2).to_csv(source, index=False)

    output = run_burst_pipeline(Config(source, tmp_path / "out", chunk_size=1))
    report = json.loads((output / "quality_report.json").read_text("utf-8"))

    assert report["rows"]["clean"] == 2
    assert report["rows"]["remaining_duplicate_excess"] == 1
    assert report["rows"]["removed_anomaly_redundancies"] == 0


def test_hundred_is_not_fixed_cutoff(tmp_path: Path) -> None:
    """两个小时各有150次操作，不能仅凭超过100次就删除。"""
    source = tmp_path / "input.csv"
    _raw(["2025-12-01 02"] * 150 + ["2025-12-01 03"] * 150).to_csv(source, index=False)

    output = run_burst_pipeline(Config(source, tmp_path / "out"))
    report = json.loads((output / "quality_report.json").read_text("utf-8"))

    assert report["rows"]["clean"] == 300
    assert report["hour_decisions"]["flagged_hours"] == 0


@pytest.mark.parametrize("chunk_size", [7, 1000])
def test_extreme_hour_only_redundancies_removed(
    tmp_path: Path, chunk_size: int
) -> None:
    """全局统计不受块边界影响，异常小时仍保留一条独立操作。"""
    source = tmp_path / "input.csv"
    normal = (
        pd.date_range("2025-12-01", periods=30, freq="h")
        .strftime("%Y-%m-%d %H")
        .tolist()
    )
    _raw(normal + ["2025-12-03 00"] * 100).to_csv(source, index=False)

    output = run_burst_pipeline(Config(source, tmp_path / "out", chunk_size=chunk_size))
    report = json.loads((output / "quality_report.json").read_text("utf-8"))
    removed = pd.read_parquet(output / "removed_anomalies.parquet")

    assert report["rows"]["clean"] == 31
    assert report["rows"]["removed_anomaly_redundancies"] == 99
    assert removed["timestamp"].nunique() == 1
    assert all(report["validation"].values())


def test_different_items_in_burst_hour_preserved(tmp_path: Path) -> None:
    """异常小时不同商品的独立行为不整段抹去。"""
    source = tmp_path / "input.csv"
    normal = (
        pd.date_range("2025-12-01", periods=30, freq="h")
        .strftime("%Y-%m-%d %H")
        .tolist()
    )
    raw = _raw(normal + ["2025-12-03 00"] * 100)
    raw.loc[len(raw) - 1, "item_id"] = "999"
    raw.to_csv(source, index=False)

    output = run_burst_pipeline(Config(source, tmp_path / "out"))
    cleaned = pd.read_parquet(output / "cleaned_events.parquet")

    assert (cleaned["item_id"] == "999").sum() == 1
    assert len(cleaned) == 32


def test_rule_parameters_checked() -> None:
    """禁止使用无效分位数。"""
    with pytest.raises(ValueError):
        BurstRules(hour_quantile=1.0)
