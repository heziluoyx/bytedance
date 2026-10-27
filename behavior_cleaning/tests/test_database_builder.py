"""数据库建库、批量导入、约束和核验的独立测试。"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from behavior_cleaning.database_builder import BuildConfig, build_database
from behavior_cleaning.database_recovery import finish_database


def _source(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "user_id": ["001", "001", "002"],
            "item_id": ["010", "010", "011"],
            "category_id": ["020", "020", "021"],
            "behavior_type": [1, 4, 2],
            "timestamp": pd.to_datetime(
                ["2025-12-01 02:00:00", "2025-12-01 03:00:00", "2025-12-01 02:00:00"]
            ),
        }
    )
    source = tmp_path / "cleaned.parquet"
    frame.to_parquet(source, index=False)
    return source


def test_build_counts_and_time_format(tmp_path: Path) -> None:
    """分块导入后维度去重、前导零及 SQLite 时间格式正确。"""
    source = _source(tmp_path)

    target = build_database(
        BuildConfig(source, tmp_path / "result.sqlite", chunk_size=1)
    )
    report = json.loads(target.with_suffix(".validation.json").read_text("utf-8"))
    connection = sqlite3.connect(target)
    try:
        first = connection.execute(
            "SELECT user_id, event_time FROM user_behavior ORDER BY event_id LIMIT 1"
        ).fetchone()
        weekday = connection.execute(
            "SELECT weekday FROM v_time_attributes LIMIT 1"
        ).fetchone()[0]
    finally:
        connection.close()

    assert report["table_rows"] == {
        "user_behavior": 3,
        "dim_user": 2,
        "dim_item": 2,
        "dim_time": 2,
    }
    assert first == ("001", "2025-12-01 02:00:00")
    assert weekday == 1
    assert report["integrity_check"] == "ok"


def test_existing_database_never_overwritten(tmp_path: Path) -> None:
    """已有数据库不能因重新运行被覆盖。"""
    source = _source(tmp_path)
    target = tmp_path / "existing.sqlite"
    target.write_bytes(b"preserve me")

    with pytest.raises(FileExistsError):
        build_database(BuildConfig(source, target))

    assert target.read_bytes() == b"preserve me"


def test_cross_batch_category_conflict_aborts(tmp_path: Path) -> None:
    """跨块商品类目冲突使构建失败，不发布半成品数据库。"""
    source = _source(tmp_path)
    frame = pd.read_parquet(source)
    frame.loc[1, "category_id"] = "999"
    frame.to_parquet(source, index=False)
    target = tmp_path / "result.sqlite"

    with pytest.raises(ValueError, match="跨批次商品类目冲突"):
        build_database(BuildConfig(source, target, chunk_size=1))

    assert not target.exists()
    assert not target.with_suffix(".build.lock").exists()


def test_duplicate_quadruple_aborts(tmp_path: Path) -> None:
    """跨块四元组重复由唯一索引发现，不能发布为成功结果。"""
    source = _source(tmp_path)
    frame = pd.read_parquet(source)
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(source, index=False)
    target = tmp_path / "result.sqlite"

    with pytest.raises(sqlite3.IntegrityError):
        build_database(BuildConfig(source, target, chunk_size=1))

    assert not target.exists()


def test_foreign_key_enforced_on_future_writes(tmp_path: Path) -> None:
    """连接启用外键后拒绝不存在的用户引用。"""
    target = build_database(BuildConfig(_source(tmp_path), tmp_path / "result.sqlite"))
    connection = sqlite3.connect(target)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO user_behavior(user_id,item_id,behavior_type,event_time) "
                + "VALUES(?,?,?,?)",
                ("missing", "010", 1, "2025-12-01 02:00:00"),
            )
    finally:
        connection.rollback()
        connection.close()


def test_cross_check_wrong_cleaning_report_aborts(tmp_path: Path) -> None:
    """输入与清洗报告不一致时拒绝发布。"""
    source = _source(tmp_path)
    (tmp_path / "quality_report.json").write_text(
        json.dumps({"rows": {"clean": 999}, "behavior_counts": {"1": 3}}),
        encoding="utf-8",
    )
    target = tmp_path / "result.sqlite"

    with pytest.raises(ValueError, match="交叉核对失败"):
        build_database(BuildConfig(source, target))

    assert not target.exists()


def test_subhour_timestamp_rejected(tmp_path: Path) -> None:
    """输入的分钟信息不能被自动截断为整点。"""
    source = _source(tmp_path)
    frame = pd.read_parquet(source)
    frame.loc[0, "timestamp"] = pd.Timestamp("2025-12-01 02:30:00")
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="小时精度"):
        build_database(BuildConfig(source, tmp_path / "result.sqlite"))


def test_invalid_chunk_size(tmp_path: Path) -> None:
    """块大小必须为正整数。"""
    with pytest.raises(ValueError, match="正整数"):
        BuildConfig(tmp_path / "source", tmp_path / "target", chunk_size=0)


def test_new_policy_preserves_repetitions_in_database(tmp_path: Path) -> None:
    """新版报告允许正常重复导入，四元组普通索引不再拒绝这些事件。"""
    source = _source(tmp_path)
    frame = pd.read_parquet(source)
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(source, index=False)
    (tmp_path / "quality_report.json").write_text(
        json.dumps(
            {
                "rows": {"clean": 4},
                "behavior_counts": {"1": 2, "2": 1, "4": 1},
                "configuration": {"duplicate_policy": "preserve_regular_repetitions"},
            }
        ),
        encoding="utf-8",
    )

    target = build_database(BuildConfig(source, tmp_path / "new.sqlite"))
    report = json.loads(target.with_suffix(".validation.json").read_text("utf-8"))

    assert report["table_rows"]["user_behavior"] == 4
    assert "idx_behavior_quadruple" in report["indexes"]
    assert "uq_behavior_quadruple" not in report["indexes"]


@pytest.mark.parametrize("incomplete", [False, True])
def test_resume_only_complete_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incomplete: bool
) -> None:
    """索引中断可续建，但未完整导入的临时库必须拒绝发布。"""
    import behavior_cleaning.database_builder as builder

    source = _source(tmp_path)
    target = tmp_path / "result.sqlite"
    original = builder._create_indexes

    def _fail_index(connection: sqlite3.Connection, sql_text: str) -> None:
        connection.execute(
            "CREATE INDEX idx_behavior_user_time ON user_behavior(user_id,event_time)"
        )
        connection.commit()
        raise sqlite3.OperationalError("simulated full temporary disk")

    monkeypatch.setattr(builder, "_create_indexes", _fail_index)
    with pytest.raises(sqlite3.OperationalError):
        build_database(BuildConfig(source, target))
    monkeypatch.setattr(builder, "_create_indexes", original)
    temporary = next(tmp_path.glob("*.building_*.sqlite"))
    if incomplete:
        with sqlite3.connect(temporary) as connection:
            connection.execute("DELETE FROM user_behavior WHERE event_id=1")
        with pytest.raises(ValueError, match="未完整导入"):
            finish_database(BuildConfig(source, target), temporary)
        assert not target.exists()
    else:
        result = finish_database(BuildConfig(source, target), temporary)
        report = json.loads(result.with_suffix(".validation.json").read_text("utf-8"))
        assert report["table_rows"]["user_behavior"] == 3
        assert not temporary.exists()
