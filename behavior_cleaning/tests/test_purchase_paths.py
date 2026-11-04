"""验证购买路径的小时顺序、七天边界、重复及不完整观察期。"""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from behavior_cleaning.purchase_paths import run_purchase_path_analysis


def make_database(path: Path, records: list[tuple]) -> None:
    """建立带已知观察截止时间的最小SQLite样本。"""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE user_behavior "
            + "(user_id TEXT,item_id TEXT,behavior_type INTEGER,event_time TEXT)"
        )
        connection.execute(
            "CREATE INDEX idx_behavior_quadruple ON user_behavior "
            + "(user_id,item_id,behavior_type,event_time)"
        )
        connection.execute("CREATE TABLE dim_time(event_time TEXT)")
        connection.executemany(
            "INSERT INTO dim_time VALUES (?)",
            [("2025-12-01 00:00:00",), ("2025-12-15 00:00:00",)],
        )
        connection.executemany("INSERT INTO user_behavior VALUES (?,?,?,?)", records)


SQL = Path(__file__).resolve().parents[1] / "sql/04_purchase_paths.sql"


def test_seven_day_boundary_ties_and_censoring(tmp_path):
    """包含168小时边界，排除同小时、超窗和观察不足的起点。"""
    records = [
        ("boundary", "i", 3, "2025-12-01 00:00:00"),
        ("boundary", "i", 3, "2025-12-01 00:00:00"),
        ("boundary", "i", 4, "2025-12-08 00:00:00"),
        ("outside", "i", 3, "2025-12-01 00:00:00"),
        ("outside", "i", 4, "2025-12-08 01:00:00"),
        ("tie", "i", 3, "2025-12-01 00:00:00"),
        ("tie", "i", 4, "2025-12-01 00:00:00"),
        ("late", "i", 3, "2025-12-09 00:00:00"),
        ("late", "i", 4, "2025-12-10 00:00:00"),
        ("repeat", "i", 4, "2025-12-01 00:00:00"),
        ("repeat", "i", 3, "2025-12-02 00:00:00"),
        ("repeat", "i", 4, "2025-12-03 00:00:00"),
    ]
    database = tmp_path / "input.sqlite"
    make_database(database, records)
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    result = run_purchase_path_analysis(database, tmp_path / "out", SQL)
    cart = next(x for x in result["paths"] if x["start_action"] == "cart")
    assert cart["pairs"] == 5
    assert cart["later_buy_pairs"] == 4
    assert cart["complete_7d_pairs"] == 4
    assert cart["complete_7d_buy_pairs"] == 2
    assert cart["complete_7d_rate"] == 0.5
    assert hashlib.sha256(database.read_bytes()).hexdigest() == digest
    assert (tmp_path / "out/SUCCESS").exists()
    assert (
        next(x for x in result["paths"] if x["start_action"] == "fav")[
            "complete_7d_rate"
        ]
        is None
    )


def test_observed_direct_path_excludes_same_hour_intent(tmp_path):
    """无先前或同小时意向与顺序未知分开，避免把同小时误作直购。"""
    records = [
        ("browse", "i", 1, "2025-12-01 00:00:00"),
        ("browse", "i", 4, "2025-12-01 01:00:00"),
        ("unknown", "i", 1, "2025-12-01 00:00:00"),
        ("unknown", "i", 3, "2025-12-01 01:00:00"),
        ("unknown", "i", 4, "2025-12-01 01:00:00"),
        ("intent", "i", 3, "2025-12-01 00:00:00"),
        ("intent", "i", 2, "2025-12-01 00:00:00"),
        ("intent", "i", 4, "2025-12-01 01:00:00"),
    ]
    database = tmp_path / "input.sqlite"
    make_database(database, records)
    result = run_purchase_path_analysis(database, tmp_path / "out", SQL)
    counts = result["all_period_and_first_purchase"]
    assert counts["purchased_pairs"] == 3
    assert counts["observed_browse_without_cart_fav_before_or_same_hour"] == 1
    assert counts["no_earlier_intent_but_same_hour_intent"] == 1
    assert counts["both_before_first_buy"] == 1
    assert result["paths_overlap"] is True


def test_existing_output_is_preserved(tmp_path):
    """重复执行不能覆盖已有分析结果。"""
    database = tmp_path / "input.sqlite"
    make_database(database, [("u", "i", 1, "2025-12-01 00:00:00")])
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_purchase_path_analysis(database, output, SQL)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
