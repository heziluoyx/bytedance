"""官方日历边界和无节假日样本时的统计语义。"""

import sqlite3

import pandas as pd

from behavior_cleaning.china_calendar import calendar_2025, run_calendar_analysis


def test_official_holiday_and_makeup_dates():
    """校验周日补班、周内假期和节后周六补班，不能只看星期。"""
    calendar = calendar_2025().set_index("date")
    assert len(calendar) == 365
    assert calendar.loc["2025-01-26", "is_workday"] == 1
    assert calendar.loc["2025-01-26", "is_weekend"] == 1
    assert calendar.loc["2025-01-28", "is_holiday"] == 1
    assert calendar.loc["2025-02-04", "holiday_name"] == "春节"
    assert calendar.loc["2025-10-08", "is_holiday"] == 1
    assert calendar.loc["2025-10-11", "is_makeup_workday"] == 1
    assert calendar.loc["2025-12-12", "is_holiday"] == 0


def test_missing_holiday_group_is_unknown_not_zero(tmp_path):
    """真实 SQLite 导入全年日历，缺假期组必须输出空指标与明确状态。"""
    database = tmp_path / "source.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE user_behavior(user_id TEXT,"
            + "behavior_type INTEGER,event_time TEXT)"
        )
        connection.executemany(
            "INSERT INTO user_behavior VALUES (?,?,?)",
            [
                ("001", 1, "2025-12-12 09:00:00"),
                ("001", 4, "2025-12-12 10:00:00"),
                ("002", 1, "2025-12-13 09:00:00"),
            ],
        )
    output = tmp_path / "results"
    summary = run_calendar_analysis(database, output)
    assert summary["holiday_comparison_estimable"] is False
    assert summary["holiday_effect"] is None
    groups = pd.read_csv(output / "holiday_comparison.csv").set_index("is_holiday")
    assert groups.loc[1, "observed_days"] == 0
    assert pd.isna(groups.loc[1, "events_per_day"])
    assert groups.loc[0, "events_per_day"] == 1.5
    with sqlite3.connect(output / "holiday_analysis.sqlite") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM dim_calendar").fetchone()[0] == 365
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM daily_behavior").fetchone()[0] == 2
        )
