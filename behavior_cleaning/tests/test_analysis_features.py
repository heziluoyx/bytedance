"""验证时间泄漏、小时序列及目标编码边界。"""

import sqlite3

import pandas as pd
import pytest

from behavior_cleaning.analysis_runner import summarize_sequence
from behavior_cleaning.feature_snapshots import labels, scoped_connection
from behavior_cleaning.feature_validation import encode_target


def test_historical_view_excludes_cutoff_and_future(tmp_path):
    """临界时刻归入标签，修改未来记录不能改变历史聚合。"""
    database = tmp_path / "history.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE user_behavior(user_id TEXT,event_time TEXT)")
        connection.execute("CREATE TABLE dim_time(event_time TEXT)")
        connection.executemany(
            "INSERT INTO user_behavior VALUES (?,?)",
            [
                ("001", "2025-12-04 23:00:00"),
                ("001", "2025-12-05 00:00:00"),
                ("002", "2025-12-06 00:00:00"),
            ],
        )
    with scoped_connection(database, "2025-12-05 00:00:00") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM user_behavior").fetchone()[0] == 1
        )
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM user_behavior").fetchone()[0] == 3
        )


def test_labels_reject_incomplete_seven_days(tmp_path):
    """不把尚未观察完的未来购买标签当负样本。"""
    database = tmp_path / "labels.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE dim_time(event_time TEXT)")
        connection.execute("INSERT INTO dim_time VALUES ('2025-12-18 23:00:00')")
    with pytest.raises(ValueError, match="不足7天"):
        labels(database, "2025-12-13 00:00:00")


def test_multi_behavior_hours_do_not_invent_order():
    """混合小时桶相邻跳转不能被当作可识别的单行为跳转。"""
    frame = pd.DataFrame(
        {
            "user_id": ["001"] * 4,
            "mask": [1, 3, 4, 8],
            "previous_mask": [None, 1, 3, 4],
            "gap_hours": [None, 1, 1, 1],
        }
    )
    transitions, users = summarize_sequence(frame)
    assert transitions[["previous_mask", "mask"]].values.tolist() == [[4.0, 8.0]]
    assert users.iloc[0]["identifiable_pairs"] == 1


def test_target_encoding_unique_categories_use_fold_prior():
    """唯一类目不能直接编码出本用户标签，未知类目使用训练先验。"""
    train = pd.DataFrame(
        {"dominant_category": ["a", "b", "c", "d"], "target": [0, 1, 0, 1]}
    )
    held = pd.DataFrame({"dominant_category": ["unknown"]})
    encoded, unknown, artifacts = encode_target(train, held)
    assert encoded.tolist() == [0.5] * 4
    assert unknown.iloc[0] == 0.5
    assert artifacts["prior"] == 0.5


def test_end_to_end_sql_and_temporal_features(tmp_path, monkeypatch):
    """通过真实 SQLite 验证全链路，并核验两段标签的已知购买人数。"""
    from pathlib import Path

    from behavior_cleaning.analysis_runner import analyze
    from behavior_cleaning.feature_validation import run_validation

    monkeypatch.syspath_prepend(str(Path(__file__).absolute().parents[1] / ".runtime"))
    database = tmp_path / "complete.sqlite"
    records = []
    for day in range(31):
        date = (pd.Timestamp("2025-11-18") + pd.Timedelta(days=day)).strftime(
            "%Y-%m-%d"
        )
        for user in range(16):
            item = str((user + day) % 12).zfill(3)
            for hour, behavior in [(0, 1), (1, 3), (2, 2)]:
                records.append(
                    (
                        str(user).zfill(3),
                        item,
                        behavior,
                        date + " " + str(hour).zfill(2) + ":00:00",
                    )
                )
            divisor = 4 if day < 17 else 2 if day < 24 else 3
            if user % divisor == 0:
                records.append((str(user).zfill(3), item, 4, date + " 03:00:00"))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE user_behavior(user_id TEXT,item_id TEXT,"
            + "behavior_type INTEGER,event_time TEXT)"
        )
        connection.executemany("INSERT INTO user_behavior VALUES (?,?,?,?)", records)
        connection.execute(
            "CREATE INDEX idx_behavior_quadruple ON user_behavior"
            + "(user_id,item_id,behavior_type,event_time)"
        )
        connection.execute(
            "CREATE TABLE dim_time AS SELECT DISTINCT event_time "
            + "FROM user_behavior"
        )
        connection.execute("CREATE TABLE dim_item(item_id TEXT,category_id TEXT)")
        connection.executemany(
            "INSERT INTO dim_item VALUES (?,?)",
            [(str(i).zfill(3), str(i % 3)) for i in range(12)],
        )
    output = tmp_path / "reports"
    summary = analyze(database, output)
    assert summary["bounds"]["rows"] == len(records)
    result = run_validation(database, output, max_items=10)
    metrics = result["preassessment"]
    assert metrics["train_positive_rate"] == 8 / 16
    assert metrics["validation_positive_rate"] == 6 / 16
    assert metrics["selected_features"]
    categories = pd.read_csv(output / "category_analysis.csv")
    # 同一用户商品可能跨多日才首次购买，不能假设所有转化都发生在当日。
    oracle = []
    events = pd.DataFrame(records, columns=["user", "item", "behavior", "time"])
    for (_, item), group in events.groupby(["user", "item"]):
        first_view = group.loc[group["behavior"] == 1, "time"].min()
        purchases = group.loc[
            (group["behavior"] == 4) & (group["time"] > first_view), "time"
        ]
        if len(purchases):
            hours = (
                pd.Timestamp(purchases.min()) - pd.Timestamp(first_view)
            ).total_seconds() / 3600
            oracle.append((int(item) % 3, hours))
    expected = (
        pd.DataFrame(oracle, columns=["category", "hours"])
        .groupby("category")["hours"]
        .mean()
        .to_dict()
    )
    actual = categories.set_index("category_id")[
        "first_pv_to_first_later_buy_hours"
    ].to_dict()
    assert actual == pytest.approx(expected)
    assert (output / "validation_standardized.parquet").is_file()
