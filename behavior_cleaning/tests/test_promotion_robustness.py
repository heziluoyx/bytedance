"""检验固定历史窗口、候选资格和跨期预处理标签隔离。"""
import sqlite3

import numpy as np
import pandas as pd

from behavior_cleaning.promotion_report import population_stability
from behavior_cleaning.promotion_robustness import (
    EARLY,
    NUMERIC,
    build_snapshot,
    history_bounds,
    prepare_matrices,
)


def create_source(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """CREATE TABLE user_behavior(user_id TEXT,item_id TEXT,behavior_type INTEGER,event_time TEXT);
          CREATE INDEX idx_behavior_quadruple ON user_behavior(user_id,item_id,behavior_type,event_time);
          CREATE TABLE dim_item(item_id TEXT PRIMARY KEY,category_id TEXT);
          CREATE TABLE dim_time(event_time TEXT PRIMARY KEY);"""
        )
        connection.executemany(
            "INSERT INTO dim_time VALUES (?)",
            [("2025-11-17 00:00:00",), ("2025-12-01 23:00:00",)],
        )
        for i in range(150):
            connection.execute(
                "INSERT INTO dim_item VALUES (?,?)", (str(i), str(i % 7))
            )
            connection.execute(
                "INSERT INTO user_behavior VALUES (?,?,1,?)",
                (str(i), str(i), "2025-11-20 00:00:00"),
            )
            if i % 3 == 0:
                connection.execute(
                    "INSERT INTO user_behavior VALUES (?,?,4,?)",
                    (str(i), str(i), "2025-11-25 00:00:00"),
                )
        connection.execute("INSERT INTO dim_item VALUES ('only_future','x')")
        connection.execute(
            "INSERT INTO user_behavior VALUES ('new','only_future',4,'2025-11-26 00:00:00')"
        )
        connection.execute(
            "INSERT INTO user_behavior VALUES ('old','only_future',1,'2025-11-17 00:00:00')"
        )


def test_fixed_window_bounds():
    assert history_bounds(EARLY) == (
        "2025-11-18 00:00:00",
        "2025-11-25 00:00:00",
        "2025-12-02 00:00:00",
    )


def test_candidates_do_not_use_future_or_pre_history_rows(tmp_path):
    source = tmp_path / "source.sqlite"
    create_source(source)
    frame, info = build_snapshot(source, EARLY, tmp_path / "snapshot.parquet", 1000)
    assert len(frame) == 150 and info["candidate_pairs"] == 150
    assert set(frame.user_id) == {str(i) for i in range(150)}
    assert frame.target.sum() == 50
    assert (frame.pair_buys == 0).all()
    assert (frame.pair_pv == 1).all()
    assert (
        frame.pair_recent_events
        == frame.pair_pv + frame.pair_cart + frame.pair_fav + frame.pair_buys
    ).all()


def test_future_label_changes_do_not_change_snapshot_features(tmp_path):
    source = tmp_path / "source.sqlite"
    create_source(source)
    expected, _ = build_snapshot(source, EARLY, tmp_path / "first.parquet", 1000)
    with sqlite3.connect(source) as connection:
        connection.execute(
            "UPDATE user_behavior SET behavior_type=1 WHERE event_time>='2025-11-25 00:00:00'"
        )
        connection.execute(
            "INSERT INTO user_behavior VALUES ('1','1',4,'2025-11-28 00:00:00')"
        )
    actual, _ = build_snapshot(source, EARLY, tmp_path / "second.parquet", 1000)
    expected = expected.sort_values("user_id").reset_index(drop=True)
    actual = actual.sort_values("user_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected[NUMERIC], actual[NUMERIC])
    assert not np.array_equal(expected.target, actual.target)


def test_later_targets_cannot_change_encoding_or_scaling(tmp_path):
    source = tmp_path / "source.sqlite"
    create_source(source)
    early, _ = build_snapshot(source, EARLY, tmp_path / "sample.parquet", 1000)
    late = early.copy()
    x, _ = prepare_matrices(early, late, np.arange(100))
    late["target"] = 1 - late.target
    changed, _ = prepare_matrices(early, late, np.arange(100))
    np.testing.assert_array_equal(x, changed)


def test_psi_self_zero_and_constant_shift_detected():
    assert population_stability(np.arange(100), np.arange(100)) == 0
    assert population_stability(np.zeros(100), np.ones(100)) > 0
