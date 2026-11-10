"""核验分层、未来信息隔离及空序列和填充位置的模型行为。"""

import sqlite3

import numpy as np
import pytest
import torch

from behavior_cleaning.pair_dataset import build_dataset, encode_category, split_indices
from behavior_cleaning.sequence_models import PurchaseSequenceModel


def test_pair_split_is_disjoint_and_stratified():
    """不重采样验证测试集，三个集合无交叉且保留正例比例。"""
    y = np.tile([0] * 19 + [1], 100)
    train, valid, test = split_indices(y)
    assert [len(train), len(valid), len(test)] == [1400, 400, 200]
    assert not set(train) & set(valid)
    assert not set(train) & set(test)
    assert not set(valid) & set(test)
    assert [y[i].mean() for i in [train, valid, test]] == [0.05] * 3


def test_target_encoder_does_not_read_held_labels():
    """改变验证和测试标签不能影响训练编码或已拟合映射。"""
    y = np.tile([0, 1], 30)
    train = np.arange(40)
    category = np.array(["a", "b", "c"] * 20)
    before, mapping = encode_category(category, y, train)
    changed = y.copy()
    changed[40:] = 1 - changed[40:]
    after, changed_mapping = encode_category(category, changed, train)
    np.testing.assert_array_equal(before, after)
    assert mapping == changed_mapping


def test_future_events_cannot_change_historical_features(tmp_path):
    """未来浏览仅是未来记录，不能回流进历史特征或序列。"""
    database = tmp_path / "input.sqlite"
    with sqlite3.connect(database) as con:
        con.execute(
            "CREATE TABLE user_behavior(user_id TEXT,item_id TEXT,behavior_type INTEGER,event_time TEXT)"
        )
        con.execute(
            "CREATE INDEX idx_behavior_quadruple ON user_behavior(user_id,item_id,behavior_type,event_time)"
        )
        con.execute("CREATE TABLE dim_item(item_id TEXT PRIMARY KEY,category_id TEXT)")
        con.execute("CREATE TABLE dim_time(event_time TEXT)")
        con.executemany(
            "INSERT INTO dim_time VALUES(?)",
            [("2025-11-18 00:00:00",), ("2025-12-18 23:00:00",)],
        )
        for i in range(120):
            con.execute("INSERT INTO dim_item VALUES(?,?)", (str(i), str(i % 3)))
            con.execute(
                "INSERT INTO user_behavior VALUES(?,?,1,'2025-12-10 12:00:00')",
                (str(i), str(i)),
            )
            if i % 4 == 0:
                con.execute(
                    "INSERT INTO user_behavior VALUES(?,?,4,'2025-12-13 12:00:00')",
                    (str(i), str(i)),
                )
    build_dataset(database, tmp_path / "before", max_pairs=120)
    with sqlite3.connect(database) as con:
        con.execute("INSERT INTO user_behavior VALUES('0','0',1,'2025-12-14 12:00:00')")
    build_dataset(database, tmp_path / "after", max_pairs=120)
    with np.load(tmp_path / "before/dataset.npz") as before, np.load(
        tmp_path / "after/dataset.npz"
    ) as after:
        for name in before.files:
            np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize("kind", ["LSTM", "GRU", "DIN", "DIN_MHA"])
def test_sequence_padding_is_masked_and_empty_rows_are_finite(kind):
    """填充位置变化不改变输出，完全无历史时也不产生NaN。"""
    torch.manual_seed(42)
    model = PurchaseSequenceModel(kind, 20, 10, 3, embedding=8, hidden=8, dropout=0)
    model.eval()
    numeric = torch.zeros(2, 3)
    items = torch.zeros(2, 4, 2, dtype=torch.long)
    categories = torch.zeros_like(items)
    items[0, 0, 0] = 2
    categories[0, 0, 0] = 2
    mask = torch.tensor([[True, False, False, False], [False] * 4])
    auxiliary = torch.zeros(2, 4, 6)
    with torch.no_grad():
        first = model(
            numeric,
            torch.tensor([2, 2]),
            torch.tensor([2, 2]),
            items,
            categories,
            auxiliary,
            mask,
        )
        items[:, 1:] = 3
        categories[:, 1:] = 3
        second = model(
            numeric,
            torch.tensor([2, 2]),
            torch.tensor([2, 2]),
            items,
            categories,
            auxiliary,
            mask,
        )
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)
