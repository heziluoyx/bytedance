"""固定历史截点的用户商品样本、训练内编码及小时桶序列。"""

import gc
import json
import logging
import sqlite3
from collections import Counter, defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def split_indices(
    y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按标签分层划分70%、20%、10%，不做训练前重采样。"""
    train, rest = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    valid, test = train_test_split(
        rest, test_size=1 / 3, stratify=y[rest], random_state=seed
    )
    return train, valid, test


def encode_category(
    categories, y: np.ndarray, train: np.ndarray, seed: int = 42, smoothing: float = 20
) -> tuple[np.ndarray, dict]:
    """训练折外目标编码，其他行只使用训练统计；ID本身不入数值模型。"""
    categories = np.asarray(categories)
    encoded = np.zeros(len(y), dtype=np.float32)
    prior = float(y[train].mean())

    def mapping(index):
        frame = pd.DataFrame({"category": categories[index], "target": y[index]})
        stats = frame.groupby("category").target.agg(["sum", "count"])
        return (
            (stats["sum"] + smoothing * y[index].mean()) / (stats["count"] + smoothing)
        ).to_dict()

    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fit, held in folds.split(train, y[train]):
        fit_index, held_index = train[fit], train[held]
        values = mapping(fit_index)
        encoded[held_index] = [
            values.get(c, float(y[fit_index].mean())) for c in categories[held_index]
        ]
    values = mapping(train)
    remaining = np.setdiff1d(np.arange(len(y)), train)
    encoded[remaining] = [values.get(c, prior) for c in categories[remaining]]
    return encoded, {"mapping": values, "prior": prior, "smoothing": smoothing}


def historical_sequences(
    connection: sqlite3.Connection, cutoff: str, length: int = 16, per_hour: int = 4
) -> dict:
    """最近七天的最后若干小时，同小时按权重选商品并池化，不虚构先后。"""
    start = (pd.Timestamp(cutoff) - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    query = (
        "SELECT user_id,event_time,item_id,behavior_type,COUNT(*) n "
        + "FROM user_behavior INDEXED BY idx_behavior_quadruple "
        + "WHERE event_time>=? AND event_time<? "
        + "GROUP BY user_id,event_time,item_id,behavior_type ORDER BY user_id,event_time"
    )
    histories = defaultdict(lambda: deque(maxlen=length))
    current, items, types = None, Counter(), np.zeros(4)
    weights = {1: 1, 2: 2, 3: 3, 4: 5}

    def finish():
        if current is not None:
            chosen = sorted(items, key=lambda i: (-items[i], i))[:per_hour]
            histories[current[0]].append((current[1], chosen, types.copy()))

    for user, hour, item, behavior, count in connection.execute(query, (start, cutoff)):
        key = (user, hour)
        if key != current:
            finish()
            current, items, types = key, Counter(), np.zeros(4)
        items[item] += count * weights[behavior]
        types[behavior - 1] += count
    finish()
    return {user: list(events) for user, events in histories.items()}


def build_dataset(
    database, output, cutoff="2025-12-12 00:00:00", max_pairs=120000, seed=42
):
    """从历史最近七天交互候选均匀蓄水池抽样，未来仅用于标签。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    con = sqlite3.connect(Path(database).absolute().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-98304")
    last_hour = con.execute("SELECT MAX(event_time) FROM dim_time").fetchone()[0]
    horizon_end = pd.Timestamp(cutoff) + pd.Timedelta(days=7)
    if pd.Timestamp(last_hour) + pd.Timedelta(hours=1) < horizon_end:
        raise ValueError("未来7天标签观察期不完整")
    recent = (pd.Timestamp(cutoff) - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    end = horizon_end.strftime("%Y-%m-%d %H:%M:%S")
    query = """SELECT user_id,item_id,
      SUM(event_time<:cutoff AND behavior_type=1) pv,
      SUM(event_time<:cutoff AND behavior_type=2) fav,
      SUM(event_time<:cutoff AND behavior_type=3) cart,
      SUM(event_time<:cutoff AND behavior_type=4) buys,
      SUM(event_time>=:recent AND event_time<:cutoff) recent_events,
      MIN(CASE WHEN event_time<:cutoff THEN event_time END) first_time,
      MAX(CASE WHEN event_time<:cutoff THEN event_time END) last_time,
      MAX(CASE WHEN event_time<:cutoff AND behavior_type=4 THEN event_time END) last_buy,
      COUNT(DISTINCT CASE WHEN event_time<:cutoff THEN date(event_time) END) days,
      MAX(event_time>=:cutoff AND event_time<:end AND behavior_type=4) target
      FROM user_behavior INDEXED BY idx_behavior_quadruple GROUP BY user_id,item_id"""
    rng = np.random.RandomState(seed)
    records, item_stats = [], {}
    total, positive, all_future = 0, 0, 0
    logging.info("流式聚合候选用户商品对及历史商品统计")
    for row in con.execute(query, {"cutoff": cutoff, "recent": recent, "end": end}):
        (
            user,
            item,
            pv,
            fav,
            cart,
            buys,
            n_recent,
            first,
            last,
            last_buy,
            days,
            target,
        ) = row
        all_future += target
        events = pv + fav + cart + buys
        if events:
            values = item_stats.setdefault(item, [0, 0, 0])
            values[0] += events
            values[1] += buys
            values[2] += 1
        if not n_recent:
            continue
        total += 1
        positive += target
        if len(records) < max_pairs:
            records.append(row)
        else:
            position = rng.randint(total)
            if position < max_pairs:
                records[position] = row
    columns = [
        "user_id",
        "item_id",
        "pair_pv",
        "pair_fav",
        "pair_cart",
        "pair_buys",
        "pair_recent_events",
        "first_time",
        "last_time",
        "last_buy",
        "pair_days",
        "target",
    ]
    frame = pd.DataFrame.from_records(records, columns=columns)
    if len(frame) < 100 or frame.target.nunique() != 2:
        raise ValueError("样本不足或缺少正负标签")
    del records
    for offset, name in enumerate(["item_events", "item_buys", "item_users"]):
        frame[name] = [item_stats[i][offset] for i in frame.item_id]
    del item_stats
    gc.collect()
    logging.info("候选池%d对，正例%d；均匀采样%d对", total, positive, len(frame))
    users = pd.read_sql_query(
        "SELECT user_id,COUNT(*) user_events,COUNT(DISTINCT item_id) user_items,"
        + "COUNT(DISTINCT date(event_time)) user_days,"
        + "SUM(behavior_type=1) user_pv,SUM(behavior_type=2) user_fav,"
        + "SUM(behavior_type=3) user_cart,SUM(behavior_type=4) user_buys,"
        + "MAX(event_time) user_last,MAX(CASE WHEN behavior_type=4 THEN event_time END) user_last_buy "
        + "FROM user_behavior INDEXED BY idx_behavior_quadruple WHERE event_time<? GROUP BY user_id",
        con,
        params=(cutoff,),
    )
    frame = frame.merge(users, on="user_id", validate="many_to_one")
    for field in ["first_time", "last_time", "last_buy", "user_last", "user_last_buy"]:
        frame[field + "_days_ago"] = (
            pd.Timestamp(cutoff) - pd.to_datetime(frame[field])
        ).dt.total_seconds() / 86400
    frame["pair_buy_share"] = frame.pair_buys / (
        frame.pair_pv + frame.pair_fav + frame.pair_cart + frame.pair_buys
    )
    frame["item_buy_share"] = frame.item_buys / frame.item_events
    frame["user_buy_share"] = frame.user_buys / frame.user_events
    frame["pair_user_event_share"] = (
        frame.pair_pv + frame.pair_fav + frame.pair_cart + frame.pair_buys
    ) / frame.user_events
    logging.info("构建小时桶序列")
    histories = historical_sequences(con, cutoff)
    needed = set(frame.item_id)
    for history in histories.values():
        for _, items, _ in history:
            needed.update(items)
    # query_only只约束源库；为避免临时写表，按小批IN参数查商品维度。
    category_map = {}
    needed = list(needed)
    for start in range(0, len(needed), 800):
        batch = needed[start : start + 800]
        category_map.update(
            con.execute(
                "SELECT item_id,category_id FROM dim_item WHERE item_id IN ("
                + ",".join("?" for _ in batch)
                + ")",
                batch,
            )
        )
    con.close()
    frame["category_id"] = frame.item_id.map(category_map).fillna("unknown")
    y = frame.target.to_numpy(dtype=np.int64)
    train, valid, test = split_indices(y, seed)
    numeric = [
        c
        for c in frame
        if c.startswith(("pair_", "item_", "user_"))
        and c not in ["user_id", "item_id", "user_last", "user_last_buy"]
    ]
    numeric += ["first_time_days_ago", "last_time_days_ago", "last_buy_days_ago"]
    numeric = list(dict.fromkeys(numeric))
    raw = frame[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = raw.iloc[train].median().fillna(0)
    # 统一log1p减少长尾量级；所有填补和缩放参数仅训练拟合。
    values = np.log1p(raw.fillna(medians).clip(lower=0)).to_numpy(dtype=np.float32)
    te, target_encoder = encode_category(frame.category_id, y, train, seed)
    values = np.column_stack([values, te])
    numeric += ["category_target_encoding"]
    scaler = StandardScaler().fit(values[train])
    x = scaler.transform(values).astype(np.float32)
    training_users = set(frame.iloc[train].user_id)
    item_frequency = Counter(frame.iloc[train].item_id)
    for user in training_users:
        for _, items, _ in histories.get(user, []):
            item_frequency.update(items)
    item_vocab = {
        item: i + 2 for i, (item, _) in enumerate(item_frequency.most_common(20000))
    }
    training_categories = set(frame.iloc[train].category_id)
    for item in item_frequency:
        training_categories.add(category_map.get(item, "unknown"))
    category_vocab = {c: i + 2 for i, c in enumerate(sorted(training_categories))}
    user_list = sorted(histories)
    user_index = {u: i for i, u in enumerate(user_list)}
    length, per_hour = 16, 4
    sequence_items = np.zeros((len(user_list), length, per_hour), dtype=np.int64)
    sequence_categories = np.zeros_like(sequence_items)
    sequence_aux = np.zeros((len(user_list), length, 5), dtype=np.float32)
    sequence_mask = np.zeros((len(user_list), length), dtype=bool)
    item_sets = []
    for u in user_list:
        row_index = user_index[u]
        sets = []
        for j, (hour, items, types) in enumerate(histories[u]):
            sequence_mask[row_index, j] = True
            sets.append(set(items))
            for k, item in enumerate(items):
                sequence_items[row_index, j, k] = item_vocab.get(item, 1)
                sequence_categories[row_index, j, k] = category_vocab.get(
                    category_map.get(item, "unknown"), 1
                )
            sequence_aux[row_index, j, :4] = types / max(types.sum(), 1)
            sequence_aux[row_index, j, 4] = np.log1p(
                (pd.Timestamp(cutoff) - pd.Timestamp(hour)).total_seconds() / 3600
            ) / np.log1p(168)
        item_sets.append(sets)
    row_users = np.array([user_index[u] for u in frame.user_id], dtype=np.int64)
    matches = np.zeros((len(frame), length), dtype=np.float32)
    for i, (u, item) in enumerate(zip(row_users, frame.item_id)):
        for j, items in enumerate(item_sets[u]):
            matches[i, j] = item in items
    target_items = np.array(
        [item_vocab.get(i, 1) for i in frame.item_id], dtype=np.int64
    )
    target_categories = np.array(
        [category_vocab.get(c, 1) for c in frame.category_id], dtype=np.int64
    )
    frame["split"] = "train"
    frame.loc[valid, "split"] = "validation"
    frame.loc[test, "split"] = "test"
    frame.to_parquet(output / "pair_samples.parquet", index=False)
    np.savez_compressed(
        output / "dataset.npz",
        x=x,
        y=y,
        train=train,
        valid=valid,
        test=test,
        row_users=row_users,
        target_items=target_items,
        target_categories=target_categories,
        sequence_items=sequence_items,
        sequence_categories=sequence_categories,
        sequence_aux=sequence_aux,
        sequence_mask=sequence_mask,
        matches=matches,
    )
    joblib.dump(
        {
            "medians": medians,
            "scaler": scaler,
            "target_encoder": target_encoder,
            "features": numeric,
            "item_vocab": item_vocab,
            "category_vocab": category_vocab,
        },
        output / "preprocessing.joblib",
    )
    summary = {
        "cutoff": cutoff,
        "label_end_exclusive": end,
        "candidate_history_days": 7,
        "candidate_pool_pairs": total,
        "candidate_pool_positive_pairs": positive,
        "all_future_purchase_pairs": all_future,
        "candidate_future_purchase_pair_coverage": positive / all_future,
        "sampling": "uniform reservoir independent of future labels",
        "seed": seed,
        "feature_count": len(numeric),
        "features": numeric,
        "sample_pairs": len(frame),
        "sequence_hours": length,
        "items_per_hour_cap": per_hour,
        "item_vocabulary": len(item_vocab) + 2,
        "category_vocabulary": len(category_vocab) + 2,
        "splits": {},
    }
    for name, index in [("train", train), ("validation", valid), ("test", test)]:
        summary["splits"][name] = {
            "rows": len(index),
            "positives": int(y[index].sum()),
            "positive_rate": float(y[index].mean()),
            "target_item_oov_rate": float((target_items[index] == 1).mean()),
            "users_unseen_in_train": len(
                set(frame.iloc[index].user_id) - training_users
            ),
        }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("样本完成：%s", summary["splits"])
    return summary
