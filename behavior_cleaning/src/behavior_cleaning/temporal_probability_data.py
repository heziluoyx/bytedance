"""固定历史、非重叠标签窗口及训练期编码的购买概率样本。"""
import gc
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from behavior_cleaning.promotion_robustness import NUMERIC
from behavior_cleaning.traditional_dataset import encode_category

WINDOWS = [
    ("train", "2025-11-21"),
    ("validation", "2025-11-28"),
    ("meta_calibration", "2025-12-05"),
    ("test", "2025-12-12"),
]
FEATURES = NUMERIC + ["category_target_encoding"]


def window_bounds(cutoff, history_days=3):
    cutoff = pd.Timestamp(cutoff)
    return (
        cutoff - pd.Timedelta(days=history_days),
        cutoff,
        cutoff + pd.Timedelta(days=7),
    )


def calibration_user(user_id):
    """仅用标识的稳定哈希分组，不参考后续购买标签。"""
    return (
        int(hashlib.sha256(("20261109:" + str(user_id)).encode()).hexdigest()[:8], 16)
        % 2
        == 0
    )


def open_events(source, temporary):
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='3GB'")
    c.execute(
        "SET temp_directory='"
        + str(temporary).replace("\\", "/").replace("'", "''")
        + "'"
    )
    p = str(Path(source)).replace("\\", "/").replace("'", "''")
    c.execute(
        "CREATE VIEW events AS SELECT user_id,item_id,category_id,behavior_type,timestamp event_time FROM read_parquet('"
        + p
        + "')"
    )
    return c


def snapshot(c, cutoff, maximum, seed=42, *, labeled=True):
    """构建三天历史快照；推理用labeled=False，完全不查询未来行为。

    labeled=True保持既有实验标签与抽样规则；False允许只有历史的数据，
    返回结果不含target，也不要求出现两类标签。
    """
    start, cutoff, finish = window_bounds(cutoff)
    c.execute(
        "CREATE OR REPLACE TEMP TABLE history AS SELECT * FROM events WHERE event_time>=? AND event_time<?",
        [start, cutoff],
    )
    c.execute(
        """CREATE OR REPLACE TEMP TABLE pairs AS SELECT user_id,item_id,
        MIN(category_id) category_id,COUNT(*) pair_recent_events,
        SUM(behavior_type=1) pair_pv,SUM(behavior_type=2) pair_fav,
        SUM(behavior_type=3) pair_cart,SUM(behavior_type=4) pair_buys,
        COUNT(DISTINCT CAST(event_time AS DATE)) pair_days,
        MIN(event_time) first_time,MAX(event_time) last_time,
        MAX(CASE WHEN behavior_type=4 THEN event_time END) last_buy
        FROM history GROUP BY user_id,item_id"""
    )
    if labeled:
        c.execute(
            """CREATE OR REPLACE TEMP TABLE future AS SELECT user_id,item_id FROM events
        WHERE event_time>=? AND event_time<? AND behavior_type=4 GROUP BY user_id,item_id""",
            [cutoff, finish],
        )
    else:
        c.execute(
            "CREATE OR REPLACE TEMP TABLE future AS "
            "SELECT user_id,item_id FROM history WHERE FALSE"
        )
    pool = c.execute(
        "SELECT COUNT(*),COUNT(f.item_id) FROM pairs p LEFT JOIN future f USING(user_id,item_id)"
    ).fetchone()
    future_count = c.execute("SELECT COUNT(*) FROM future").fetchone()[0]
    # 资格与抽样均只由历史标识决定，排序不读取target。
    frame = c.execute(
        """WITH selected AS (SELECT * FROM pairs ORDER BY hash(user_id,item_id,?) LIMIT ?),
      users AS (SELECT user_id,COUNT(*) user_events,COUNT(DISTINCT item_id) user_items,
       COUNT(DISTINCT CAST(event_time AS DATE)) user_days,SUM(behavior_type=1) user_pv,
       SUM(behavior_type=2) user_fav,SUM(behavior_type=3) user_cart,SUM(behavior_type=4) user_buys,
       MAX(event_time) user_last,MAX(CASE WHEN behavior_type=4 THEN event_time END) user_last_buy
       FROM history GROUP BY user_id),
      items AS (SELECT item_id,COUNT(*) item_events,SUM(behavior_type=4) item_buys,
       COUNT(DISTINCT user_id) item_users FROM history GROUP BY item_id)
      SELECT p.*,u.* EXCLUDE(user_id),i.* EXCLUDE(item_id),
       CAST(f.item_id IS NOT NULL AS INTEGER) AS "target" FROM selected p
       JOIN users u USING(user_id) JOIN items i USING(item_id)
       LEFT JOIN future f USING(user_id,item_id)""",
        [seed, maximum],
    ).df()
    for field in ["first_time", "last_time", "last_buy", "user_last", "user_last_buy"]:
        frame[field + "_days_ago"] = (
            cutoff - pd.to_datetime(frame[field])
        ).dt.total_seconds() / 86400
    frame["pair_buy_share"] = frame.pair_buys / frame.pair_recent_events
    frame["item_buy_share"] = frame.item_buys / frame.item_events
    frame["user_buy_share"] = frame.user_buys / frame.user_events
    frame["pair_user_event_share"] = frame.pair_recent_events / frame.user_events
    frame["cutoff"] = cutoff
    info = {
        "history_start": str(start),
        "cutoff": str(cutoff),
        "label_end_exclusive": str(finish),
        "candidate_pairs": int(pool[0]),
        "candidate_positives": int(pool[1]),
        "all_future_purchase_pairs": int(future_count),
        "candidate_purchase_coverage": pool[1] / max(future_count, 1),
        "sample_pairs": len(frame),
        "positives": int(frame.target.sum()),
    }
    if (labeled and frame.target.nunique() != 2) or frame.duplicated(
        ["user_id", "item_id"]
    ).any():
        raise ValueError("快照缺少两类标签或出现重复用户商品对")
    if not labeled:
        frame = frame.drop(columns="target")
        for key in [
            "candidate_positives",
            "all_future_purchase_pairs",
            "candidate_purchase_coverage",
            "positives",
        ]:
            info.pop(key)
        info["labeled"] = False
    return frame, info


def collect_sequences(c):
    """保留每用户最后16个活跃小时，每小时最多4件商品并池化。"""
    rows = c.execute(
        """WITH hours AS (SELECT DISTINCT user_id,event_time FROM history),
      recent AS (SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY event_time DESC) r FROM hours),
      counts AS (SELECT h.user_id,h.event_time,h.item_id,MIN(h.category_id) category_id,
        SUM(CASE behavior_type WHEN 1 THEN 1 WHEN 2 THEN 2 WHEN 3 THEN 3 ELSE 5 END) weight,
        SUM(behavior_type=1) pv,SUM(behavior_type=2) fav,SUM(behavior_type=3) cart,SUM(behavior_type=4) buy
        FROM history h JOIN recent r USING(user_id,event_time) WHERE r.r<=16
        GROUP BY h.user_id,h.event_time,h.item_id),
      ranked AS (SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id,event_time ORDER BY weight DESC,item_id) k,
        SUM(pv) OVER(PARTITION BY user_id,event_time) total_pv,
        SUM(fav) OVER(PARTITION BY user_id,event_time) total_fav,
        SUM(cart) OVER(PARTITION BY user_id,event_time) total_cart,
        SUM(buy) OVER(PARTITION BY user_id,event_time) total_buy FROM counts)
      SELECT user_id,event_time,item_id,category_id,total_pv,total_fav,total_cart,total_buy
      FROM ranked WHERE k<=4 ORDER BY user_id,event_time,k"""
    ).fetchall()
    result = defaultdict(list)
    key = None
    for user, hour, item, category, pv, fav, cart, buy in rows:
        if (user, hour) != key:
            result[user].append([hour, [], np.array([pv, fav, cart, buy], np.float32)])
            key = (user, hour)
        result[user][-1][1].append((item, category))
    return dict(result)


def build_temporal_dataset(source, output, maximum=80000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "dataset"
    data_dir.mkdir(exist_ok=True)
    if (data_dir / "summary.json").exists():
        info = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
        if info["max_pairs"] != maximum or info["source"] != str(
            Path(source).resolve()
        ):
            raise ValueError("已有数据配置不匹配")
        return info
    temporary = output / "duckdb_temp"
    temporary.mkdir(exist_ok=True)
    c = open_events(source, temporary)
    minimum, maximum_time = c.execute(
        "SELECT MIN(event_time),MAX(event_time) FROM events"
    ).fetchone()
    if pd.Timestamp(minimum) > pd.Timestamp("2025-11-18") or pd.Timestamp(
        maximum_time
    ) < pd.Timestamp("2025-12-18 23:00:00"):
        raise ValueError("缺少完整历史或7天标签期")
    frames = []
    infos = []
    histories = []
    for name, cutoff in WINDOWS:
        logging.info("构建时间快照 %s %s", name, cutoff)
        f, info = snapshot(c, cutoff, maximum)
        f["split"] = name
        if name == "meta_calibration":
            f["split"] = np.where(
                f.user_id.map(calibration_user), "calibration", "meta"
            )
        frames.append(f)
        infos.append({"window": name, **info})
        histories.append(collect_sequences(c))
    c.close()
    frame = pd.concat(frames, ignore_index=True)
    train = np.flatnonzero(frame.split.eq("train"))
    valid = np.flatnonzero(frame.split.eq("validation"))
    y = frame.target.to_numpy(np.int64)
    raw = frame[NUMERIC].replace([np.inf, -np.inf], np.nan)
    medians = raw.iloc[train].median().fillna(0)
    numeric = np.log1p(raw.fillna(medians).clip(lower=0)).to_numpy(np.float32)
    encoded, encoder = encode_category(frame.category_id, y, train)
    values = np.column_stack([numeric, encoded])
    scaler = StandardScaler().fit(values[train])
    x = scaler.transform(values).astype(np.float32)
    frequency = Counter()
    for events in histories[0].values():
        for _, items, _ in events:
            frequency.update(item for item, _ in items)
    item_vocab = {
        item: i + 2 for i, (item, _) in enumerate(frequency.most_common(20000))
    }
    categories = set(frames[0].category_id)
    for events in histories[0].values():
        for _, items, _ in events:
            categories.update(cat for _, cat in items)
    category_vocab = {cat: i + 2 for i, cat in enumerate(sorted(categories))}
    sequences = []
    sequence_categories = []
    aux = []
    masks = []
    sets = []
    user_lookup = {}
    for w, ((_, cutoff), history) in enumerate(zip(WINDOWS, histories)):
        for user, events in sorted(history.items()):
            u = len(sequences)
            user_lookup[(w, user)] = u
            a = np.zeros((16, 4), np.int64)
            b = np.zeros_like(a)
            v = np.zeros((16, 5), np.float32)
            m = np.zeros(16, bool)
            ss = []
            for j, (hour, items, types) in enumerate(events):
                m[j] = True
                ss.append({item for item, _ in items})
                for k, (item, cat) in enumerate(items):
                    a[j, k] = item_vocab.get(item, 1)
                    b[j, k] = category_vocab.get(cat, 1)
                v[j, :4] = types / max(float(types.sum()), 1)
                v[j, 4] = np.log1p(
                    (pd.Timestamp(cutoff) - pd.Timestamp(hour)).total_seconds() / 3600
                ) / np.log1p(72)
            sequences.append(a)
            sequence_categories.append(b)
            aux.append(v)
            masks.append(m)
            sets.append(ss)
    window_ids = np.concatenate([np.full(len(f), i) for i, f in enumerate(frames)])
    row_users = np.array(
        [user_lookup[(w, u)] for w, u in zip(window_ids, frame.user_id)], np.int64
    )
    matches = np.zeros((len(frame), 16), np.float32)
    for i, (u, item) in enumerate(zip(row_users, frame.item_id)):
        for j, items in enumerate(sets[u]):
            matches[i, j] = item in items
    arrays = {
        "x": x,
        "y": y,
        "train": train,
        "valid": valid,
        "row_users": row_users,
        "target_items": np.array(
            [item_vocab.get(i, 1) for i in frame.item_id], np.int64
        ),
        "target_categories": np.array(
            [category_vocab.get(i, 1) for i in frame.category_id], np.int64
        ),
        "sequence_items": np.stack(sequences),
        "sequence_categories": np.stack(sequence_categories),
        "sequence_aux": np.stack(aux),
        "sequence_mask": np.stack(masks),
        "matches": matches,
    }
    split_info = {}
    for name in ["train", "validation", "meta", "calibration", "test"]:
        ix = np.flatnonzero(frame.split.eq(name))
        arrays[name if name != "validation" else "valid"] = ix
        if len(np.unique(y[ix])) != 2:
            raise ValueError("分组缺少两类标签 " + name)
        split_info[name] = {
            "rows": len(ix),
            "positives": int(y[ix].sum()),
            "positive_rate": float(y[ix].mean()),
            "item_oov": float(np.mean(arrays["target_items"][ix] == 1)),
        }
    np.savez_compressed(data_dir / "arrays.npz", **arrays)
    frame.to_parquet(data_dir / "samples.parquet", index=False)
    joblib.dump(
        {
            "medians": medians,
            "scaler": scaler,
            "target_encoder": encoder,
            "features": FEATURES,
            "item_vocab": item_vocab,
            "category_vocab": category_vocab,
        },
        data_dir / "preprocessing.joblib",
    )
    info = {
        "source": str(Path(source).resolve()),
        "history_days": 3,
        "target_days": 7,
        "max_pairs": maximum,
        "windows": infos,
        "splits": split_info,
        "features": FEATURES,
        "item_vocabulary": len(item_vocab) + 2,
        "category_vocabulary": len(category_vocab) + 2,
        "target": "user-item pair purchase within next 7 days",
        "sampling": "stable hash of historical user/item IDs, independent of labels",
        "meta_calibration": "same date, disjoint user-hash groups; all labels mature before final test",
    }
    (data_dir / "summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("时间数据集完成 %s", split_info)
    return info
