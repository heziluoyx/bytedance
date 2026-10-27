"""历史快照、稀疏隐式特征及有界共现规则。"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from behavior_cleaning.analysis_runner import connect_readonly, summarize_sequence
from behavior_cleaning.analysis_sql import QUERIES


def scoped_connection(database: Path, cutoff: str) -> sqlite3.Connection:
    """建立严格早于截止时间的历史视图；参数只写入临时表。"""
    connection = connect_readonly(database)
    connection.execute("CREATE TEMP TABLE feature_cutoff(value TEXT)")
    connection.execute("INSERT INTO feature_cutoff VALUES (?)", (cutoff,))
    has_covering_index = connection.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='index' AND name=?",
        ("idx_behavior_quadruple",),
    ).fetchone()
    # 固定已知覆盖索引，避免按时间索引逐行回表；不接受外部索引名。
    source = "main.user_behavior b"
    if has_covering_index:
        source += " INDEXED BY idx_behavior_quadruple"
    connection.execute(
        "CREATE TEMP VIEW user_behavior AS SELECT b.* FROM "
        + source
        + " WHERE event_time<(SELECT value FROM feature_cutoff)"
    )
    connection.execute(
        "CREATE TEMP VIEW dim_time AS SELECT t.* FROM main.dim_time t "
        + "WHERE event_time<(SELECT value FROM feature_cutoff)"
    )
    return connection


def snapshot(database: Path, cutoff: str) -> pd.DataFrame:
    """从历史生成用户生命周期、序列和严格路径漏斗特征。"""
    with scoped_connection(database, cutoff) as connection:
        logging.info("历史用户生命周期聚合")
        users = pd.read_sql_query(QUERIES["user_features"], connection)
        # 以预先指定的预测时点计算间隔，而非用户各自最后活跃日期。
        users["inactivity_days"] = (
            pd.Timestamp(cutoff) - pd.to_datetime(users["last_time"])
        ).dt.total_seconds() / 86400
        users["purchase_recency_days"] = (
            pd.Timestamp(cutoff) - pd.to_datetime(users["last_buy_time"])
        ).dt.total_seconds() / 86400
        logging.info("历史小时桶序列聚合")
        sequence = pd.read_sql_query(QUERIES["hour_sequence"], connection)
        _, sequence_users = summarize_sequence(sequence)
        users = users.merge(sequence_users, on="user_id", validate="one_to_one")
        identifiable = sequence["mask"].isin([1, 2, 4, 8]) & sequence[
            "previous_mask"
        ].isin([1, 2, 4, 8])
        transition_counts = pd.crosstab(
            sequence.loc[identifiable, "user_id"],
            sequence.loc[identifiable, "previous_mask"].astype(int).astype(str)
            + "_"
            + sequence.loc[identifiable, "mask"].astype(str),
        )
        for source in [1, 2, 4, 8]:
            columns = [str(source) + "_" + str(target) for target in [1, 2, 4, 8]]
            counts = transition_counts.reindex(columns=columns, fill_value=0)
            probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
            probabilities.columns = ["transition_" + name for name in columns]
            users = users.merge(
                probabilities,
                left_on="user_id",
                right_index=True,
                how="left",
                validate="one_to_one",
            )
        logging.info("历史主偏好类目聚合")
        dominant = pd.read_sql_query(
            "WITH counts AS (SELECT b.user_id,i.category_id,COUNT(*) n "
            + "FROM user_behavior b JOIN dim_item i USING(item_id) "
            + "GROUP BY b.user_id,i.category_id), ranked AS (SELECT *,"
            + "ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY n DESC,category_id) r "
            + "FROM counts) SELECT user_id,category_id dominant_category "
            + "FROM ranked WHERE r=1",
            connection,
        )
        users = users.merge(dominant, on="user_id", validate="one_to_one")
        prefix = QUERIES["ordered_funnel"].split("SELECT 'pv' stage")[0]
        logging.info("历史用户商品严格路径漏斗")
        funnel = pd.read_sql_query(
            prefix
            + "SELECT p.user_id,COUNT(*) funnel_pv_pairs,"
            + "COUNT(c.item_id) funnel_cart_pairs,COUNT(f.item_id) funnel_fav_pairs,"
            + "COUNT(b.item_id) funnel_buy_pairs FROM pv p "
            + "LEFT JOIN cart c USING(user_id,item_id) "
            + "LEFT JOIN fav f USING(user_id,item_id) "
            + "LEFT JOIN buy b USING(user_id,item_id) GROUP BY p.user_id",
            connection,
        )
        users = users.merge(funnel, on="user_id", how="left", validate="one_to_one")
    for earlier, later in [("pv", "cart"), ("cart", "fav"), ("fav", "buy")]:
        users["funnel_" + earlier + "_to_" + later] = users[
            "funnel_" + later + "_pairs"
        ] / users["funnel_" + earlier + "_pairs"].replace(0, np.nan)
    # 固定业务分段与价值代理，不使用验证期分位数拟合阈值。
    users["activity_level"] = pd.cut(
        users["active_days"], [0, 7, 20, np.inf], labels=["low", "medium", "high"]
    )
    users["rf_proxy_score"] = (
        np.log1p(users["buy_count"]) / (1 + users["purchase_recency_days"])
    ).fillna(0)
    users["cutoff"] = cutoff
    return users


def labels(database: Path, cutoff: str) -> pd.DataFrame:
    """标签为指定时点起未来完整7天内是否有购买事件。"""
    finish = pd.Timestamp(cutoff) + pd.Timedelta(days=7)
    with connect_readonly(database) as connection:
        latest = pd.read_sql_query("SELECT MAX(event_time) t FROM dim_time", connection)
        if finish > pd.Timestamp(latest.iloc[0, 0]) + pd.Timedelta(hours=1):
            raise ValueError("标签观察期不足7天，禁止将截断用户标记为未购买")
        return pd.read_sql_query(
            "SELECT user_id,1 target FROM user_behavior WHERE behavior_type=4 "
            + "AND event_time>=? AND event_time<? GROUP BY user_id",
            connection,
            params=(cutoff, finish.strftime("%Y-%m-%d %H:%M:%S")),
        )


def interactions(database: Path, cutoff: str, vocabulary: list[str]) -> pd.DataFrame:
    """聚合历史交互，保留正常重复，用业务权重降低后续稀疏存储成本。"""
    with scoped_connection(database, cutoff) as connection:
        connection.execute("CREATE TEMP TABLE vocabulary(item_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO vocabulary VALUES (?)", [(item,) for item in vocabulary]
        )
        return pd.read_sql_query(
            "SELECT b.user_id,b.item_id,SUM(CASE behavior_type WHEN 4 THEN 5 "
            + "WHEN 3 THEN 3 WHEN 2 THEN 2 ELSE 1 END) weight "
            + "FROM user_behavior b JOIN vocabulary v USING(item_id) "
            + "GROUP BY b.user_id,b.item_id",
            connection,
        )


def matrix(
    frame: pd.DataFrame, users: list[str], items: list[str]
) -> sparse.csr_matrix:
    """按固定词表创建 float32 CSR，使用 log1p 缓和高频重复权重。"""
    row = pd.Index(users).get_indexer(frame["user_id"])
    col = pd.Index(items).get_indexer(frame["item_id"])
    keep = (row >= 0) & (col >= 0)
    return sparse.csr_matrix(
        (
            np.log1p(frame.loc[keep, "weight"].to_numpy(dtype=np.float32)),
            (row[keep], col[keep]),
        ),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )


def implicit_features(
    database: Path, snapshots: list[pd.DataFrame], output: Path, max_items: int = 1000
) -> tuple[list[pd.DataFrame], dict]:
    """训练历史拟合 SVD；在训练热门商品子集计算有界共现和规则。

    此入口为资源可控的原型。max_items 限制须写入覆盖率，不能宣称全量覆盖。
    """
    training = snapshots[0]
    cutoff = training["cutoff"].iloc[0]
    with scoped_connection(database, cutoff) as connection:
        vocabulary = pd.read_sql_query(
            "SELECT item_id,COUNT(DISTINCT user_id) users FROM user_behavior "
            + "GROUP BY item_id ORDER BY users DESC,item_id LIMIT ?",
            connection,
            params=(max_items,),
        )["item_id"].tolist()
    if len(vocabulary) < 2:
        raise ValueError("SVD 至少需要两个观测商品")
    users = training["user_id"].tolist()
    matrices = []
    coverage = []
    for frame in snapshots:
        logging.info("聚合快照交互：%s", frame["cutoff"].iloc[0])
        pairs = interactions(database, frame["cutoff"].iloc[0], vocabulary)
        current = matrix(pairs, users, vocabulary)
        matrices.append(current)
        coverage.append(
            {
                "cutoff": frame["cutoff"].iloc[0],
                "covered_users": int((current.getnnz(axis=1) > 0).sum()),
                "user_count": len(users),
                "covered_pairs": int(current.nnz),
                "covered_pair_share": float(
                    current.nnz / frame["distinct_items"].sum()
                ),
                "covered_business_weight_share": float(
                    np.expm1(current.data.astype(np.float64)).sum()
                    / frame["preference_weight_sum"].sum()
                ),
            }
        )
    components = min(16, len(vocabulary) - 1, len(users) - 1)
    decomposition = TruncatedSVD(n_components=components, random_state=42)
    decomposition.fit(matrices[0])
    item_embeddings = decomposition.components_.T
    pd.DataFrame(item_embeddings, index=pd.Index(vocabulary, name="item_id")).to_csv(
        output / "item_svd_embeddings.csv", encoding="utf-8-sig"
    )
    binary = matrices[0].copy()
    binary.data[:] = 1
    cooccurrence = (binary.T @ binary).tocoo()
    support_counts = np.asarray(binary.sum(axis=0)).ravel()
    valid = (cooccurrence.row != cooccurrence.col) & (cooccurrence.data >= 5)
    source, target, counts = (
        cooccurrence.row[valid],
        cooccurrence.col[valid],
        cooccurrence.data[valid],
    )
    cosine = counts / np.sqrt(support_counts[source] * support_counts[target])
    confidence = counts / support_counts[source]
    rules = pd.DataFrame(
        {
            "source_item": np.asarray(vocabulary)[source],
            "target_item": np.asarray(vocabulary)[target],
            "cooccurrence_users": counts,
            "cosine": cosine,
            "support": counts / len(users),
            "confidence": confidence,
            "lift": confidence / (support_counts[target] / len(users)),
        }
    )
    rules.to_csv(output / "item_similarity_association_rules.csv", index=False)
    similarity = sparse.csr_matrix(
        (cosine, (source, target)), shape=(len(vocabulary), len(vocabulary))
    )
    transformed = []
    for frame, current in zip(snapshots, matrices):
        embedding = decomposition.transform(current)
        extra = pd.DataFrame(
            embedding,
            index=pd.Index(users, name="user_id"),
            columns=["svd_" + str(i) for i in range(components)],
        )
        # 用户历史向量与商品向量余弦，逐个已观测用户商品对计算，避免笛卡尔积。
        row, col = current.nonzero()
        affinity = np.einsum(
            "ij,ij->i", normalize(embedding)[row], normalize(item_embeddings)[col]
        )
        observed = pd.DataFrame(
            {
                "user_id": np.asarray(users)[row],
                "item_id": np.asarray(vocabulary)[col],
                "latent_cosine": affinity,
            }
        )
        raw_weights = current.copy()
        raw_weights.data = np.expm1(raw_weights.data)
        weighted_totals = np.asarray(raw_weights.sum(axis=1)).ravel()
        observed["weighted_interaction_count"] = raw_weights.data
        observed[
            "preference_strength_within_covered_items"
        ] = raw_weights.data / np.maximum(weighted_totals[row], 1)
        recommendation = current @ similarity
        observed["user_item_cooccurrence_affinity"] = np.asarray(
            recommendation[row, col]
        ).ravel() / np.maximum(np.asarray(current.sum(axis=1)).ravel()[row], 1)
        observed.to_csv(
            output / ("user_item_similarity_" + frame["cutoff"].iloc[0][:10] + ".csv"),
            index=False,
        )
        extra["observed_item_affinity_mean"] = (
            observed.groupby("user_id")["latent_cosine"]
            .mean()
            .reindex(users)
            .to_numpy()
        )
        extra["cooccurrence_score_mean"] = np.asarray(
            recommendation.sum(axis=1)
        ).ravel() / max(1, len(vocabulary))
        transformed.append(
            frame.merge(
                extra,
                left_on="user_id",
                right_index=True,
                how="left",
                validate="one_to_one",
            )
        )
    return transformed, {
        "vocabulary_size": len(vocabulary),
        "max_items": max_items,
        "components": components,
        "coverage": coverage,
        "min_cooccurrence_users": 5,
        "rules": len(rules),
        "scope": "training_popular_item_subset_not_full_catalog",
    }
