"""SQLite 报表导出与结果审计；不修改源数据库。"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_cleaning.analysis_sql import QUERIES


def connect_readonly(database: Path) -> sqlite3.Connection:
    """创建只读连接，限制内存缓存。"""
    connection = sqlite3.connect(database.absolute().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def summarize_sequence(sequence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅对相邻且均为单一行为的小时桶统计可识别跳转。"""
    valid = sequence["mask"].isin([1, 2, 4, 8]) & sequence["previous_mask"].isin(
        [1, 2, 4, 8]
    )
    transitions = (
        sequence.loc[valid]
        .groupby(["previous_mask", "mask"])
        .size()
        .rename("count")
        .reset_index()
    )
    transitions["probability"] = transitions["count"] / transitions.groupby(
        "previous_mask"
    )["count"].transform("sum")
    sequence = sequence.copy()
    if "event_time" in sequence:
        sequence = sequence.sort_values(["user_id", "event_time"])
    sequence["identifiable_pair"] = valid.astype(int)
    sequence["same_behavior_pair"] = (
        valid & sequence["mask"].eq(sequence["previous_mask"])
    ).astype(int)
    sequence["pattern_segment"] = (sequence["same_behavior_pair"] == 0).cumsum()
    pattern_lengths = (
        sequence[sequence["mask"].isin([1, 2, 4, 8])]
        .groupby(["user_id", "pattern_segment"])
        .size()
        .groupby("user_id")
        .max()
        .rename("longest_identifiable_behavior_run")
    )
    users = sequence.groupby("user_id").agg(
        observed_hour_bins=("mask", "size"),
        mean_distinct_hour_gap=("gap_hours", "mean"),
        max_distinct_hour_gap=("gap_hours", "max"),
        identifiable_pairs=("identifiable_pair", "sum"),
        same_behavior_pairs=("same_behavior_pair", "sum"),
    )
    return transitions, users.join(pattern_lengths).reset_index()


def add_rf_groups(users: pd.DataFrame) -> pd.DataFrame:
    """生成 RF 代理分群；没有消费金额，不声称是真实 RFM。"""
    users = users.copy()
    frequency = users["buy_count"].rank(pct=True)
    recency = (
        users["purchase_recency_days"].fillna(np.inf).rank(pct=True, ascending=False)
    )
    users["rf_proxy_score"] = (frequency + recency) / 2
    users["rf_proxy_group"] = np.select(
        [
            (users["buy_count"] > 0) & (users["rf_proxy_score"] >= 0.75),
            users["inactivity_days"] >= 14,
            users["inactivity_days"] >= 7,
        ],
        ["高价值代理", "沉睡", "流失风险"],
        default="潜力",
    )
    users["activity_level"] = pd.cut(
        users["active_days"], bins=[0, 7, 20, np.inf], labels=["低", "中", "高"]
    )
    return users


def export_query(connection: sqlite3.Connection, query: str, path: Path) -> int:
    """分批导出大表，避免同时持有全量商品特征。"""
    total = 0
    for index, frame in enumerate(
        pd.read_sql_query(query, connection, chunksize=50000)
    ):
        if "recent_heat_ratio" in frame:
            frame["two_window_heat_decay_per_day"] = (
                -np.log(frame["recent_heat_ratio"]) / 3.5
            )
        frame.to_csv(
            path,
            index=False,
            mode="w" if index == 0 else "a",
            header=index == 0,
            encoding="utf-8-sig" if index == 0 else "utf-8",
        )
        total += len(frame)
    return total


def analyze(database: Path, output: Path, calendar: Path | None = None) -> dict:
    """执行分析并输出 CSV、图和审计说明。"""
    output.mkdir(parents=True, exist_ok=False)
    summary: dict = {"database": str(database), "reports": {}, "limitations": []}
    with connect_readonly(database) as connection:
        summary["bounds"] = (
            pd.read_sql_query(
                "SELECT MIN(event_time) first_time,MAX(event_time) last_time,"
                + "COUNT(*) rows FROM user_behavior",
                connection,
            )
            .iloc[0]
            .to_dict()
        )
        for name, query in QUERIES.items():
            logging.info("SQLite 分析：%s", name)
            if name == "item_lifecycle":
                summary["reports"][name] = export_query(
                    connection, query, output / (name + ".csv")
                )
                continue
            frame = pd.read_sql_query(query, connection)
            if name == "user_features":
                frame = add_rf_groups(frame)
                groups = frame.groupby("rf_proxy_group").agg(
                    users=("user_id", "size"),
                    events_per_user=("events", "mean"),
                    buy_events_per_user=("buy_count", "mean"),
                    mean_active_days=("active_days", "mean"),
                    buyer_users=("buy_count", lambda values: int((values > 0).sum())),
                )
                groups["buyer_share"] = groups["buyer_users"] / groups["users"]
                groups.to_csv(output / "rf_group_comparison.csv", encoding="utf-8-sig")
            if name == "ordered_funnel":
                previous = frame["user_item_pairs"].shift()
                frame["conversion_from_previous"] = frame["user_item_pairs"] / previous
                frame["drop_from_previous"] = previous - frame["user_item_pairs"]
            if name == "hour_sequence":
                transitions, user_sequence = summarize_sequence(frame)
                transitions.to_csv(output / "transition_probabilities.csv", index=False)
                user_sequence.to_csv(output / "user_sequence_features.csv", index=False)
                summary["sequence_coverage"] = {
                    "hour_bins": len(frame),
                    "adjacent_pairs": int(frame["previous_mask"].notna().sum()),
                    "identifiable_pairs": int(transitions["count"].sum()),
                }
                # 小时桶用于序列分析，无须额外复制几百万原始事件。
            frame.to_csv(output / (name + ".csv"), index=False, encoding="utf-8-sig")
            summary["reports"][name] = len(frame)
        summary["limitations"] = [
            "无金额字段：RF 代理分不是真实 RFM；购买事件数不等于订单数。",
            "小时精度：gap 为小时桶差；同小时先后未知，严格漏斗仅统计更晚小时。",
            "活跃度阈值7/20天、风险7/14天为演示业务规则，需结合业务校准。",
            "复购采用跨日再次购买的用户商品对占比；不是订单复购率。",
            "热度为近3日/前4日日均行为比；衰减为两窗口对数比/3.5天的代理估计。",
            "全期描述性报表不可直接用于未来7天预测；预测须使用历史快照。",
            "严格漏斗 pv→cart→fav→buy 是指定路径，不代表购买必经顺序。",
            "小时买家占比描述关联，不代表该小时浏览带来的因果转化。",
        ]
        if calendar is None:
            summary["limitations"].append("未提供地区节假日日历：节假日分析未完成。")
        else:
            holiday_analysis(connection, calendar, output)
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_plots(output)
    return summary


def holiday_analysis(
    connection: sqlite3.Connection, calendar: Path, output: Path
) -> None:
    """合并用户提供的 date,is_holiday,region 日历，拒绝缺日/混地区。"""
    dates = pd.read_csv(calendar, dtype={"date": str, "region": str})
    if not {"date", "is_holiday", "region"}.issubset(dates.columns):
        raise ValueError("日历必须包含 date,is_holiday,region")
    if dates["date"].duplicated().any() or dates["region"].nunique() != 1:
        raise ValueError("日历日期必须唯一且只能使用一个地区")
    if not dates["is_holiday"].isin([0, 1]).all():
        raise ValueError("is_holiday 只能为0或1")
    daily = pd.read_sql_query(
        "SELECT date(event_time) date,COUNT(*) events,"
        + "COUNT(DISTINCT user_id) users,SUM(behavior_type=4) buys "
        + "FROM user_behavior GROUP BY date(event_time)",
        connection,
    ).merge(dates, on="date", how="left", validate="one_to_one")
    if daily["is_holiday"].isna().any():
        raise ValueError("日历没有覆盖全部观测日期")
    daily.to_csv(output / "holiday_daily.csv", index=False, encoding="utf-8-sig")
    daily.groupby("is_holiday").agg(
        days=("date", "size"),
        events_per_day=("events", "mean"),
        users_per_day=("users", "mean"),
        buys_per_day=("buys", "mean"),
    ).to_csv(output / "holiday_comparison.csv", encoding="utf-8-sig")


def make_plots(output: Path) -> None:
    """输出基础趋势图；数字与完整明细以 CSV 为准。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, x, y in [
        ("behavior_distribution", "behavior_type", "event_count"),
        ("daily_activity", "day", "dau"),
        ("hour_activity", "hour", "buyer_user_share"),
        ("ordered_funnel", "stage", "user_item_pairs"),
    ]:
        frame = pd.read_csv(output / (name + ".csv"))
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(frame[x].astype(str), frame[y], marker="o")
        axis.set_title(name)
        axis.set_ylabel(y)
        axis.tick_params(axis="x", rotation=60)
        figure.tight_layout()
        figure.savefig(output / (name + ".png"), dpi=150)
        plt.close(figure)
