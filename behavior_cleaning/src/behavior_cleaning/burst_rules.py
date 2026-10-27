"""根据全量分布与个人基线识别异常小时，仅压缩异常小时的重复冗余。"""

import math
from dataclasses import asdict, dataclass

import duckdb


@dataclass(frozen=True)
class BurstRules:
    """数据自适应异常识别参数，不使用固定的每小时 100 次阈值。

    Attributes:
        hour_quantile: 小时操作量的全局尾部分位数。
        repeat_quantile: 同小时同商品同行为最大重复量的尾部分位数。
        min_baseline_hours: 使用个人基线所需的最少活跃小时数。
        personal_multiplier: 相对个人中位数的倍数下界。
        mad_multiplier: 个人稳健标准差的倍数。
    """

    hour_quantile: float = 0.999
    repeat_quantile: float = 0.999
    min_baseline_hours: int = 12
    personal_multiplier: float = 3.0
    mad_multiplier: float = 6.0

    def __post_init__(self) -> None:
        """拒绝不合理的分位数和个人基线参数。

        Raises:
            ValueError: 参数不在允许范围。
        """
        if not 0.5 <= self.hour_quantile < 1:
            raise ValueError("hour_quantile 必须在 [0.5, 1) 内")
        if not 0.5 <= self.repeat_quantile < 1:
            raise ValueError("repeat_quantile 必须在 [0.5, 1) 内")
        if self.min_baseline_hours < 1:
            raise ValueError("min_baseline_hours 必须大于零")
        if self.personal_multiplier <= 1 or self.mad_multiplier <= 0:
            raise ValueError("个人基线倍数不合理")


def assess_hours(
    connection: duckdb.DuckDBPyConnection,
    rules: BurstRules,
) -> dict:
    """生成 assessed_hours 表和可重现的阈值定义。

    Args:
        connection: 已含 hourly_profile 表的连接。
        rules: 参数配置。

    Returns:
        根据分位数和远端 IQR 得到的操作量、重复量阈值及参数。
    """
    statistics = connection.execute(
        "SELECT quantile_cont(event_count, ?), quantile_cont(event_count,.25), "
        + "quantile_cont(event_count,.75), quantile_cont(max_same_operation, ?), "
        + "quantile_cont(max_same_operation,.25), quantile_cont(max_same_operation,.75) "
        + "FROM hourly_profile",
        [rules.hour_quantile, rules.repeat_quantile],
    ).fetchone()
    hour_limit = math.ceil(
        max(statistics[0], statistics[2] + 3 * (statistics[2] - statistics[1]))
    )
    repeat_limit = math.ceil(
        max(statistics[3], statistics[5] + 3 * (statistics[5] - statistics[4]))
    )
    connection.execute(
        "CREATE TABLE user_hour_baselines AS SELECT user_id, COUNT(*) AS active_hours, "
        + "MEDIAN(event_count) AS median_count, MAD(event_count) AS mad_count "
        + "FROM hourly_profile GROUP BY user_id"
    )
    connection.execute(
        "CREATE TABLE assessed_hours AS WITH joined AS ("
        + "SELECT h.*, b.active_hours,b.median_count,b.mad_count, "
        + "greatest(? * b.median_count, b.median_count + ? * 1.4826 * b.mad_count) "
        + "AS personal_limit FROM hourly_profile h JOIN user_hour_baselines b USING(user_id)"
        + "), flagged AS (SELECT *, event_count > ? AS is_high_volume_candidate, "
        + "(active_hours >= ? AND event_count > personal_limit) AS is_personal_burst, "
        + "max_same_operation > ? AS is_repetition_extreme FROM joined) "
        + "SELECT *, (is_high_volume_candidate AND "
        + "(is_personal_burst OR is_repetition_extreme)) AS is_burst_anomaly FROM flagged",
        [
            rules.personal_multiplier,
            rules.mad_multiplier,
            hour_limit,
            rules.min_baseline_hours,
            repeat_limit,
        ],
    )
    return {
        "parameters": asdict(rules),
        "hour_limit": hour_limit,
        "repeat_limit": repeat_limit,
        "global_formula": "ceil(max(Pq, Q3 + 3 * IQR))",
        "personal_formula": "max(3 * median, median + 6 * 1.4826 * MAD) with default parameters",
        "decision": "hour_count > global_limit AND (personal_burst OR repetition_extreme)",
        "action": "only_remove_redundancies_in_flagged_hours_keep_one_per_quadruple",
        "time_resolution": "calendar_hour_bins_not_rolling_60_minutes",
        "certainty": "statistical_candidate_not_verified_bot_or_fraud",
    }


def retain_regular_repetitions(connection: duckdb.DuckDBPyConnection) -> None:
    """保留正常重复，只隔离异常小时内每组第一条以外的冗余记录。

    Args:
        connection: 已含 events 和 assessed_hours 的连接。
    """
    connection.execute(
        "CREATE TABLE ranked_burst_events AS SELECT e.*, "
        + "ROW_NUMBER() OVER(PARTITION BY e.user_id,e.item_id,e.behavior_type,e.timestamp "
        + "ORDER BY e.source_file,e.source_row) AS repeat_ordinal "
        + "FROM events e JOIN assessed_hours h USING(user_id,timestamp) "
        + "WHERE h.is_burst_anomaly"
    )
    connection.execute(
        "CREATE TABLE removed_anomalies AS SELECT *, "
        + "'redundancy_in_statistically_extreme_user_hour' AS removal_reason "
        + "FROM ranked_burst_events WHERE repeat_ordinal > 1"
    )
    connection.execute(
        "CREATE TABLE retained_events AS SELECT e.*, h.event_count AS source_hour_event_count, "
        + "h.is_high_volume_candidate,h.is_burst_anomaly FROM events e "
        + "JOIN assessed_hours h USING(user_id,timestamp) "
        + "WHERE NOT EXISTS (SELECT 1 FROM removed_anomalies r "
        + "WHERE r.source_file=e.source_file AND r.source_row=e.source_row)"
    )
