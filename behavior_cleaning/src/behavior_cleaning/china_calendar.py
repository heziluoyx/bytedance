"""中国大陆2025年全国统一放假调休日历及只读行为分析。"""

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from behavior_cleaning.analysis_runner import connect_readonly

SOURCE_URL = "https://www.forestry.gov.cn/c/www/szxx/594663.jhtml"
ORIGINAL_URL = "https://www.gov.cn/zhengce/content/202411/content_6986382.htm"
HOLIDAY_PERIODS = [
    ("2025-01-01", "2025-01-01", "元旦"),
    ("2025-01-28", "2025-02-04", "春节"),
    ("2025-04-04", "2025-04-06", "清明节"),
    ("2025-05-01", "2025-05-05", "劳动节"),
    ("2025-05-31", "2025-06-02", "端午节"),
    ("2025-10-01", "2025-10-08", "国庆节及中秋节"),
]
MAKEUP_WORKDAYS = {"2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11"}


def calendar_2025() -> pd.DataFrame:
    """生成全年日历，放假区间包含通知安排的调休和连休日期。

    Returns:
        每日唯一的日历；is_holiday不表示该区间每一天均为法定节日本日。
    """
    holidays = {}
    for start, finish, name in HOLIDAY_PERIODS:
        current = date.fromisoformat(start)
        while current <= date.fromisoformat(finish):
            holidays[current.isoformat()] = name
            current += timedelta(days=1)
    records = []
    current = date(2025, 1, 1)
    while current.year == 2025:
        day = current.isoformat()
        holiday = int(day in holidays)
        weekend = int(current.weekday() >= 5)
        makeup = int(day in MAKEUP_WORKDAYS)
        workday = int(makeup or (not weekend and not holiday))
        records.append(
            {
                "date": day,
                "region": "CN-mainland",
                "is_holiday": holiday,
                "holiday_name": holidays.get(day, ""),
                "is_weekend": weekend,
                "is_makeup_workday": makeup,
                "is_workday": workday,
                "is_rest_day": 1 - workday,
                "day_type": "holiday_break"
                if holiday
                else "makeup_workday"
                if makeup
                else "ordinary_weekend"
                if weekend
                else "ordinary_workday",
                "source_url": SOURCE_URL,
            }
        )
        current += timedelta(days=1)
    return pd.DataFrame(records)


def comparison(daily: pd.DataFrame) -> pd.DataFrame:
    """补齐假期/非假期两组；缺样本时保留空值而不是补零指标。"""
    groups = (
        daily.groupby("is_holiday")
        .agg(
            observed_days=("date", "size"),
            events_per_day=("events", "mean"),
            users_per_day=("users", "mean"),
            buys_per_day=("buys", "mean"),
            daily_buyer_share=("buyer_share", "mean"),
        )
        .reindex([0, 1])
    )
    groups["observed_days"] = groups["observed_days"].fillna(0).astype(int)
    groups["sample_available"] = groups["observed_days"] > 0
    return groups.reset_index()


def run_calendar_analysis(database: Path, output: Path) -> dict:
    """导入全年日历到独立 SQLite，将事实表按天聚合后重新比较。

    Args:
        database: 新清洗数据对应的源数据库，只读访问。
        output: 本次新建输出目录，不覆盖旧报告。

    Returns:
        包含数据范围、分组天数及是否可比较的审计摘要。
    """
    output.mkdir(parents=True, exist_ok=False)
    calendar = calendar_2025()
    calendar.to_csv(
        output / "china_mainland_calendar_2025.csv", index=False, encoding="utf-8-sig"
    )
    with connect_readonly(database) as connection:
        daily = pd.read_sql_query(
            "SELECT date(event_time) date,COUNT(*) events,"
            + "COUNT(DISTINCT user_id) users,SUM(behavior_type=4) buys,"
            + "COUNT(DISTINCT CASE WHEN behavior_type=4 THEN user_id END) buyers "
            + "FROM user_behavior GROUP BY date(event_time) ORDER BY date",
            connection,
        )
    if daily.empty:
        raise ValueError("源数据没有可分析日期")
    if not daily["date"].str.startswith("2025-").all():
        raise ValueError("此日历仅核验2025年，其他年份须补充对应年度官方日历")
    with sqlite3.connect(output / "holiday_analysis.sqlite") as results:
        calendar.to_sql("dim_calendar", results, index=False, if_exists="fail")
        results.execute("CREATE UNIQUE INDEX idx_calendar_date ON dim_calendar(date)")
        daily.to_sql("daily_behavior", results, index=False, if_exists="fail")
        joined = pd.read_sql_query(
            "SELECT d.*,c.region,c.is_holiday,c.holiday_name,c.is_weekend,"
            + "c.is_makeup_workday,c.is_workday,c.is_rest_day,c.day_type,"
            + "1.0*d.buyers/d.users buyer_share FROM daily_behavior d "
            + "JOIN dim_calendar c USING(date) ORDER BY date",
            results,
        )
        if len(joined) != len(daily):
            raise ValueError("日历未完整匹配观测日期")
        groups = comparison(joined)
        workdays = (
            joined.groupby("is_workday")
            .agg(
                observed_days=("date", "size"),
                events_per_day=("events", "mean"),
                users_per_day=("users", "mean"),
                buys_per_day=("buys", "mean"),
                mean_daily_buyer_share=("buyer_share", "mean"),
            )
            .reset_index()
        )
        for name, frame in [
            ("calendar_daily_analysis", joined),
            ("holiday_comparison", groups),
            ("workday_restday_comparison", workdays),
        ]:
            frame.to_csv(output / (name + ".csv"), index=False, encoding="utf-8-sig")
            frame.to_sql(name, results, index=False, if_exists="fail")
    holiday_days = int(joined["is_holiday"].sum())
    comparable = bool(groups["sample_available"].all())
    summary = {
        "source_database": str(database),
        "calendar_region": "CN-mainland",
        "calendar_year": 2025,
        "calendar_days": len(calendar),
        "source_url": SOURCE_URL,
        "original_url": ORIGINAL_URL,
        "document": "国办发明电〔2024〕12号",
        "definition": "国务院全国统一放假调休区间（含连休），不将普通周末视为节假日",
        "date_start": joined["date"].min(),
        "date_end": joined["date"].max(),
        "observed_days": len(joined),
        "holiday_days": holiday_days,
        "nonholiday_days": len(joined) - holiday_days,
        "makeup_workdays": int(joined["is_makeup_workday"].sum()),
        "workdays": int(joined["is_workday"].sum()),
        "ordinary_weekend_days": int((joined["day_type"] == "ordinary_weekend").sum()),
        "holiday_comparison_estimable": comparable,
        "holiday_effect": None,
        "status": "completed" if comparable else "completed_no_holiday_observations",
        "limitations": [
            "使用记录中的本地日期，按中国大陆全国统一安排解释，未转换原始时区。",
            "不覆盖港澳台或少数民族地方性假期，不推断个体实际是否休息。",
            "无节假日样本时差异不可估计，不表示节假日没有影响。",
            "购买为行为事件数，没有订单信息，不能称为订单数或销售额。",
        ],
    }
    (output / "calendar_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    note = (
        "# 中国大陆日历重分析\n\n"
        + f"观测期：{summary['date_start']}至{summary['date_end']}，共{len(joined)}天。\n\n"
        + f"放假调休日{holiday_days}天，非假期{len(joined)-holiday_days}天，"
        + f"调休补班{summary['makeup_workdays']}天；"
        + f"工作日{summary['workdays']}天，普通周末{summary['ordinary_weekend_days']}天。\n\n"
        + (
            "缺少假期组，无法估计假期相对非假期的转化差异或提升率。"
            if not comparable
            else "两组均有样本，可进行描述性对比，不能据此断言因果。"
        )
        + "12月12日不属于法定节假日放假区间，促销活动需另加业务活动日历。\n\n"
        + "假期指标空值表示不可估计，不是0；工作日/休息日对比按日均值计算。"
        + "如需假期对比，应补充至少覆盖某一完整假期及前后普通日期的行为数据，"
        + "理想情况下包含多个假期和更长基线；例如补充2025年国庆中秋假期及前后日期。\n\n"
        + f"来源：[国务院2025年放假安排（官方转载）]({SOURCE_URL})。\n"
    )
    (output / "分析结论.md").write_text(note, encoding="utf-8")
    (output / "SUCCESS").write_text(summary["status"] + "\n", encoding="utf-8")
    return summary
