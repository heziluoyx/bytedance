"""只读复核用户商品购买路径，单独处理小时顺序和完整七天观察期。"""

import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def run_purchase_path_analysis(
    database: Path, output: Path, sql_path: Path
) -> dict[str, Any]:
    """执行路径SQL并导出汇总；不覆盖输出，也不修改源数据库。"""
    database = database.resolve(strict=True)
    query = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-131072")
        connection.row_factory = sqlite3.Row
        start, end = connection.execute(
            "SELECT MIN(event_time),MAX(event_time) FROM dim_time"
        ).fetchone()
        if start is None or end is None:
            raise ValueError("数据库时间维度为空，无法确定完整观察期")
        cutoff = (
            datetime.strptime(end, "%Y-%m-%d %H:%M:%S") - timedelta(days=7)
        ).strftime("%Y-%m-%d %H:%M:%S")
        raw = dict(connection.execute(query, {"cutoff": cutoff}).fetchone())
    counts = {key: int(value or 0) for key, value in raw.items()}
    if not counts["all_interaction_pairs"]:
        raise ValueError("行为主表为空，无法进行路径分析")
    groups = [
        "cart_only_before_first_buy",
        "fav_only_before_first_buy",
        "both_before_first_buy",
        "neither_earlier_before_first_buy",
    ]
    if sum(counts[key] for key in groups) != counts["purchased_pairs"]:
        raise ValueError("首次购买前意向分组核对失败")
    paths = []
    for action in ("pv", "cart", "fav"):
        total = counts[action + "_pairs"]
        later = counts[action + "_later_buy_pairs"]
        eligible = counts[action + "_complete7d_pairs"]
        bought = counts[action + "_complete7d_buy_pairs"]
        if not (0 <= bought <= eligible <= total and bought <= later <= total):
            raise ValueError("路径计数核对失败：" + action)
        paths.append(
            {
                "start_action": action,
                "pairs": total,
                "later_buy_pairs": later,
                "window_rate": later / total if total else None,
                "complete_7d_pairs": eligible,
                "complete_7d_buy_pairs": bought,
                "complete_7d_rate": bought / eligible if eligible else None,
            }
        )
    result = {
        "start": start,
        "end": end,
        "complete7d_anchor_cutoff": cutoff,
        "unit": "distinct user-item pair",
        "definition": (
            "Each first observed action anchors its own window; purchase in "
            + "(anchor, anchor+168h]; same-hour order is unknown."
        ),
        "paths_overlap": True,
        "all_period_and_first_purchase": counts,
        "paths": paths,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "purchase_path_review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "purchase_path_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paths[0]))
        writer.writeheader()
        writer.writerows(paths)
    (output / "SUCCESS").write_text("ok\n", encoding="utf-8")
    return result
