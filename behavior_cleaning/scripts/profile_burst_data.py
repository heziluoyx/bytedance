"""对当前原始 CSV 进行只读业务诊断，保留频数诊断数据库。"""

import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def main() -> None:
    """将原始数据载入独立诊断目录并输出统计摘要。"""
    root = Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(root / ".runtime"))
    sys.path.insert(0, str(root / "src"))
    import duckdb

    from behavior_cleaning.burst_profile import create_profiles

    output = (
        root
        / "outputs"
        / f"burst_review_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    output.mkdir(exist_ok=False)
    connection = duckdb.connect(str(output / "raw_profile.duckdb"))
    try:
        connection.execute("SET memory_limit='1GB'")
        connection.execute("SET threads=2")
        connection.execute(
            "CREATE TABLE events AS SELECT user_id, item_id, "
            + "item_category AS category_id, CAST(behavior_type AS TINYINT) AS behavior_type, "
            + "strptime(time, '%Y-%m-%d %H') AS timestamp "
            + "FROM read_csv(?, all_varchar=true)",
            [str(root.parent / "user_behavior.csv" / "user_behavior.csv")],
        )
        print("Raw loaded", flush=True)
        report = create_profiles(connection)
        (output / "profile.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        connection.execute("CHECKPOINT")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
