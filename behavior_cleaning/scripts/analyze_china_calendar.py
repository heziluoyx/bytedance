"""导入2025年中国大陆日历并只重跑日期分析。"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """设置项目临时目录，读取新版 SQLite，避免重跑无关特征。"""
    root = Path(__file__).absolute().parents[1]
    temporary = root / "outputs/sqlite_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for variable in ("TMP", "TEMP", "SQLITE_TMPDIR"):
        os.environ[variable] = str(temporary)
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description="中国大陆2025日历重分析")
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "outputs/sql_analysis_v2/user_behavior.sqlite",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    output = arguments.output or root / "outputs" / (
        "china_calendar_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    from behavior_cleaning.china_calendar import run_calendar_analysis

    print("正在按日聚合新版 SQLite，并导入2025年中国大陆日历……", flush=True)
    summary = run_calendar_analysis(arguments.database, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("输出目录：" + str(output))


if __name__ == "__main__":
    main()
