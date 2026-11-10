"""复核浏览、加购和收藏后的购买行为，仅依赖Python标准库。"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """配置本地SQLite临时目录并执行只读路径分析。"""
    root = Path(__file__).resolve().parents[1]
    temporary = root / "outputs/sqlite_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for variable in ("TMP", "TEMP", "SQLITE_TMPDIR"):
        os.environ[variable] = str(temporary)
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description="多路径及完整7天购买行为复核")
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "outputs/sql_analysis_v2/user_behavior.sqlite",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or root / "outputs" / (
        "purchase_paths_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    from behavior_cleaning.purchase_paths import run_purchase_path_analysis

    result = run_purchase_path_analysis(
        arguments.database, output, root / "sql/04_purchase_paths.sql"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("输出目录：" + str(output))


if __name__ == "__main__":
    main()
