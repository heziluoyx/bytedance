"""免安装入口：SQLite 分析及可选的历史快照特征预评估。"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """设置临时目录后再导入 SQLite 相关依赖。"""
    root = Path(__file__).absolute().parents[1]
    temporary = root / "outputs" / "sqlite_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for variable in ("TMP", "TEMP", "SQLITE_TMPDIR"):
        os.environ[variable] = str(temporary)
    sys.path.insert(0, str(root / ".runtime"))
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description="SQLite 业务分析与特征有效性预评估")
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "outputs/sql_analysis_v2/user_behavior.sqlite",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calendar", type=Path, help="date,is_holiday,region CSV")
    parser.add_argument("--advanced", action="store_true", help="追加历史快照特征预评估")
    parser.add_argument("--features-only", action="store_true", help="跳过全期EDA报表")
    parser.add_argument("--max-items", type=int, default=1000)
    arguments = parser.parse_args()
    if arguments.max_items < 2 or arguments.max_items > 5000:
        parser.error("资源可控原型的 --max-items 范围为2至5000")
    if arguments.features_only and not arguments.advanced:
        parser.error("--features-only 必须同时使用 --advanced")
    output = arguments.output or root / "outputs" / (
        "analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from behavior_cleaning.analysis_runner import analyze

    if arguments.features_only:
        output.mkdir(parents=True, exist_ok=False)
    else:
        analyze(arguments.database, output, arguments.calendar)
    if arguments.advanced:
        from behavior_cleaning.feature_validation import run_validation

        result = run_validation(arguments.database, output, arguments.max_items)
        (output / "feature_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "SUCCESS").write_text("completed requested scope\n", encoding="utf-8")
    print("输出目录：" + str(output))


if __name__ == "__main__":
    main()
