"""免安装项目包的启动入口：python run_cleaning.py。

将项目 src 目录加入本次进程的导入路径，不修改系统环境变量。
若存在项目本地依赖 .runtime，也将其加入导入路径。
"""

import sys
from pathlib import Path


def main() -> None:
    """加载项目源码和可选的本地依赖，调用清洗命令行入口。"""
    project_root = Path(__file__).absolute().parent
    runtime = project_root / ".runtime"
    if runtime.is_dir():
        sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(project_root / "src"))

    from behavior_cleaning.__main__ import main as cleaning_main

    cleaning_main()


if __name__ == "__main__":
    main()
