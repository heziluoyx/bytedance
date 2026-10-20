```powershell
python -m pip install -r behavior_cleaning/requirements.txt
python behavior_cleaning/run_cleaning.py --input "D:\字节\user_behavior.csv\user_behavior.csv"
```

将 `--input` 替换为自己的 CSV 文件或目录路径。输入文件需包含 `time,user_id,item_id,item_category,behavior_type`，也支持对应的标准字段名。

默认输出到 `behavior_cleaning/outputs/run_.../`。目录包含 `SUCCESS` 文件代表执行成功。

- [运行入口](behavior_cleaning/run_cleaning.py)
- [标准化与业务校验](behavior_cleaning/src/behavior_cleaning/cleaning.py)
- [清洗流程与质量报告](behavior_cleaning/src/behavior_cleaning/pipeline.py)
- [全局去重、聚合和 Parquet 导出](behavior_cleaning/src/behavior_cleaning/storage.py)
- [详细规则和输出说明](behavior_cleaning/README.md)
- [测试](behavior_cleaning/tests/test_cleaning.py)

原始数据、清洗结果和本地依赖不包含在仓库中。IQR 和 3σ 异常默认仅标记，不自动删除真实高活跃行为。


现有代码已通过 13 项测试，并对真实 CSV 前 100,000 行完成全流程验证：移除 7,458 条四元组重复记录，保留 92,542 行。此结果属于样本验证，不代表全量清洗统计。
