# bytedance

抖音商城用户行为数据清洗与 SQLite 数据工程项目。主要代码在 [behavior_cleaning](behavior_cleaning/)。

## 最新更新

- 清洗默认保留正常重复，只隔离统计异常小时内的重复冗余。
- 按全量分布、个人基线和同小时重复强度识别异常，100 次仅为对照参考。
- 新增重复频数、跨小时重复间隔、阈值敏感性和隔离明细输出。
- 新增 SQLite 规范化四表、批量导入、六个索引及全量核验。
- 新版四元组使用普通索引，以 event_id 保留正常重复。
- 将 SQLite 排序临时目录放到项目数据盘，支持完整导入后的安全续建。

## SQLite 分析与特征预评估

建库后可执行基础 SQL 分析、历史快照特征和未来 7 天购买预评估：

```powershell
python -m pip install -r behavior_cleaning/requirements-analysis.txt
python behavior_cleaning/scripts/analyze_features.py
python behavior_cleaning/scripts/analyze_features.py --advanced --features-only
python behavior_cleaning/scripts/analyze_china_calendar.py
```

- [SQL 查询合集](behavior_cleaning/sql/03_analysis.sql)：行为分布、日活/滚动周活、严格路径漏斗、生命周期、类目及小时序列。
- [特征与分析口径](behavior_cleaning/docs/feature_analysis.md)：训练/验证时点、预处理、筛选和结果文件。
- [中国大陆日历分析](behavior_cleaning/docs/china_calendar_analysis.md)：包含官方来源及调休；当前观察期无假期，不能估计假期差异。
- [订单数据字段要求](behavior_cleaning/docs/order_data_requirements.md)：真实 RFM、订单与商品复购所需信息。

输出自动保存到本地 `behavior_cleaning/outputs/`。SVD 和关联规则默认仅训练热门 1,000 个商品；无金额字段只计算 RF 代理。时间仅有小时精度，不能还原同小时内的行为先后。

## 运行

需要 Python 3.10+，在仓库目录逐条执行：

```powershell
python -m pip install -r behavior_cleaning/requirements.txt
python behavior_cleaning/run_cleaning.py --input "D:\字节\user_behavior.csv\user_behavior.csv"
python behavior_cleaning/scripts/build_database.py
```

将 --input 替换为自己的 CSV 文件或目录路径。命令末尾不要添加反斜杠。
清洗输出保存在 behavior_cleaning/outputs/run_...，有 SUCCESS 才代表成功。
新版数据库默认保存在 behavior_cleaning/outputs/sql_analysis_v2/user_behavior.sqlite。
已有结果不会覆盖；重新建库时可通过 --database 指定新文件。

## 主要文件

| 内容 | 入口 |
| --- | --- |
| 清洗运行入口 | [run_cleaning.py](behavior_cleaning/run_cleaning.py) |
| 新版清洗流程 | [burst_pipeline.py](behavior_cleaning/src/behavior_cleaning/burst_pipeline.py) |
| 频数与时间差诊断 | [burst_profile.py](behavior_cleaning/src/behavior_cleaning/burst_profile.py) |
| 数据自适应异常规则 | [burst_rules.py](behavior_cleaning/src/behavior_cleaning/burst_rules.py) |
| SQLite 建库入口 | [build_database.py](behavior_cleaning/scripts/build_database.py) |
| SQLite 安全续建 | [resume_database.py](behavior_cleaning/scripts/resume_database.py) |
| 规则与新旧结果对比 | [修订说明](behavior_cleaning/docs/repeat_policy_v2.md) |
| SQLite 操作及验收 | [新版建库说明](behavior_cleaning/docs/sql_setup_v2.md) |

## 验证

当前完整测试 47 项通过，isort 与 Black 检查通过。
本地全量输入 12,256,906 行，新版隔离异常冗余 52,576 行，保留 12,204,330 行。
SQLite 已完成同数量导入、四张基础表及六个索引，外键和 integrity_check 均通过。
这些结果属于已有本地数据验证，原始数据和数据库不随代码上传。
新版基础 SQL 分析、历史特征和未来 7 天购买预评估已完成本地运行；完整任务范围及剩余限制见上述分析说明。

原始时间只有小时精度，统计异常不等于已经证实的机器人行为。
小时桶间隔不是秒级时间差，正常重复不会被统一删除。

## 最新分析报告与购买路径复核

- [报告目录](behavior_cleaning/reports/README.md)：已按批注更新的三份 Word 报告。
- [购买路径复核说明](behavior_cleaning/docs/purchase_path_analysis.md)：完整7天观察期、同小时顺序和路径重叠口径。
- [路径复核入口](behavior_cleaning/scripts/analyze_purchase_paths.py)：仅依赖Python标准库，只读查询本地SQLite。

```powershell
python behavior_cleaning/scripts/analyze_purchase_paths.py
```

本地全量复核中，浏览、加购、收藏后的完整7天购买比例分别为1.07%、8.20%、3.30%。这些是行为关联，不是支付订单转化或营销因果提升；汇总依据随报告归档。

## 用户商品购买模型训练

新增用户—商品对未来7天标签、分层7:2:1样本、不平衡方案比较、Optuna调优以及LSTM、GRU、DIN、多头注意力扩展。

运行入口：`python behavior_cleaning/scripts/train_purchase_models.py`（仓库根目录）。

详细安装、分阶段运行和适用边界见 [模型训练说明](behavior_cleaning/docs/model_training.md)。训练结果保存在本地 `behavior_cleaning/outputs/purchase_models_...`，不上传用户样本和预测明细。

## 模型融合与业务模拟

运行 `python behavior_cleaning/scripts/analyze_stacking_business.py`，执行三折OOF融合、模型SHAP解释、错误诊断和用户级A/B模拟。详见 [融合与业务模拟说明](behavior_cleaning/docs/stacking_business.md)。报告包含实际离线结果和单列的假设收益情景，不将预测效果当作因果转化提升。

## 双十二窗口复核

新增较早截点的三种传统模型重训与跨期压力评估，入口为 `behavior_cleaning/scripts/review_promotion_models.py`。详见 [窗口选择与结果限制](behavior_cleaning/docs/promotion_review.md)。项目特征工程实施说明与相关报告已补充活动期适用范围。

## 最新时间窗口购买概率与四份模型报告

最新版本统一3天历史、未来7天购买标签，按时间分开训练、调参、融合校准和最终测试，输出连续购买概率。七种单模型与原时间留出Stacking保留，说明见 [时间窗口预测](behavior_cleaning/docs/temporal_probability.md)。

```powershell
python behavior_cleaning/scripts/train_temporal_probability.py --source "清洗后Parquet路径"
python behavior_cleaning/scripts/analyze_probability_diagnostics.py
```

四份更新的Word报告见 [报告目录](behavior_cleaning/reports/README.md)，覆盖单模型、融合、SHAP解释与模拟A/B。补充诊断计算排序、Lift、PSI、融合对照和消融，详见 [运行说明与评估边界](behavior_cleaning/docs/probability_diagnostics.md)。早期随机划分实验保留作历史版本，不与最新时间窗口结果混为同一评估。
