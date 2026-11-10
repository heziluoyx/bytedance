# 用户行为数据清洗、SQLite 分析与特征预评估

## 1.1.0 完整项目交付

现已封装清洗、特征、训练、批量概率推理与模型包导出，完整90项测试通过。

- [完整项目总报告](reports/抖音商城用户行为分析项目总报告.docx)：42页，含全部分析主题、历史与最新模型实验、融合解释及模拟业务结果。
- [项目汇报PPT](presentations/抖音商城用户行为分析项目汇报.pptx)与[技术分享PPT](presentations/抖音商城购买概率建模技术分享.pptx)：分别18页和19页，配逐页讲解提纲。
- [API文档与运行示例](docs/api_reference.md)、[特征拆分Excel](docs/特征拆分整理.xlsx)、[项目复盘](docs/project_retrospective.md)。
- [全部报告索引](reports/README.md)、[聚合结果与调参清单](reports/results/result_manifest.json)、[版本发布记录](releases/v1.1.0.md)。

安装稳定入口：`python -m pip install -e ./behavior_cleaning`（仓库根目录），建模环境见API文档；随后运行`behavior-toolkit --help`。预测使用真实0/1标签训练，主要输出连续概率。现有实验窗口固定，双十二影响、长尾局限与未实施真实A/B均在报告中说明。

当前默认使用新版清洗：保留正常重复，只压缩统计异常小时内的重复冗余。四元组不再要求全局唯一。
旧版全量四元组去重可通过 `--repeat-policy legacy` 显式复现。

## 运行

在仓库根目录中逐条执行，命令末尾不添加反斜杠：

```powershell
python -m pip install -r behavior_cleaning/requirements.txt
python behavior_cleaning/run_cleaning.py --input "D:\字节\user_behavior.csv\user_behavior.csv" --repeat-policy burst
python behavior_cleaning/scripts/build_database.py
```

请把 --input 替换成自己的数据路径。建库脚本默认查找本地最近一次带 SUCCESS 的清洗结果。
新版数据库默认保存在 `outputs/sql_analysis_v2/user_behavior.sqlite`，已有文件禁止覆盖。
SQLite 排序临时目录放在项目 `outputs/sqlite_temp`，只影响当前进程，避免占满系统临时盘。

## 当前成果与说明

- [重复规则、诊断依据与新旧数据对比](docs/repeat_policy_v2.md)
- [新版 SQLite 第 1—5 步与全量验收](docs/sql_setup_v2.md)
- [旧版 SQLite 建库说明](docs/sql_setup.md)
- 正常重复保留；每小时 100 次不是硬编码删除阈值。
- 本次数据自适应候选阈值为大于 206 次/用户小时，仍需个人基线或重复强度证据。
- 全量清洗隔离 52,576 条异常重复冗余，保留 12,204,330 条记录。
- 新版 SQLite 保留全部 12,204,330 条记录，完成四表、六索引及完整性核验。
- 当前完整测试 90 项通过，导入排序及 Black 格式检查通过。
- 已补充基础业务 SQL 分析、历史特征及中国大陆日历分析。
- 原始 CSV、Parquet、数据库、清洗输出和本地依赖不在仓库中。

## SQLite 分析、特征与中国大陆日历

以下命令在仓库根目录运行：

```powershell
python -m pip install -r behavior_cleaning/requirements-analysis.txt
python behavior_cleaning/scripts/analyze_features.py
python behavior_cleaning/scripts/analyze_features.py --advanced --features-only
python behavior_cleaning/scripts/analyze_china_calendar.py
```

若要一次运行描述性分析和历史特征，可使用 `python behavior_cleaning/scripts/analyze_features.py --advanced`。
两个入口默认读取新版数据库，只读访问源数据，分别创建新的分析输出目录。

| 内容 | 文件 |
| --- | --- |
| SQLite 业务查询 | [sql/03_analysis.sql](sql/03_analysis.sql) |
| 分析与历史特征入口 | [scripts/analyze_features.py](scripts/analyze_features.py) |
| 中国大陆官方日历分析 | [scripts/analyze_china_calendar.py](scripts/analyze_china_calendar.py) |
| SQL 指标与预测口径 | [docs/feature_analysis.md](docs/feature_analysis.md) |
| 日历来源与无假期样本说明 | [docs/china_calendar_analysis.md](docs/china_calendar_analysis.md) |
| 订单与退款字段要求 | [docs/order_data_requirements.md](docs/order_data_requirements.md) |

当前输出包括行为分布、DAU/滚动 WAU、严格顺序漏斗、用户/商品/类目分析、可识别小时桶序列，以及按历史时点构建的用户特征。预评估对 61 个数值输入进行筛选，保留 30 个；这属于单次时间验证，不能作为最终模型结论。

SVD/共现规则默认仅覆盖训练热门 1,000 个商品，金额缺失时仅计算 RF 代理；商品全期生命周期特征尚未接入历史预测。2025-11-18 至 12-18 没有全国统一放假调休日，假期差异不可估计。正式预测实验还需历史内拟合清洗阈值。

安装分析依赖后，在 `behavior_cleaning` 目录运行测试：

```powershell
python -m pip install pytest
python -B -m pytest tests -q --assert=plain
```

## 历史版本说明

下面保留原有旧版规则和交付说明，仅适用于 legacy 模式；不作为新版的删除规则。

### 旧版清洗规则

| 原始字段 | 标准字段 | 校验和存储方式 |
| --- | --- | --- |
| `user_id` | `user_id` | 正整数字符串，保留前导零 |
| `item_id` | `item_id` | 正整数字符串，保留前导零 |
| `item_category` | `category_id` | 正整数字符串，保留前导零 |
| `behavior_type` | `behavior_type` | 仅允许 1、2、3、4，输出 int8 |
| `time` | `timestamp` | 严格 YYYY-MM-DD HH，输出 Parquet timestamp |

也接受已经规范化的标准字段名。出现重名、别名与标准名同时出现、缺少必要字段时立即报错。
其他字段从清洗明细中剔除，字段名写入报告。原始文件不修改。

1. 去除字段首尾空白，将空白、null、none、nan、na、n/a、`<na>` 识别为缺失，
   不区分大小写。必要字段缺失的行单独隔离，不填造 ID 或行为类型。
2. 检查正整数 ID、行为范围、严格时间格式和真实日期，及可选的起止范围。
   不转换时区，也不把小时数据伪造成更精细的行为时间。负数、小数形式 ID 不接受。
3. 对有效记录按 `user_id + item_id + behavior_type + timestamp` 全局去重。
   SQLite 磁盘唯一主键覆盖跨块和跨文件重复，不使用内存无限增长的 Python set。
   同小时真实重复浏览也会被此任务规定的四元组合并，需据此解释清洗后 PV。
4. 去重时稳定保留第一条。若同一四元组类目不同，在重复审计表中保留原值、
   被保留的类目和来源。对全量有效记录中的商品多类目情况输出冲突表，
   不擅自取众数修正，不删除冲突商品。报告的一致性检查可因此为 false。
5. 在全局去重后的用户日行为量上计算 IQR 上界 `Q3 + 1.5 * (Q3 - Q1)`
   及 3σ 上界 `mean + 3 * population_std`。从精确频数分布计算分位数，
   不使用分块阈值或抽样阈值；分位数采用线性插值。
   默认仅标记高活跃用户日，并把标记关联回行为明细，不自动删掉潜在真实行为。
   ID 和离散行为编码不适合套用连续变量异常值法。
6. Parquet 使用 Zstandard 压缩及字典编码；行为编码 int8，行为名称使用类别型。
   高基数用户和商品 ID 保持字符串，避免整数溢出、精度丢失和前导零丢失。

## 输出与审计

每次创建独立的 `outputs/run_日期时间_随机后缀/`，不会覆盖以前结果。
只有包含 `SUCCESS` 的目录才代表执行成功。失败时保留已有审计和日志供排查。

| 文件 | 内容 |
| --- | --- |
| `cleaned_events.parquet` | 标准五字段、来源文件、数据记录号、行为名称、两个异常标记 |
| `user_summary.parquet` | 用户行为量、四类行为量、商品数、活跃天数、首次/末次时间 |
| `item_summary.parquet` | 商品行为量、用户数、类目数、四类行为量 |
| `time_summary.parquet` | 小时行为量、独立用户数、四类行为量 |
| `user_daily_flags.parquet` | 用户日行为量和 IQR、3σ 标记 |
| `item_category_conflicts.parquet` | 存在多个类目的商品及类目数，包含去重丢弃行的类目信息 |
| `duplicates.parquet` | 被丢弃的有效重复行及保留行的来源、类目 |
| `rejected.jsonl` | 规则拒绝的原始记录、全部拒绝原因、源文件和数据记录号 |
| `quality_report.json` | SHA-256、配置、行数、缺失、拒绝、重复、时间、类目冲突、异常阈值 |
| `audit.sqlite` | 全局去重、冲突追踪和聚合使用的磁盘审计数据库 |
| `cleaning.log` | 进度及失败日志 |

`source_row` 是原始文件表头之后从 1 开始的数据记录编号，不是包含表头的物理文本行号。
拒绝原因统计可能一行对应多个原因，各原因之和不等于拒绝行数。
用户和商品 ID 均以不透明字符串处理；商品跨类目可能是脏数据，也可能是业务变化，
需由数据提供方确认。行为计数是清洗后事件数，不是订单数或销量。

自动核对：`输入行数 = 规则拒绝行数 + 重复行数 + 清洗后行数`，
导出行数等于清洗后行数，用户聚合行为总量等于清洗后行数。
唯一性由数据库主键约束保障；必要字段有效性由写入前校验保障。
CSV 语法损坏时停止，不使用 `on_bad_lines="skip"` 静默丢行。


## 测试及代码检查

```powershell
python -m pytest
python -m isort --check-only src tests
python -m black --check src tests
python -m mypy src
```

测试覆盖时间格式、非法行为、缺失 ID、别名冲突、端点时间、跨块和跨文件去重、
类目冲突追踪、拒绝行核对以及异常只标记不删除。测试使用临时目录，原始 CSV 不受影响。

本目录包含完整代码和测试。后续特征编码、归一化、未来七天标签和模型训练不在清洗阶段执行，
避免在全量清洗过程中引入未来信息。源代码和运行说明纳入 Git 版本管理。


## 最新报告与多路径复核

最新三份 Word 报告见 [reports/README.md](reports/README.md)，包含按批注修订的特征有效性、探索性分析及核心洞察。

```powershell
python behavior_cleaning/scripts/analyze_purchase_paths.py
```

上面的命令在仓库根目录运行。脚本只读访问数据库，输出新的JSON、CSV和SUCCESS。SQL见 `sql/04_purchase_paths.sql`，详细口径与已核验结果见 [购买路径复核](docs/purchase_path_analysis.md)。完整7天购买比例为浏览1.07%、加购8.20%、收藏3.30%，各起点人群可重叠，不能相加或解释为因果提升。测试新增小时顺序、168小时边界及输出保护的3项验证。

## 用户商品购买模型训练

新增用户—商品对未来7天标签、分层7:2:1样本、不平衡方案比较、Optuna调优以及LSTM、GRU、DIN、多头注意力扩展。

运行入口：`python behavior_cleaning/scripts/train_purchase_models.py`（仓库根目录）。

详细安装、分阶段运行和适用边界见 [模型训练说明](docs/model_training.md)。训练结果保存在本地 `behavior_cleaning/outputs/purchase_models_...`，不上传用户样本和预测明细。

## 模型融合与业务模拟

运行 `python behavior_cleaning/scripts/analyze_stacking_business.py`（仓库根目录），执行三折OOF融合、逻辑回归系数、XGBoost与DIN的SHAP解释、错误诊断和用户级A/B模拟。详见 [融合与业务模拟说明](docs/stacking_business.md)。本轮融合未超过树模型测试AUC；业务金额为假设情景，并非真实实验收益。项目现有60项测试通过。

## 双十二窗口复核

新增 `python behavior_cleaning/scripts/review_promotion_models.py`，在11月25日较早截点重新调优三种传统模型，冻结后评估12月12日跨期压力样本。两期特征统一固定7天；较早7:2:1留出与后期压力评估分别报告，不把促销期成绩当作常态结论。详见 [重训设计与限制](docs/promotion_review.md)。

## 严格时间窗口的连续购买概率

最新入口为 python behavior_cleaning/scripts/train_temporal_probability.py（仓库根目录）。本轮对七种单模型和时间留出Stacking重新训练，输出用户商品对未来7天购买的0到1连续概率，统一3天历史，按四个不重叠7天标签窗口隔离训练、调参、融合校准和最终测试。说明见 [时间概率预测](docs/temporal_probability.md)。

默认结果为 outputs/temporal_probability_20261109，包含模型、概率校准器、连续概率CSV/Parquet、三份HTML/Markdown报告及解释图表。原实验保留。自然分布漂移、促销影响和源清洗回看限制在新报告中单独说明，不把购买倾向当营销增量。

四份最新Word模型报告已归档在 [reports](reports/README.md)。补充诊断入口为 `python behavior_cleaning/scripts/analyze_probability_diagnostics.py`（仓库根目录），生成用户内排序、十分位Lift、PSI、融合与消融、SHAP依赖及精确交互等结果，见 [诊断说明](docs/probability_diagnostics.md)。
