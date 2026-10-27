# 新版数据：任务核验与 SQLite 特征分析

数据源为 `outputs/sql_analysis_v2/user_behavior.sqlite`，对应保留正常重复的新清洗结果，不能与旧版 621 万条的数据混用。

## 核验结论

此前已实际完成：数据清洗、正常重复保留和异常操作隔离、Parquet 转换、类型适配、用户/商品/时间基础聚合、SQLite 四表建库与索引、导入核验。

此前的中间表只是基础统计，不能据此认定下面的高级特征和探索分析已经完成。本次新增代码的范围如下；真实数据是否执行完毕，以输出目录 `SUCCESS` 和 JSON 报告为准，代码存在不等于全部计算完成。

| 任务 | 本次实现与结果文件 | 必须保留的边界 |
|---|---|---|
| 用户活跃等级、连续活跃天数、流失间隔 | `user_features.csv`，历史预测快照另有 Parquet | 最长连续活跃天数；流失间隔是最后活跃至观察时点，不是已确认流失 |
| 商品热度衰减、类目竞争 | `item_lifecycle.csv` | 近3日/前4日日均热度比、两窗口衰减代理、类目商品数和 HHI，不是拟合完整生命周期 |
| 相邻行为间隔 | `user_sequence_features.csv` | 不同小时桶间隔，不能还原秒级相邻操作 |
| 行为跳转概率和连续模式 | `transition_probabilities.csv`、用户连续同类型桶模式 | 仅相邻且两侧均为单一行为类型的小时桶可识别；混合桶不臆测顺序 |
| SVD 用户和商品嵌入、稀疏预嵌入 | `item_svd_embeddings.csv`、快照中的 `svd_*` | 训练热门商品子集原型，默认1000个商品，最多5000；不是全量商品覆盖 |
| 用户商品相似度/偏好 | `user_item_similarity_*.csv` | 潜向量余弦、共现加权亲和度、加权偏好占比；仅已覆盖交互对 |
| 商品共现、关联规则 | `item_similarity_association_rules.csv` | 按用户出现的集合共现；至少5个共现用户；输出余弦、支持度、置信度、lift；非因果 |
| RFM 用户价值与四类分群 | `rf_group_comparison.csv`、RF代理分 | 缺少金额M，只完成RF代理；不能写真实RFM已完成 |
| 全链路转化漏斗 | `ordered_funnel.csv`、历史用户快照漏斗特征 | 同用户同商品严格更晚小时的 pv→cart→fav→buy 路径；不能代表所有购买必经路径 |
| 用户偏好强度 | 全量用户权重和，覆盖商品内的用户商品偏好占比 | 浏览1、收藏2、加购3、购买5是可调整业务规则，不是学习所得 |
| 标准化/归一化、缺失值处理 | `*_standardized.parquet`、`*_minmax.parquet` | 仅训练快照拟合中位数及缩放；验证值允许超出训练MinMax范围 |
| 高维类目 Target Encoding | 历史主偏好类目平滑折外编码 | 当前用户级任务只编码主偏好类目，不是对商品ID逐个编码；验证只用训练标签统计 |
| 相关性、方差、互信息筛选 | `feature_importance_preassessment.csv` | MinMax后方差阈值1e-5、绝对相关>0.95剔除、训练正互信息最多30项，阈值是示例 |
| 未来7天购买目标与重要性预评估 | `feature_validation.json`、验证预测 CSV | 一次时间验证；不是正式模型调参或最终效果结论 |
| 行为分布、DAU、WAU、小时分布 | 对应 CSV 和 PNG | WAU为滚动7日；开头6天是不完整窗口；小时买家占比是关联统计 |
| 热门类目/商品、转化周期、复购 | 类目分析及商品生命周期 CSV | 转化周期为首次浏览到其后首次购买；跨日重复购买用户商品对占比是代理，不能算订单复购 |
| 工作日与周末 | `weekday_weekend.csv` | 周一到周五与周六周日；使用日均值避免天数差异，不识别调休 |
| 节假日与非节假日 | 已补充中国大陆2025年全年日历并重分析 | 日历核验已完成，观测期无放假调休日，无法比较两组；见china_calendar_analysis.md |

## 一次运行全部代码

在 Anaconda PowerShell 中逐行运行，不要添加 Linux 风格的反斜线续行：

```powershell
cd D:\字节\behavior_cleaning
python -m pip install -r requirements-analysis.txt
python scripts\analyze_features.py --advanced
```

现有电脑已具备相关科学计算依赖，项目 `.runtime` 中已有 Parquet 依赖；正常情况下可直接运行第三行。脚本以只读方式打开源 SQLite，不修改清洗文件，也不会清除正常重复。

默认创建 `outputs/analysis_年月日_时分秒/`，成功才写入 `SUCCESS`。不要用 Excel 直接打开287万行商品文件：超过单工作表行数上限；使用 Python、SQLite 或分批读取。

只跑探索分析：

```powershell
python scripts\analyze_features.py
```

已有探索报表，只补特征与验证：

```powershell
python scripts\analyze_features.py --advanced --features-only --max-items 1000
```

指定数据库与节假日日历：

```powershell
python scripts\analyze_features.py --database "D:\字节\behavior_cleaning\outputs\sql_analysis_v2\user_behavior.sqlite" --calendar "D:\字节\holiday_calendar.csv" --advanced
```

日历应完整覆盖观测日期，每日唯一，只含一个地区；列为 `date,is_holiday,region`，示意如下（数值仅示意，必须按实际地区权威日历填写）：

```csv
date,is_holiday,region
2025-11-18,0,指定地区
```

## 预测评估口径与剩余工作

本数据观察至2025-12-18 23时，训练时点为2025-12-05 00时，训练标签窗口为12月5日至11日；验证时点12月12日00时，验证标签窗口为12月12日至18日。特征严格使用各时点之前记录。用户可能同时出现在两个时间截点，因此测量的是已见用户的未来行为预测能力。

SVD、商品共现词表及规则只拟合训练历史；Target Encoding对训练用户折外计算，验证只应用训练统计；缺失值、缩放、方差、相关性、互信息只拟合训练快照。原有清洗异常阈值曾使用全期数据，严谨的生产预测实验还需历史内重估清洗阈值；本报告定位为探索性预评估。

全期商品生命周期和类目报表不能直接合并进预测特征，否则会回看未来。本版预测快照包含用户生命周期、小时序列、漏斗、偏好、主类目编码及稀疏隐式特征；商品生命周期、商品/类目复购等尚需在每个历史时点重新生成，再用于用户商品级模型。

剩余完整任务包括：补充金额/订单信息后的真实RFM和订单复购；更细时间戳下的完整行为序列；全量商品隐式特征与规模化关联规则；历史商品特征接入预测模型；多时间窗口回测和稳定性验证；补充包含假期的行为数据后比较节假日差异。中国大陆2025日历已导入，当前31天没有假期样本。

源代码：`analysis_sql.py`存放SQL，`analysis_runner.py`负责报表，`feature_snapshots.py`构建历史和隐式特征，`feature_validation.py`负责标签、预处理、筛选与预评估。所有模块位于 `src/behavior_cleaning/`。

测试命令：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest tests -q --assert=plain
```

`-B --assert=plain`用于避免旧版pytest向Anaconda安装目录写断言缓存；当前全部37项测试已通过，其中新增7项覆盖时间边界、小时顺序、编码、端到端已知标签，以及中国大陆日历边界和缺少假期样本的空值处理。
