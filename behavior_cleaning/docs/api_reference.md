# Python工具包API

版本1.1.0，Python 3.10。稳定入口集中在`behavior_cleaning.api`，包顶层支持延迟导入。旧CLI与旧实验目录保留。数据清洗不需要加载PyTorch；模型训练与推理需要建模依赖。

## 安装与环境

在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r behavior_cleaning/requirements-lock-reproduction.txt
python -m pip install -e ./behavior_cleaning --no-deps
behavior-toolkit --version
```

锁定清单用于复现本项目已训练产物，不代表新机器上的最新推荐版本。模型文件只能来自可信本地实验；不要载入陌生joblib或PyTorch文件。Git不包含原始数据、权重和逐行预测，新环境需自行提供数据并训练。

## 稳定入口

|API|输入与返回|关键约束|
|---|---|---|
|`clean_data(input_path, output_root, *, chunk_size=100000, workers=2, rules=None)`|CSV文件/目录 → 独立清洗输出Path|保留正常重复；rules为BurstRules；不覆盖既有run|
|`build_features(source, output, *, max_pairs=80000)`|清洗Parquet → 训练快照摘要dict|固定项目四窗口；需2025-11-18至12-18完整数据|
|`TrainingConfig(source, output, max_pairs=80000, trials=12, epochs=8, deep_configurations=2)`|不可变训练配置|每窗口至少200对；trials/epochs为正；配置数1或2|
|`train_models(config, *, include_analysis=True)`|训练配置 → 实验目录Path|七种单模型、原Stacking和校准；完整目录禁止重训覆盖|
|`load_predictor(run_dir, *, models=None)`|实验或导出包 → PurchasePredictor|默认八模型，允许按需指定名称|
|`predictor.predict(source, cutoff, *, max_pairs=80000)`|历史Parquet → 标识、截点和概率DataFrame|无需未来标签；至少3天历史更新到前一小时|
|`predictor.predict_arrays(arrays)`|已编码同口径数组 → 概率DataFrame|用于回放；不代替原始数据预处理|
|`predictor.export_bundle(destination)`|不存在的目标目录 → 本地模型包Path|包含商品词表；不含样本；SHA256检测损坏|

可选model名称：`LogisticRegression`、`XGBoost`、`LightGBM`、`LSTM`、`GRU`、`DIN`、`DIN_MHA`、`Stacking`。这些名称区分大小写。新补充的加权与路由诊断不是本版推理接口中已经验收的独立部署模型。

## 使用示例

```python
from pathlib import Path
from behavior_cleaning import clean_data, TrainingConfig, train_models, load_predictor

cleaned = clean_data(Path('input/events.csv'), Path('behavior_cleaning/outputs'))
# 训练数据必须满足本项目固定四窗口的历史和标签覆盖要求。
run = train_models(TrainingConfig(
    source=cleaned / 'cleaned_events.parquet',
    output=Path('behavior_cleaning/outputs/my_temporal_run'),
))
predictor = load_predictor(run, models=['XGBoost', 'Stacking'])
scores = predictor.predict(
    source=Path('history_only.parquet'), cutoff='2025-12-19 00:00:00'
)
scores.to_parquet('behavior_cleaning/outputs/my_predictions.parquet', index=False)
predictor.export_bundle('behavior_cleaning/artifacts/purchase_v1_1_0')
```

示例中的12月19日是历史数据完整到12月18日23时的本地批量预测截点，并非报告中已有标签验证结果。输出为用户—商品对未来7天出现购买行为的条件概率；购买行为不等于核验支付，概率不是营销增量。

## 命令行

```powershell
behavior-toolkit clean --input input/events.csv --output behavior_cleaning/outputs
behavior-toolkit features --source cleaned.parquet --output behavior_cleaning/outputs/new_run
behavior-toolkit train --source cleaned.parquet --output behavior_cleaning/outputs/new_train --trials 12 --epochs 8
behavior-toolkit predict --run-dir behavior_cleaning/outputs/new_train --source history_only.parquet --cutoff "2025-12-19 00:00:00" --models XGBoost Stacking --output behavior_cleaning/outputs/scores.parquet
behavior-toolkit bundle --run-dir behavior_cleaning/outputs/new_train --output behavior_cleaning/artifacts/purchase_v1_1_0
```

未安装console entry时，可用`python -m behavior_cleaning.toolkit_cli`替换`behavior-toolkit`。输出文件已存在时预测命令拒绝覆盖；训练参数改变时使用新目录。

## 数据与时间契约

清洗后Parquet包含字符串`user_id/item_id/category_id`、1至4的`behavior_type`和本地无时区的小时级`timestamp`。时间截点为整点，历史采用`[cutoff-3天, cutoff)`，不查询未来事件，也不要求输入有target。

原模型的融合校准标签到2025-12-12才成熟，因此拒绝更早时点的真实推理。训练用Target Encoding、填补、log1p、标准化与词表全部复用，不在预测样本重新拟合；类别编码在列拼接前保持float32，防止树边界受精度变化影响。序列最多16个活跃小时、每小时4个商品，同小时池化。未知商品/类目进入UNK。结果按用户、商品排序，抽样资格仅由历史ID哈希决定。

## 既有分析模块

- `database_builder.BuildConfig`及`build_database(config)`：CSV或Parquet导入SQLite、基础表和索引验收。
- `analysis_runner.analyze(database, output, calendar=None)`：只读行为、用户、商品和时间分析。
- `feature_validation.run_validation(database, output, max_items=1000)`：早期用户级61项特征与SVD预评估。
- `feature_snapshots.snapshot(database, cutoff)`：早期历史用户特征；不要与最新用户商品对26项特征混用。
- `scripts/analyze_probability_diagnostics.py`：冻结模型的补充排序、漂移、概率融合和SHAP诊断。

## 错误与资产边界

缺文件抛`FileNotFoundError`，非法配置、特征契约或时间缺失抛`ValueError`，覆盖保护抛`FileExistsError`。DuckDB和模型依赖的底层异常保留原始堆栈，便于定位。空数组返回有列的空结果；非有限编码输入拒绝预测。模型包SHA256只防意外损坏，不是来源认证。该类为离线批量接口，尚未验证线上并发、SLA或在线推荐效果。

## 测试与回放

```powershell
python -B -m pytest --assert=plain -q behavior_cleaning/tests
python behavior_cleaning/scripts/verify_release.py --run-dir behavior_cleaning/outputs/temporal_probability_20261109 --source behavior_cleaning/outputs/run_20261026_181850_f5ed88ab/cleaned_events.parquet --output behavior_cleaning/outputs/release_check.json
```

Windows受限环境使用`-B --assert=plain`避免pytest向只读Anaconda依赖目录写入断言缓存；断言仍执行，只关闭重写。真实回放只保留12月9日至11日历史，核对2,000对、八模型概率与冻结结果，并验证导出包重载。

## 构建分发包

在仓库根目录执行`python -m pip wheel ./behavior_cleaning --no-deps --wheel-dir behavior_cleaning/dist`。
`dist`不纳入Git，源码、版本记录和锁定清单用于复现。预训练权重和词表由本地`export_bundle`生成。
