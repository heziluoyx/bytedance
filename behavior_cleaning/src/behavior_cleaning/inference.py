"""无未来标签的购买概率推理，复用训练特征、词表、序列与校准器。"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .promotion_robustness import NUMERIC
from .temporal_probability_data import collect_sequences, open_events, snapshot

BASE_MODELS = (
    "LogisticRegression",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "GRU",
    "DIN",
    "DIN_MHA",
)
ALL_MODELS = BASE_MODELS + ("Stacking",)


def transform_features(frame, preprocessing):
    """用训练期统计转换原始特征，不fit、不读取target。

    未知类目使用训练先验；无穷值按缺失处理，计数和间隔下界截为0。
    """
    required = set(NUMERIC) | {"category_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("缺少特征列: " + ", ".join(missing))
    raw = frame[NUMERIC].replace([np.inf, -np.inf], np.nan)
    numeric = np.log1p(raw.fillna(preprocessing["medians"]).clip(lower=0)).to_numpy(
        np.float32
    )
    encoder = preprocessing["target_encoder"]
    encoded = (
        frame.category_id.map(encoder["mapping"])
        .fillna(encoder["prior"])
        .to_numpy(np.float32)
    )
    result = (
        preprocessing["scaler"]
        .transform(np.column_stack([numeric, encoded]))
        .astype(np.float32)
    )
    if not np.isfinite(result).all():
        raise ValueError("转换后出现非有限特征，请核对输入类型和预处理文件")
    return result


def sequence_arrays(frame, histories, preprocessing, x, cutoff):
    """将真实小时桶序列转换为与训练相同的16×4输入；不使用真实标签。"""
    users = sorted(histories)
    lookup = {u: i for i, u in enumerate(users)}
    items = np.zeros((len(users), 16, 4), np.int64)
    categories = np.zeros_like(items)
    aux = np.zeros((len(users), 16, 5), np.float32)
    mask = np.zeros((len(users), 16), bool)
    matches = np.zeros((len(frame), 16), np.float32)
    raw_sets = {}
    iv, cv = preprocessing["item_vocab"], preprocessing["category_vocab"]
    for user in users:
        u = lookup[user]
        buckets = histories[user][-16:]
        raw_sets[user] = []
        for j, (hour, values, types) in enumerate(buckets):
            mask[u, j] = True
            raw_sets[user].append({item for item, _ in values})
            for k, (item, cat) in enumerate(values[:4]):
                items[u, j, k] = iv.get(item, 1)
                categories[u, j, k] = cv.get(cat, 1)
            aux[u, j, :4] = types / max(float(types.sum()), 1)
            aux[u, j, 4] = np.log1p(
                (cutoff - pd.Timestamp(hour)).total_seconds() / 3600
            ) / np.log1p(72)
    for i, row in enumerate(frame.itertuples()):
        for j, bucket in enumerate(raw_sets[row.user_id]):
            matches[i, j] = row.item_id in bucket
    return {
        "x": x,
        "y": np.zeros(len(frame), np.int64),  # 仅兼容批次组装签名，模型forward不使用此项。
        "row_users": frame.user_id.map(lookup).to_numpy(np.int64),
        "target_items": frame.item_id.map(iv).fillna(1).to_numpy(np.int64),
        "target_categories": frame.category_id.map(cv).fillna(1).to_numpy(np.int64),
        "sequence_items": items,
        "sequence_categories": categories,
        "sequence_aux": aux,
        "sequence_mask": mask,
        "matches": matches,
    }


class PurchasePredictor:
    """加载可信模型产物，预测近期交互用户商品对未来7天购买概率。

    Args:
        run_dir: 含dataset预处理/摘要和models权重的实验或导出包目录。
        models: 输出模型名；默认输出七种单模型与原Stacking。

    模型按需加载。joblib和PyTorch文件只接受可信来源。此类用于离线批量，
    不是已通过并发延迟验收的在线服务。概率不保证促销期校准，也不表示营销增量。
    """

    def __init__(self, run_dir, *, models=None):
        self.root = Path(run_dir).resolve()
        manifest = self.root / "bundle_manifest.json"
        if manifest.exists():
            for record in json.loads(manifest.read_text("utf-8"))["files"]:
                path = (self.root / record["path"]).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("模型清单路径越界")
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("模型文件校验失败: " + record["path"])
        self.models = tuple(ALL_MODELS if models is None else models)
        if (
            not self.models
            or len(set(self.models)) != len(self.models)
            or set(self.models) - set(ALL_MODELS)
        ):
            raise ValueError("模型名必须为非重复的已支持模型列表")
        required = [
            "dataset/summary.json",
            "dataset/preprocessing.joblib",
            "models/probability_calibrators.joblib",
        ]
        for name in required:
            if not (self.root / name).is_file():
                raise FileNotFoundError(self.root / name)
        self.summary = json.loads(
            (self.root / "dataset/summary.json").read_text("utf-8")
        )
        self.preprocessing = joblib.load(self.root / "dataset/preprocessing.joblib")
        if self.preprocessing["features"] != self.summary["features"]:
            raise ValueError("特征顺序与模型摘要不一致")
        self.calibrators = joblib.load(
            self.root / "models/probability_calibrators.joblib"
        )
        self._models = {}

    def _raw(self, name, arrays, cache):
        if name not in cache:
            cache[name] = self._compute_raw(name, arrays, cache)
        return cache[name]

    def _compute_raw(self, name, arrays, cache):
        from .temporal_probability_models import logit

        if name == "Stacking":
            if name not in self._models:
                self._models[name] = joblib.load(self.root / "models/Stacking.joblib")
            info = self._models[name]
            logits = np.column_stack(
                [logit(self._raw(n, arrays, cache)) for n in info["columns"]]
            )
            return info["model"].predict_proba(info["scaler"].transform(logits))[:, 1]
        if name not in self._models:
            if name in BASE_MODELS[:3]:
                self._models[name] = joblib.load(
                    self.root / "models" / (name + ".joblib")
                )
            else:
                from .temporal_probability_models import load_sequence

                info = json.loads(
                    (self.root / "models/sequence_selection.json").read_text("utf-8")
                )
                self._models[name] = load_sequence(
                    name, info[name], arrays, self.summary, self.root
                )
        model = self._models[name]
        if name in BASE_MODELS[:3]:
            return model.predict_proba(arrays["x"])[:, 1]
        from .model_experiments import deep_predict

        return deep_predict(model, arrays, np.arange(len(arrays["x"])))

    def predict_arrays(self, arrays):
        """对已准备的同口径数组输出校准概率，适合离线回放核对。"""
        from .temporal_probability_models import apply_calibrator

        if arrays["x"].ndim != 2 or arrays["x"].shape[1] != len(
            self.summary["features"]
        ):
            raise ValueError("x维度不匹配")
        if not np.isfinite(arrays["x"]).all():
            raise ValueError("x包含非有限数值")
        if len(arrays["x"]) == 0:
            return pd.DataFrame(
                {n + "_probability": pd.Series(dtype=float) for n in self.models}
            )
        result = {}
        # 当前批次缓存，避免Stacking再次运行其三个基础模型。
        cache = {}
        for name in self.models:
            result[name + "_probability"] = apply_calibrator(
                self.calibrators[name], self._raw(name, arrays, cache)
            )
        return pd.DataFrame(result)

    def predict(self, source, cutoff, *, max_pairs=80000):
        """从历史Parquet预测，无需未来7天数据；按历史ID稳定哈希抽样。

        source采用清洗后的timestamp列，含至少截止时点前完整3天。
        cutoff须为无时区的本地整点，并不早于本模型校准标签成熟时间。
        """
        if max_pairs < 1:
            raise ValueError("max_pairs必须为正数")
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.floor("h"):
            raise ValueError("cutoff必须为不含时区的本地整点")
        trained_until = max(
            pd.Timestamp(w["label_end_exclusive"])
            for w in self.summary["windows"]
            if w["window"] in ["train", "validation", "meta_calibration"]
        )
        if cutoff < trained_until:
            raise ValueError("预测时间早于本模型训练校准信息成熟时间")
        with tempfile.TemporaryDirectory(prefix="behavior_inference_") as temporary:
            c = open_events(Path(source), temporary)
            try:
                low, high = c.execute(
                    "SELECT MIN(event_time),MAX(event_time) FROM events"
                ).fetchone()
                if (
                    low is None
                    or pd.Timestamp(low) > cutoff - pd.Timedelta(days=3)
                    or pd.Timestamp(high) < cutoff - pd.Timedelta(hours=1)
                ):
                    raise ValueError("历史观察期不完整或数据未更新到预测前一小时")
                frame, info = snapshot(c, cutoff, max_pairs, labeled=False)
                frame = frame.sort_values(
                    ["user_id", "item_id"], kind="stable"
                ).reset_index(drop=True)
                if len(frame):
                    arrays = sequence_arrays(
                        frame,
                        collect_sequences(c),
                        self.preprocessing,
                        transform_features(frame, self.preprocessing),
                        cutoff,
                    )
                    result = pd.concat(
                        [
                            frame[["user_id", "item_id", "cutoff"]].reset_index(
                                drop=True
                            ),
                            self.predict_arrays(arrays),
                        ],
                        axis=1,
                    )
                else:
                    result = frame[["user_id", "item_id", "cutoff"]].copy()
                    for name in self.models:
                        result[name + "_probability"] = pd.Series(dtype=float)
            finally:
                c.close()
        result.attrs.update(info)
        return result

    def export_bundle(self, destination):
        """导出权重、预处理、词表、摘要和SHA256清单，不含训练样本。

        词表含商品编码，导出包仍是本地受控模型资产，不自动上传Git。
        destination必须不存在；这里只复制可信模型文件，不执行重训练。
        """
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        files = (
            [
                self.root / "dataset/preprocessing.joblib",
                self.root / "dataset/summary.json",
            ]
            + sorted((self.root / "models").glob("*.joblib"))
            + sorted((self.root / "models").glob("*.pt"))
            + sorted((self.root / "models").glob("*.json"))
        )
        destination.mkdir(parents=True, exist_ok=False)
        records = []
        for source in files:
            rel = source.relative_to(self.root)
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append(
                {
                    "path": rel.as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        (destination / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "package_version": "1.1.0",
                    "trusted_artifact_required": True,
                    "files": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination
