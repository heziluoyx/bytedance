"""训练内分层划分与折外类目目标编码。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_indices(
    y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按标签分层划分70%、20%、10%，不做训练前重采样。"""
    train, rest = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    valid, test = train_test_split(
        rest, test_size=1 / 3, stratify=y[rest], random_state=seed
    )
    return train, valid, test


def encode_category(
    categories, y: np.ndarray, train: np.ndarray, seed: int = 42, smoothing: float = 20
) -> tuple[np.ndarray, dict]:
    """训练折外目标编码，其他行只使用训练统计；ID本身不入数值模型。"""
    categories = np.asarray(categories)
    encoded = np.zeros(len(y), dtype=np.float32)
    prior = float(y[train].mean())

    def mapping(index):
        frame = pd.DataFrame({"category": categories[index], "target": y[index]})
        stats = frame.groupby("category").target.agg(["sum", "count"])
        return (
            (stats["sum"] + smoothing * y[index].mean()) / (stats["count"] + smoothing)
        ).to_dict()

    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fit, held in folds.split(train, y[train]):
        fit_index, held_index = train[fit], train[held]
        values = mapping(fit_index)
        encoded[held_index] = [
            values.get(c, float(y[fit_index].mean())) for c in categories[held_index]
        ]
    values = mapping(train)
    remaining = np.setdiff1d(np.arange(len(y)), train)
    encoded[remaining] = [values.get(c, prior) for c in categories[remaining]]
    return encoded, {"mapping": values, "prior": prior, "smoothing": smoothing}
