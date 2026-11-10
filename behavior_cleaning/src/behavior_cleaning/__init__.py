"""用户行为分析工具包；清洗、特征、训练与推理支持按需导入。"""

__version__ = "1.1.0"

__all__ = [
    "TrainingConfig",
    "clean_data",
    "build_features",
    "train_models",
    "load_predictor",
]


def __getattr__(name):
    """轻量包导入不提前加载PyTorch或其他可选建模依赖。"""
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)
