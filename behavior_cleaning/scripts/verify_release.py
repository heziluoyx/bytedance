"""以真实冻结结果核验历史重建推理与导出包，输出不含逐行标识的检查记录。"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for folder in [root / "src", root / ".runtime", root / ".model_runtime"]:
    sys.path.insert(0, str(folder))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import duckdb
    import numpy as np
    import pandas as pd
    import torch

    from behavior_cleaning import load_predictor

    torch.set_num_threads(2)
    predictor = load_predictor(args.run_dir)
    truth = pd.read_parquet(args.run_dir / "test_predictions.parquet")
    with tempfile.TemporaryDirectory(prefix="release_parity_") as temporary:
        temporary = Path(temporary)
        source = temporary / "history_only.parquet"
        c = duckdb.connect()
        c.execute("SET threads=2")
        c.read_parquet(str(args.source)).create_view("input")
        dest = str(source).replace("'", "''")
        c.execute(
            "COPY (SELECT * FROM input WHERE timestamp >= TIMESTAMP '2025-12-09' AND timestamp < TIMESTAMP '2025-12-12') TO '"
            + dest
            + "' (FORMAT PARQUET)"
        )
        c.close()
        inferred = predictor.predict(source, "2025-12-12", max_pairs=2000)
        joined = inferred.merge(
            truth,
            on=["user_id", "item_id", "cutoff"],
            suffixes=("_inference", "_frozen"),
            validate="one_to_one",
        )
        if len(joined) != len(inferred):
            raise AssertionError("推理抽样不在原冻结候选中")
        errors = {}
        for name in predictor.models:
            errors[name] = float(
                np.max(
                    np.abs(
                        joined[name + "_probability_inference"]
                        - joined[name + "_probability_frozen"]
                    )
                )
            )
            if errors[name] > 2e-6:
                raise AssertionError(f"{name}推理与冻结结果不一致: {errors[name]}")
        exported = predictor.export_bundle(temporary / "bundle")
        reloaded = load_predictor(exported, models=["XGBoost"])
        b = reloaded.predict(source, "2025-12-12", max_pairs=2000)
        pd.testing.assert_frame_equal(
            inferred[["user_id", "item_id", "cutoff", "XGBoost_probability"]],
            b,
            check_exact=False,
            atol=2e-6,
            rtol=0,
        )
        report = {
            "rows": len(inferred),
            "models": list(predictor.models),
            "future_data_present": False,
            "maximum_absolute_probability_errors": errors,
            "bundle_reload_parity": True,
            "status": "passed",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
