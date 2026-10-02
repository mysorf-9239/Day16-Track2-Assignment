"""Run the Lab 16 CPU benchmark on the real Kaggle dataset, on the cloud VM."""

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def measured_seconds(model, data, repeats):
    """Median predict_proba duration; input slicing and warm-up are excluded."""
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(data)
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def main():
    seed = 16
    started = time.perf_counter()
    df = pd.read_csv("creditcard.csv")
    load_seconds = time.perf_counter() - started
    if df.shape != (284807, 31) or int(df["Class"].sum()) != 492:
        raise ValueError("Unexpected dataset: check the original Kaggle download.")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    X, y = df.drop(columns="Class"), df["Class"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    # 0.25 of the remaining 80% is 20% of the complete dataset.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed,
        stratify=y_trainval,
    )
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, random_state=seed,
        n_jobs=2, verbosity=-1,
    )
    print("Training LightGBM on the cloud CPU VM...", flush=True)
    started = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)], eval_metric="auc",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    training_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    one_row, batch = X_test.iloc[:1], X_test.iloc[:1000]
    model.predict_proba(one_row)
    model.predict_proba(batch)
    single_seconds = measured_seconds(model, one_row, 50)
    batch_seconds = measured_seconds(model, batch, 10)
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "cloud": "GCP",
        "project_id": os.environ.get("LAB_PROJECT_ID"),
        "zone": os.environ.get("LAB_ZONE"),
        "machine_type": os.environ.get("LAB_MACHINE_TYPE"),
        "source_commit": "55539f67d7c78b43afe334a2ec3271c4bfdbbe2d",
        "versions": {
            "python": platform.python_version(), "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_rows": len(df), "dataset_columns": len(df.columns),
        "fraud_rows": int(y.sum()), "missing_values": 0, "seed": seed,
        "split": {
            "train": len(X_train), "validation": len(X_valid), "test": len(X_test),
        },
        "n_jobs": 2, "decision_threshold": 0.5,
        "data_load_seconds": load_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "latency_1_row_ms": single_seconds * 1000,
        "latency_repeats": 50,
        "batch_rows": len(batch), "batch_repeats": 10,
        "batch_1000_rows_seconds": batch_seconds,
        "throughput_1000_rows_per_second": len(batch) / batch_seconds,
        "timing_summary": "median; warm-up excluded; predict_proba on pandas input",
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    Path("benchmark_result.json").write_text(output + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
