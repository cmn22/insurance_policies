# src/evaluate.py
from __future__ import annotations
import argparse
import json
import os
from math import sqrt
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
import yaml


def _load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def _project_root_from(params_path: str) -> Path:
    return Path(params_path).resolve().parent


def _read_processed(root: Path):
    processed = root / "data" / "processed"
    X_train = pd.read_csv(processed / "X_train.csv")
    X_test  = pd.read_csv(processed / "X_test.csv")

    def read_y(path: Path, n_expected: int) -> pd.Series:
        y = pd.read_csv(path).iloc[:, 0]
        if len(y) == n_expected:
            return y
        y2 = pd.read_csv(path, header=None).iloc[:, 0]
        if len(y2) == n_expected:
            return y2
        raise ValueError(f"Inconsistent length for {path}")

    y_train = read_y(processed / "y_train.csv", len(X_train))
    y_test  = read_y(processed / "y_test.csv", len(X_test))
    return X_train, X_test, y_train, y_test


def _evaluate(model, X_train, y_train, X_test, y_test):
    preds_train = model.predict(X_train)
    preds_test  = model.predict(X_test)

    mse_train = mean_squared_error(y_train, preds_train)
    mse_test  = mean_squared_error(y_test, preds_test)
    rmse_train = sqrt(mse_train)
    rmse_test  = sqrt(mse_test)

    r2_train = r2_score(y_train, preds_train)
    r2_test  = r2_score(y_test, preds_test)

    metrics = {
        "rmse": float(rmse_train),
        "rmse_test": float(rmse_test),
        "r2": float(r2_train),
        "r2_test": float(r2_test),
        "mse": float(mse_train),
        "mse_test": float(mse_test),
    }
    return metrics


def _save_reports(root: Path, model, X_train, metrics: dict) -> dict:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Optional feature importances
    try:
        fi = pd.DataFrame(
            {"feature": list(X_train.columns), "importance": getattr(model, "feature_importances_", [])}
        ).sort_values("importance", ascending=False)
        fi.to_csv(reports_dir / "feature_importance.csv", index=False)
        fi_path = reports_dir / "feature_importance.csv"
    except Exception:
        fi_path = None

    return {
        "metrics": str(reports_dir / "metrics.json"),
        "feature_importance": str(fi_path) if fi_path else None,
    }


def _mlflow_log(params: dict, metrics: dict, artifacts: dict, model=None):
    if os.getenv("USE_MLFLOW", "0") != "1":
        print("MLflow disabled (set USE_MLFLOW=1 to enable).")
        return
    try:
        import mlflow
        os.environ.setdefault("MLFLOW_DISABLE_ENV_MANAGER", "1")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "insurance-policies")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # params
            for section, vals in params.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        mlflow.log_param(f"{section}.{k}", v)
                else:
                    mlflow.log_param(section, vals)

            # metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # artifacts
            for _, path in artifacts.items():
                if path and Path(path).is_file():
                    mlflow.log_artifact(path)

            # model (optional)
            if model is not None:
                try:
                    from mlflow import sklearn as mlflow_sklearn
                    mlflow_sklearn.log_model(model, artifact_path="model")
                except Exception as e:
                    print(f"Skipping mlflow.sklearn.log_model: {e}")

        print(f"MLflow run logged to {tracking_uri} (experiment='{experiment_name}').")
    except Exception as e:
        print(f"MLflow logging skipped due to error: {e}")


def evaluate(params_path: str = "params.yaml") -> None:
    params = _load_params(params_path)
    root = _project_root_from(params_path)

    # inputs
    X_train, X_test, y_train, y_test = _read_processed(root)
    model = load(root / "models" / "model.joblib")

    # eval
    metrics = _evaluate(model, X_train, y_train, X_test, y_test)
    artifacts = _save_reports(root, model, X_train, metrics)

    print("Evaluation complete.\nMetrics:", json.dumps(metrics, indent=2))
    print("Artifacts:", json.dumps(artifacts, indent=2))

    # optional MLflow
    _mlflow_log(params, metrics, artifacts, model=model)


def cli():
    ap = argparse.ArgumentParser(description="Evaluate a trained model and save reports/metrics.")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml (defaults to repo root).")
    args = ap.parse_args()
    evaluate(args.params)


if __name__ == "__main__":
    cli()