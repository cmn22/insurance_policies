# src/train.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root by default

try:
    import yaml
except Exception:
    raise SystemExit("pyyaml is required: pip install pyyaml")


def _load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def _project_root_from(params_path: str) -> Path:
    # Treat the directory that contains params.yaml as project root
    return Path(params_path).resolve().parent


def _read_processed(root: Path):
    processed = root / "data" / "processed"
    X_train = pd.read_csv(processed / "X_train.csv")
    X_test  = pd.read_csv(processed / "X_test.csv")

    def read_y(path: Path, n_expected: int) -> pd.Series:
        # try with header row (default)
        y = pd.read_csv(path).iloc[:, 0]
        if len(y) == n_expected:
            return y
        # fallback: file might have been saved without header
        y2 = pd.read_csv(path, header=None).iloc[:, 0]
        if len(y2) == n_expected:
            return y2
        raise ValueError(
            f"Inconsistent length when reading {path}. "
            f"Got {len(y)} (header) and {len(y2)} (no header), "
            f"expected {n_expected}."
        )

    y_train = read_y(processed / "y_train.csv", len(X_train))
    y_test  = read_y(processed / "y_test.csv", len(X_test))
    return X_train, X_test, y_train, y_test


def _train_rf(X_train, y_train, model_params: dict):
    rf = RandomForestRegressor(
        n_estimators=int(model_params.get("n_estimators", 200)),
        max_depth=None if (md := model_params.get("max_depth")) in (None, "None") else int(md),
        n_jobs=int(model_params.get("n_jobs", -1)),
        random_state=int(model_params.get("random_state", 42)),
    )
    rf.fit(X_train, y_train)
    return rf


from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

def _evaluate(model, X_train, y_train, X_test, y_test):
    preds_train = model.predict(X_train)
    preds_test  = model.predict(X_test)

    # Compute MSE, then RMSE manually (works on all sklearn versions)
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


def _save_artifacts(root: Path, model, X_train, metrics: dict):
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    dump(model, model_path)

    feature_names = list(X_train.columns)
    (models_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Optional feature importances
    try:
        import pandas as pd
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": getattr(model, "feature_importances_", [])
        }).sort_values("importance", ascending=False)
        fi.to_csv(reports_dir / "feature_importance.csv", index=False)
    except Exception:
        pass

    return {
        "model": str(model_path),
        "feature_names": str(models_dir / "feature_names.json"),
        "metrics": str(reports_dir / "metrics.json"),
        "feature_importance": str(reports_dir / "feature_importance.csv"),
    }


def _mlflow_log(params: dict, artifacts: dict, metrics: dict, model=None):
    """
    Enable by setting USE_MLFLOW=1 (via .env or shell).
    Logs params, metrics, artifacts; optionally logs the sklearn model.
    """
    if os.getenv("USE_MLFLOW", "0") != "1":
        print("MLflow disabled (set USE_MLFLOW=1 to enable).")
        return

    try:
        import mlflow
        # Optional: avoid env manager
        os.environ.setdefault("MLFLOW_DISABLE_ENV_MANAGER", "1")

        # Prefer SQLite to avoid FileStore deprecation warning
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "insurance-policies")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log params (support nested dicts)
            for section, vals in params.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        mlflow.log_param(f"{section}.{k}", v)
                else:
                    mlflow.log_param(section, vals)

            # Log flat metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # Log artifacts (files)
            for name, path in artifacts.items():
                p = Path(path)
                if p.exists() and p.is_file():
                    mlflow.log_artifact(str(p))

            # Optional: log model in a standard way
            if model is not None:
                try:
                    from mlflow import sklearn as mlflow_sklearn
                    mlflow_sklearn.log_model(model, artifact_path="model")
                except Exception as e:
                    print(f"Skipping mlflow.sklearn.log_model: {e}")

        print(f"MLflow run logged to {tracking_uri} (experiment='{experiment_name}').")
    except Exception as e:
        print(f"MLflow logging skipped due to error: {e}")


def train(params_path: str = "params.yaml") -> None:
    params = _load_params(params_path)
    root = _project_root_from(params_path)
    params.setdefault("model", {})
    X_train, X_test, y_train, y_test = _read_processed(root)
    model = _train_rf(X_train, y_train, params["model"])
    metrics = _evaluate(model, X_train, y_train, X_test, y_test)
    artifacts = _save_artifacts(root, model, X_train, metrics)
    print("Training complete.\nMetrics:", json.dumps(metrics, indent=2))
    print("Artifacts:", json.dumps(artifacts, indent=2))
    _mlflow_log(params, artifacts, metrics, model=model)


def cli():
    ap = argparse.ArgumentParser(description="Train a regression model from processed CSVs.")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml (defaults to repo root).")
    args = ap.parse_args()
    train(args.params)


if __name__ == "__main__":
    cli()