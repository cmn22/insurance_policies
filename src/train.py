# src/train.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

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


def _train_rf(X_train, y_train, model_params: dict):
    rf = RandomForestRegressor(
        n_estimators=int(model_params.get("n_estimators", 200)),
        max_depth=None if (md := model_params.get("max_depth")) in (None, "None") else int(md),
        n_jobs=int(model_params.get("n_jobs", -1)),
        random_state=int(model_params.get("random_state", 42)),
    )
    rf.fit(X_train, y_train)
    return rf


def train(params_path: str = "params.yaml") -> None:
    params = _load_params(params_path)
    root = _project_root_from(params_path)
    params.setdefault("model", {})

    X_train, X_test, y_train, y_test = _read_processed(root)
    model = _train_rf(X_train, y_train, params["model"])

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    dump(model, model_path)

    feature_names = list(X_train.columns)
    (models_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"Model saved to {model_path}")
    print(f"Feature names saved to {models_dir / 'feature_names.json'}")


def cli():
    ap = argparse.ArgumentParser(description="Train a regression model from processed CSVs.")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml (defaults to repo root).")
    args = ap.parse_args()
    train(args.params)


if __name__ == "__main__":
    cli()