# src/prepare.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _detect_types(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def _one_hot(X: pd.DataFrame, cat_cols):
    if not cat_cols:
        return X.copy()
    return pd.get_dummies(X, columns=cat_cols, drop_first=True)


def _simple_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    Xs = X.sample(frac=1.0, random_state=random_state)
    ys = y.loc[Xs.index]
    n = len(Xs)
    n_test = max(1, int(n * test_size))
    X_train, X_test = Xs.iloc[:-n_test], Xs.iloc[-n_test:]
    y_train, y_test = ys.iloc[:-n_test], ys.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


def prepare(params_path: str = "params.yaml") -> None:
    # read params
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    raw_csv = params["data"]["raw_csv"]
    target = params["data"]["target"]
    test_size = float(params["data"]["test_size"])
    random_state = int(params["data"]["random_state"])

    # load raw
    df = pd.read_csv(raw_csv)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {raw_csv}")

    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # encode
    num_cols, cat_cols = _detect_types(X)
    X_enc = _one_hot(X, cat_cols)

    # split
    X_train, X_test, y_train, y_test = _simple_split(
        X_enc, y, test_size=test_size, random_state=random_state
    )

    # save
    out_dir = Path("data/processed")
    _ensure_dir(out_dir)

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    # minimal schema for later consistency
    schema = {
        "original_num_cols": num_cols,
        "original_cat_cols": cat_cols,
        "encoded_feature_columns": list(X_enc.columns),
        "target": target,
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))

    print(
        f"Prepared datasets written to {out_dir}\n"
        f"- X_train.csv ({X_train.shape})\n"
        f"- X_test.csv  ({X_test.shape})\n"
        f"- y_train.csv ({y_train.shape})\n"
        f"- y_test.csv  ({y_test.shape})\n"
        f"- schema.json"
    )


def cli():
    ap = argparse.ArgumentParser(description="Prepare processed train/test CSVs from raw CSV using params.yaml.")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = ap.parse_args()
    prepare(args.params)


if __name__ == "__main__":
    cli()