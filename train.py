#!/usr/bin/env python3
"""Train tuned Gradient Boosting model to predict waterlevel_value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("models")
RESULTS_PATH = Path("results/training_results.json")

TARGET = "waterlevel_value"
FEATURE_COLUMNS = ["rainfall_value_T-2", "release_target_T-3", "waterlevel_value_T-1"]

PARAM_DIST = {
    "model__n_estimators": [50, 100, 200, 300],
    "model__max_depth": [3, 5, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__min_samples_leaf": [1, 3, 5, 10],
}


def load_training_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path, parse_dates=["date"])
    missing = [col for col in [TARGET, *FEATURE_COLUMNS] if col not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    clean = data.dropna(subset=[TARGET, *FEATURE_COLUMNS])
    return clean[FEATURE_COLUMNS], clean[TARGET]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("data/worked_snowy_river_data_merged.csv"),
        help="Path to training CSV",
    )
    parser.add_argument("--folds", type=int, default=10, help="CV folds (default: 10)")
    args = parser.parse_args()

    features, target = load_training_data(args.csv_path)
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        PARAM_DIST,
        n_iter=40,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(features, target)

    best_model = search.best_estimator_
    scores = cross_validate(
        best_model,
        features,
        target,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        n_jobs=-1,
    )

    metadata = {
        "target_column": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": ["streamflow_value"],
        "training_rows": len(features),
        "cv_folds": args.folds,
        "best_params": search.best_params_,
        "metrics": {
            "r2_mean": float(scores["test_r2"].mean()),
            "r2_std": float(scores["test_r2"].std()),
            "mae_mean": float(-scores["test_mae"].mean()),
            "mae_std": float(-scores["test_mae"].std()),
            "rmse_mean": float(-scores["test_rmse"].mean()),
            "rmse_std": float(-scores["test_rmse"].std()),
        },
        "feature_importance": dict(
            sorted(
                zip(
                    FEATURE_COLUMNS,
                    best_model.named_steps["model"].feature_importances_,
                ),
                key=lambda item: -item[1],
            )
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "gradient_boosting.joblib")
    joblib.dump(metadata, MODELS_DIR / "metadata.joblib")
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Trained Gradient Boosting model")
    print(f"Training rows: {metadata['training_rows']}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Best params: {metadata['best_params']}")
    print(
        "CV metrics: "
        f"R²={metadata['metrics']['r2_mean']:.4f} ± {metadata['metrics']['r2_std']:.4f}, "
        f"MAE={metadata['metrics']['mae_mean']:.4f} ± {metadata['metrics']['mae_std']:.4f}, "
        f"RMSE={metadata['metrics']['rmse_mean']:.4f} ± {metadata['metrics']['rmse_std']:.4f}"
    )
    print(f"\nSaved model to: {(MODELS_DIR / 'gradient_boosting.joblib').resolve()}")
    print(f"Saved results to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
