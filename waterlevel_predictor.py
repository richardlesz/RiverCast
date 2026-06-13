"""Importable water level predictor using the trained Gradient Boosting model."""

from __future__ import annotations

from functools import lru_cache
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

FEATURE_COLUMNS = ["rainfall_value_T-2", "release_target_T-3", "waterlevel_value_T-1"]
MODELS_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def _get_model():
    model_path = MODELS_DIR / "gradient_boosting.joblib"
    metadata_path = MODELS_DIR / "metadata.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. Run train.py first."
        )

    return joblib.load(model_path), joblib.load(metadata_path)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_row(row, row_index: int | None = None) -> list[float]:
    prefix = f"Row {row_index}: " if row_index is not None else ""

    if not isinstance(row, (list, tuple)):
        raise ValueError(f"{prefix}Each row must be a list or tuple of feature values.")

    if len(row) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{prefix}Expected {len(FEATURE_COLUMNS)} values "
            f"({', '.join(FEATURE_COLUMNS)}), got {len(row)}."
        )

    validated: list[float] = []
    for feature_name, value in zip(FEATURE_COLUMNS, row):
        if value is None:
            raise ValueError(f"{prefix}Missing value for '{feature_name}'.")
        if not _is_number(value):
            raise ValueError(
                f"{prefix}Invalid value for '{feature_name}': {value!r}. "
                "Expected a numeric value."
            )
        validated.append(float(value))

    return validated


def _normalize_input(features) -> tuple[list[list[float]], bool]:
    if not isinstance(features, (list, tuple)):
        raise ValueError("Input must be a list or tuple of feature values.")

    if not features:
        raise ValueError("Input must not be empty.")

    if _is_number(features[0]):
        return [_validate_row(features)], True

    rows = [_validate_row(row, index) for index, row in enumerate(features, start=1)]
    return rows, False


def predict(features):
    """Predict water level from feature list(s).

    Args:
        features: A flat list ``[rainfall_value_T-2, release_target_T-3, waterlevel_value_T-1]``
            for a single prediction, or a list of such rows for batch prediction.

    Returns:
        float for a single input row, or list[float] for batch input.
    """
    rows, is_single = _normalize_input(features)
    model, _metadata = _get_model()

    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    predictions = model.predict(frame)
    result = [float(value) for value in predictions]

    return result[0] if is_single else result
