"""Validate raw applicants and run the saved baseline Random Forest.

Public functions: validate_applicant, encode_applicant, encode_batch,
predict_applicant, and predict_batch. All categories are exact matches.
Predictions are model outputs, not lending decisions or calibrated guarantees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest_credit_model.pkl"

RAW_FIELDS = (
    "Age",
    "Sex",
    "Job",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Housing",
    "Purpose",
)

FEATURE_NAMES = (
    "Age",
    "Sex",
    "Job",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Housing_own",
    "Housing_rent",
    "Purpose_car",
    "Purpose_domestic appliances",
    "Purpose_education",
    "Purpose_furniture/equipment",
    "Purpose_radio/TV",
    "Purpose_repairs",
    "Purpose_vacation/others",
)

SEX_CODES = {"female": 0, "male": 1}
SAVING_CODES = {"Unknown": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
CHECKING_CODES = {"Unknown": 0, "little": 1, "moderate": 2, "rich": 3}
HOUSING_VALUES = ("free", "own", "rent")
PURPOSE_VALUES = (
    "business",
    "car",
    "domestic appliances",
    "education",
    "furniture/equipment",
    "radio/TV",
    "repairs",
    "vacation/others",
)

_NUMERIC_FIELDS = ("Age", "Job", "Credit amount", "Duration")
_INT64 = np.iinfo(np.int64)


class ApplicantValidationError(ValueError):
    """A raw applicant violates the approved input contract."""


class ModelArtifactError(RuntimeError):
    """The saved forest is missing, incompatible, or returned invalid output."""


def _validation_error(field: str, message: str, row: int | None) -> ApplicantValidationError:
    location = f"row {row}, " if row is not None else ""
    return ApplicantValidationError(f"{location}field {field!r}: {message}")


def _integer(value: object, field: str, row: int | None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise _validation_error(field, "must be a finite integer (not a boolean, string, or decimal)", row)
    number = int(value)
    if not _INT64.min <= number <= _INT64.max:
        raise _validation_error(field, "integer is outside the supported int64 representation", row)
    if field == "Job" and number not in (0, 1, 2, 3):
        raise _validation_error(field, "must be one of 0, 1, 2, or 3", row)
    return number


def _category(
    value: object,
    field: str,
    allowed: Mapping[str, int] | Sequence[str],
    row: int | None,
    *,
    allow_null: bool = False,
) -> str:
    if value is None and allow_null:
        value = "Unknown"
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        raise _validation_error(field, f"must exactly match one of: {choices}", row)
    return value


def validate_applicant(applicant: Mapping[str, Any], *, row: int | None = None) -> dict[str, int | str]:
    """Return a validated copy; never coerce or mutate the caller's record.

    Python ``None`` is the only accepted null marker, and only for the two
    account fields. ``row`` identifies a record in batch validation errors.
    """
    if not isinstance(applicant, Mapping):
        location = f"row {row}: " if row is not None else ""
        raise ApplicantValidationError(f"{location}applicant must be a mapping of the nine raw fields")

    for field in RAW_FIELDS:
        if field not in applicant:
            raise _validation_error(field, "missing required field", row)
    for field in applicant:
        if field not in RAW_FIELDS:
            raise _validation_error(str(field), "unexpected extra field", row)

    validated: dict[str, int | str] = {}
    for field in _NUMERIC_FIELDS:
        validated[field] = _integer(applicant[field], field, row)
    validated["Sex"] = _category(applicant["Sex"], "Sex", SEX_CODES, row)
    validated["Saving accounts"] = _category(
        applicant["Saving accounts"], "Saving accounts", SAVING_CODES, row, allow_null=True
    )
    validated["Checking account"] = _category(
        applicant["Checking account"], "Checking account", CHECKING_CODES, row, allow_null=True
    )
    validated["Housing"] = _category(applicant["Housing"], "Housing", HOUSING_VALUES, row)
    validated["Purpose"] = _category(applicant["Purpose"], "Purpose", PURPOSE_VALUES, row)
    return validated


def _encoded_row(applicant: Mapping[str, int | str]) -> tuple[int, ...]:
    housing = applicant["Housing"]
    purpose = applicant["Purpose"]
    return (
        int(applicant["Age"]),
        SEX_CODES[str(applicant["Sex"])],
        int(applicant["Job"]),
        SAVING_CODES[str(applicant["Saving accounts"])],
        CHECKING_CODES[str(applicant["Checking account"])],
        int(applicant["Credit amount"]),
        int(applicant["Duration"]),
        int(housing == "own"),
        int(housing == "rent"),
        *(int(purpose == category) for category in PURPOSE_VALUES if category != "business"),
    )


def _feature_frame(rows: list[tuple[int, ...]]) -> pd.DataFrame:
    features = pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=np.int64)
    if features.shape[1] != 16 or not np.isfinite(features.to_numpy()).all():
        raise ApplicantValidationError("encoded features must be exactly 16 finite numeric values")
    return features


def encode_applicant(applicant: Mapping[str, Any]) -> pd.DataFrame:
    """Encode one applicant as a one-row, fixed-schema, unscaled DataFrame."""
    return _feature_frame([_encoded_row(validate_applicant(applicant))])


def encode_batch(applicants: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Validate every row, then encode the entire batch atomically."""
    if isinstance(applicants, (str, bytes, bytearray, Mapping)) or not isinstance(applicants, Sequence):
        raise ApplicantValidationError("batch must be a non-empty sequence of applicant mappings")
    if not applicants:
        raise ApplicantValidationError("batch must contain at least one applicant")
    rows = [_encoded_row(validate_applicant(applicant, row=index)) for index, applicant in enumerate(applicants)]
    return _feature_frame(rows)


def _load_forest() -> Any:
    try:
        forest = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise ModelArtifactError(f"could not load the Random Forest artifact at {MODEL_PATH}") from exc

    names = getattr(forest, "feature_names_in_", None)
    try:
        observed_names = list(names) if names is not None else None
    except TypeError:
        observed_names = None
    if observed_names != list(FEATURE_NAMES) or getattr(forest, "n_features_in_", None) != 16:
        raise ModelArtifactError("Random Forest feature names or count do not match the required 16-feature schema")
    classes = np.asarray(getattr(forest, "classes_", None))
    if classes.shape != (2,) or classes.dtype.kind not in "iu" or set(classes.tolist()) != {0, 1}:
        raise ModelArtifactError("Random Forest classes_ must contain exactly integer classes 0 and 1")
    if not callable(getattr(forest, "predict", None)) or not callable(getattr(forest, "predict_proba", None)):
        raise ModelArtifactError("Random Forest must expose predict and predict_proba")
    return forest


def _predict_encoded(features: pd.DataFrame) -> list[dict[str, str | float]]:
    forest = _load_forest()
    try:
        predictions = np.asarray(forest.predict(features))
        probabilities = np.asarray(forest.predict_proba(features), dtype=float)
    except Exception as exc:
        raise ModelArtifactError("Random Forest inference failed") from exc

    count = len(features)
    if predictions.shape != (count,) or any(
        isinstance(label, (bool, np.bool_))
        or not isinstance(label, (int, np.integer))
        or int(label) not in (0, 1)
        for label in predictions
    ):
        raise ModelArtifactError("Random Forest returned invalid predicted classes")
    if probabilities.shape != (count, 2) or not np.isfinite(probabilities).all():
        raise ModelArtifactError("Random Forest returned malformed or non-finite probabilities")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-8
    ):
        raise ModelArtifactError("Random Forest probabilities must be within [0, 1] and sum to one")

    class_columns = {int(label): index for index, label in enumerate(forest.classes_)}
    return [
        {
            "predicted_class": "Good" if int(predictions[index]) == 1 else "Bad",
            "probability_bad": float(probabilities[index, class_columns[0]]),
            "probability_good": float(probabilities[index, class_columns[1]]),
        }
        for index in range(count)
    ]


def predict_applicant(applicant: Mapping[str, Any]) -> dict[str, str | float]:
    """Return the forest's class and labelled probabilities for one applicant."""
    return _predict_encoded(encode_applicant(applicant))[0]


def predict_batch(applicants: Sequence[Mapping[str, Any]]) -> list[dict[str, str | float]]:
    """Return ordered predictions; any invalid row aborts the entire batch."""
    return _predict_encoded(encode_batch(applicants))
