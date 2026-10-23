"""Synthetic-only tests for the baseline Random Forest inference contract."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest
import warnings
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd

import inference


def applicant(**changes):
    """A synthetic schema fixture, unrelated to any dataset record."""
    record = {
        "Age": 40,
        "Sex": "female",
        "Job": 2,
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 2500,
        "Duration": 12,
        "Housing": "free",
        "Purpose": "business",
    }
    record.update(changes)
    return record


class FakeForest:
    def __init__(self):
        self.feature_names_in_ = np.asarray(inference.FEATURE_NAMES)
        self.n_features_in_ = 16
        self.classes_ = np.asarray([1, 0])  # Reverse order tests label-based lookup.
        self.probabilities = [0.7, 0.3]  # Good, Bad in this fake order.
        self.prediction = 1
        self.frames = []

    def predict(self, features):
        self.frames.append(features.copy())
        return np.full(len(features), self.prediction)

    def predict_proba(self, features):
        self.frames.append(features.copy())
        return np.tile(self.probabilities, (len(features), 1))


class ValidationTests(unittest.TestCase):
    def assert_field_error(self, record, field):
        with self.assertRaisesRegex(inference.ApplicantValidationError, f"field '{field}'"):
            inference.validate_applicant(record)

    def test_required_schema_and_nonmutation(self):
        raw = applicant(**{"Saving accounts": None})
        original = copy.deepcopy(raw)
        validated = inference.validate_applicant(raw)
        self.assertEqual(set(validated), set(inference.RAW_FIELDS))
        self.assertEqual(validated["Saving accounts"], "Unknown")
        self.assertEqual(raw, original)

    def test_each_missing_required_key(self):
        for field in inference.RAW_FIELDS:
            with self.subTest(field=field):
                raw = applicant()
                del raw[field]
                self.assert_field_error(raw, field)

    def test_extra_keys_are_rejected(self):
        for field in ("Risk", "Unnamed: 0", "Other"):
            with self.subTest(field=field):
                self.assert_field_error(applicant(**{field: 1}), field)

    def test_null_only_for_account_fields(self):
        for field in inference.RAW_FIELDS:
            with self.subTest(field=field):
                raw = applicant(**{field: None})
                if field in ("Saving accounts", "Checking account"):
                    self.assertEqual(inference.validate_applicant(raw)[field], "Unknown")
                else:
                    self.assert_field_error(raw, field)

    def test_account_unknown_and_missing_markers(self):
        for field in ("Saving accounts", "Checking account"):
            self.assertEqual(inference.validate_applicant(applicant(**{field: "Unknown"}))[field], "Unknown")
            for bad in ("", "NA", "na", "None", " Unknown", "Unknown ", float("nan")):
                with self.subTest(field=field, bad=repr(bad)):
                    self.assert_field_error(applicant(**{field: bad}), field)

    def test_category_case_whitespace_and_unsupported_values(self):
        bad_values = {
            "Sex": ("Female", "male ", "other", ""),
            "Housing": ("Free", " own", "other", ""),
            "Purpose": ("Business", "car ", "other", ""),
            "Saving accounts": ("Little", " little", "other"),
            "Checking account": ("Rich", "rich ", "other"),
        }
        for field, values in bad_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assert_field_error(applicant(**{field: value}), field)

    def test_numeric_values_are_strict_integers(self):
        invalid = (True, np.bool_(False), "40", 40.0, 40.5, Decimal("40"), math.nan, math.inf, -math.inf)
        for field in ("Age", "Job", "Credit amount", "Duration"):
            for value in invalid:
                with self.subTest(field=field, value=repr(value)):
                    self.assert_field_error(applicant(**{field: value}), field)
        self.assertEqual(inference.validate_applicant(applicant(Age=np.int64(40)))["Age"], 40)
        self.assert_field_error(applicant(Age=2**63), "Age")

    def test_job_codes_and_no_invented_numeric_bounds(self):
        for code in (0, 1, 2, 3):
            self.assertEqual(inference.validate_applicant(applicant(Job=code))["Job"], code)
        for code in (-1, 4, 99):
            self.assert_field_error(applicant(Job=code), "Job")
        raw = applicant(Age=-2, **{"Credit amount": 0, "Duration": -1})
        validated = inference.validate_applicant(raw)
        self.assertEqual((validated["Age"], validated["Credit amount"], validated["Duration"]), (-2, 0, -1))

    def test_batch_input_and_row_errors(self):
        for value in ([], {}, "not a batch", [applicant(), "not a record"]):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(inference.ApplicantValidationError):
                    inference.encode_batch(value)
        with self.assertRaisesRegex(inference.ApplicantValidationError, "row 1, field 'Purpose'"):
            inference.encode_batch([applicant(), applicant(Purpose="unknown")])


class EncodingTests(unittest.TestCase):
    def test_exact_order_numeric_finite_values(self):
        encoded = inference.encode_applicant(applicant())
        self.assertEqual(encoded.shape, (1, 16))
        self.assertEqual(tuple(encoded.columns), inference.FEATURE_NAMES)
        self.assertTrue(all(pd.api.types.is_integer_dtype(dtype) for dtype in encoded.dtypes))
        self.assertTrue(np.isfinite(encoded.to_numpy()).all())

    def test_sex_and_account_mappings(self):
        for category, code in inference.SEX_CODES.items():
            self.assertEqual(inference.encode_applicant(applicant(Sex=category)).iloc[0]["Sex"], code)
        for field, mapping in (
            ("Saving accounts", inference.SAVING_CODES),
            ("Checking account", inference.CHECKING_CODES),
        ):
            for category, code in mapping.items():
                with self.subTest(field=field, category=category):
                    self.assertEqual(inference.encode_applicant(applicant(**{field: category})).iloc[0][field], code)
            self.assertEqual(inference.encode_applicant(applicant(**{field: None})).iloc[0][field], 0)

    def test_all_housing_categories(self):
        expected = {"free": (0, 0), "own": (1, 0), "rent": (0, 1)}
        for value, pair in expected.items():
            with self.subTest(value=value):
                row = inference.encode_applicant(applicant(Housing=value)).iloc[0]
                self.assertEqual((row["Housing_own"], row["Housing_rent"]), pair)

    def test_all_purpose_categories(self):
        columns = [name for name in inference.FEATURE_NAMES if name.startswith("Purpose_")]
        self.assertEqual(len(columns), 7)
        for value in inference.PURPOSE_VALUES:
            with self.subTest(value=value):
                row = inference.encode_applicant(applicant(Purpose=value)).iloc[0]
                expected = [int(column == f"Purpose_{value}") for column in columns]
                self.assertEqual([row[column] for column in columns], expected)
                self.assertEqual(sum(expected), 0 if value == "business" else 1)

    def test_passthrough_and_batch_consistency(self):
        first = applicant(Age=47, **{"Credit amount": 4321, "Duration": 18, "Purpose": "repairs"})
        other = applicant(Sex="male", Housing="rent", Purpose="car")
        single = inference.encode_applicant(first)
        batch = inference.encode_batch([other, first])
        reordered = inference.encode_batch([first, other])
        pd.testing.assert_frame_equal(single, batch.iloc[[1]].reset_index(drop=True))
        pd.testing.assert_frame_equal(single, reordered.iloc[[0]].reset_index(drop=True))
        self.assertEqual(single.iloc[0]["Age"], 47)
        self.assertEqual(single.iloc[0]["Credit amount"], 4321)
        self.assertEqual(single.iloc[0]["Duration"], 18)


class InferenceTests(unittest.TestCase):
    def test_labelled_probabilities_and_unscaled_features(self):
        fake = FakeForest()
        with patch.object(inference.joblib, "load", return_value=fake) as loader:
            result = inference.predict_applicant(applicant(Age=47, **{"Credit amount": 4321}))
        loader.assert_called_once_with(inference.MODEL_PATH)
        self.assertEqual(result, {"predicted_class": "Good", "probability_bad": 0.3, "probability_good": 0.7})
        self.assertEqual(len(fake.frames), 2)
        self.assertEqual(fake.frames[0].iloc[0]["Age"], 47)
        self.assertEqual(fake.frames[0].iloc[0]["Credit amount"], 4321)
        self.assertEqual(tuple(fake.frames[0].columns), inference.FEATURE_NAMES)

    def test_batch_inference_is_atomic_and_ordered(self):
        fake = FakeForest()
        with patch.object(inference.joblib, "load", return_value=fake) as loader:
            with self.assertRaisesRegex(inference.ApplicantValidationError, "row 1, field 'Housing'"):
                inference.predict_batch([applicant(), applicant(Housing="invalid")])
            loader.assert_not_called()
            results = inference.predict_batch([applicant(), applicant(Sex="male")])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(fake.frames), 2)
        self.assertEqual(fake.frames[0].iloc[0]["Sex"], 0)
        self.assertEqual(fake.frames[0].iloc[1]["Sex"], 1)

    def test_model_schema_and_classes_are_checked(self):
        for mutation in (
            lambda model: setattr(model, "feature_names_in_", np.asarray(["wrong"] * 16)),
            lambda model: setattr(model, "feature_names_in_", 16),
            lambda model: setattr(model, "n_features_in_", 15),
            lambda model: setattr(model, "classes_", np.asarray([0, 2])),
            lambda model: setattr(model, "classes_", np.asarray(["0", "1"])),
            lambda model: setattr(model, "predict_proba", None),
        ):
            fake = FakeForest()
            mutation(fake)
            with self.subTest(mutation=mutation.__code__.co_consts):
                with patch.object(inference.joblib, "load", return_value=fake):
                    with self.assertRaises(inference.ModelArtifactError):
                        inference.predict_applicant(applicant())

    def test_malformed_predictions_and_probabilities_are_rejected(self):
        for probabilities in ([math.nan, 1.0], [-0.1, 1.1], [0.2, 0.2], [math.inf, 0.0]):
            fake = FakeForest()
            fake.probabilities = probabilities
            with self.subTest(probabilities=probabilities):
                with patch.object(inference.joblib, "load", return_value=fake):
                    with self.assertRaises(inference.ModelArtifactError):
                        inference.predict_applicant(applicant())
        fake = FakeForest()
        fake.prediction = 2
        with patch.object(inference.joblib, "load", return_value=fake):
            with self.assertRaisesRegex(inference.ModelArtifactError, "predicted classes"):
                inference.predict_applicant(applicant())
        fake = FakeForest()
        fake.probabilities = [1.0]
        with patch.object(inference.joblib, "load", return_value=fake):
            with self.assertRaisesRegex(inference.ModelArtifactError, "probabilities"):
                inference.predict_applicant(applicant())

    def test_artifact_load_failure_is_clear(self):
        with patch.object(inference.joblib, "load", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(inference.ModelArtifactError, "could not load the Random Forest artifact"):
                inference.predict_applicant(applicant())

    def test_read_only_existing_artifact_smoke(self):
        if not inference.MODEL_PATH.is_file():
            self.skipTest("existing Random Forest artifact is unavailable")
        before = hashlib.sha256(inference.MODEL_PATH.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = inference.predict_applicant(applicant())
        after = hashlib.sha256(inference.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertIn(result["predicted_class"], ("Bad", "Good"))
        self.assertTrue(math.isfinite(result["probability_bad"]))
        self.assertTrue(math.isfinite(result["probability_good"]))
        self.assertAlmostEqual(result["probability_bad"] + result["probability_good"], 1.0)


if __name__ == "__main__":
    unittest.main()
