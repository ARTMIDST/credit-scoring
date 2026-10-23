# Stage F2 inference implementation report

## Result and readiness

**READY FOR ACCEPTANCE TESTING.** The new module validates raw applicant fields, produces the exact 16-feature unscaled schema, and returns the saved baseline Random Forest's class and labelled probabilities. All 20 automated tests passed, including a read-only smoke test with the existing forest. This is an implementation milestone, **not** validation for real-world lending use.

## Files created and modified

| File | Change |
| --- | --- |
| `inference.py` | New small module for validation, fixed encoding, artifact compatibility checks, and single/batch inference. |
| `tests/test_inference.py` | New standard-library `unittest` suite using synthetic applicants, fake estimators, and one read-only existing-artifact smoke test. |
| `docs/inference-implementation-report.md` | This report. |

No existing file was modified. The two earlier audit/specification files were already untracked before Stage F2 and remain untouched. No file was staged, committed, or pushed; no remote or Git configuration was changed.

## Module location and public interface

The root `inference.py` module follows this repository's simple notebook-and-model layout without adding a package, web API, database, or extra dependency. Its public functions are:

| Function | Input | Return |
| --- | --- | --- |
| `validate_applicant(applicant)` | One mapping of the nine raw fields | A validated, normalized copy; the two allowed `None` account values become `Unknown`. |
| `encode_applicant(applicant)` | One raw mapping | One-row, 16-column, unscaled integer `pandas.DataFrame`. |
| `encode_batch(applicants)` | Non-empty sequence of raw mappings | `N`-row, 16-column, unscaled integer `DataFrame`. |
| `predict_applicant(applicant)` | One raw mapping | Dictionary with `predicted_class`, `probability_bad`, and `probability_good`. |
| `predict_batch(applicants)` | Non-empty sequence of raw mappings | Ordered list of the same dictionaries. |

`ApplicantValidationError` identifies an invalid field and, for batches, a **zero-based row index**. `ModelArtifactError` identifies missing or incompatible model artifacts and malformed model outputs. Functions neither mutate caller records nor return a lending decision.

## Exact preprocessing implementation

All nine raw keys are required: `Age`, `Sex`, `Job`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Housing`, and `Purpose`. `Age`, `Job`, `Credit amount`, and `Duration` pass through as integers. `Sex` maps `female=0`, `male=1`. Savings maps `Unknown=0`, `little=1`, `moderate=2`, `quite rich=3`, `rich=4`. Checking maps `Unknown=0`, `little=1`, `moderate=2`, `rich=3`.

Housing uses `free` as the all-zero reference; `own` sets `Housing_own=1`, and `rent` sets `Housing_rent=1`. Purpose uses `business` as the all-zero reference; each of the seven other supported purposes sets exactly its matching `Purpose_*` column to one. The encoded order is exactly:

```text
Age, Sex, Job, Saving accounts, Checking account, Credit amount, Duration,
Housing_own, Housing_rent, Purpose_car, Purpose_domestic appliances,
Purpose_education, Purpose_furniture/equipment, Purpose_radio/TV,
Purpose_repairs, Purpose_vacation/others
```

The module uses explicit constants and row construction. It does **not** call `pd.get_dummies()` on incoming records, discover batch categories, fit an encoder, scale features, or change target encoding. Every returned encoded value is integer and finite. The only numeric restriction beyond the approved `Job` codes 0–3 is representability as `int64`, required by the fixed numeric table; no business bounds were invented for age, credit amount, or duration.

## Input validation and batch behaviour

The implementation rejects omitted keys and **all** extra keys, including `Risk` and `Unnamed: 0`. Python `None` is accepted only for the two account fields and becomes `Unknown`; their exact `Unknown` string is also accepted. Empty strings, literal `NA`, other missing markers, category aliases, capitalization or whitespace variants, and unsupported categories are rejected with field-specific errors. Numeric fields accept Python or NumPy integers within the `int64` representation, but reject booleans, strings, floats (including integral-looking floats), decimals, NaN, and infinity. `Job` is restricted to 0, 1, 2, or 3.

Each record uses the same fixed schema regardless of batch composition or row order. `encode_batch` validates the complete non-empty batch before returning a table. `predict_batch` completes encoding before loading the model, so any invalid row prevents **all** batch inference; there are no partial predictions. Validation does not trim, coerce, round, impute beyond the approved account nulls, or silently map an unexpected category to a reference category.

## Model loading and inference behaviour

Only `models/random_forest_credit_model.pkl` is loaded, via a path anchored to the module directory. Before prediction, the module compares the artifact's `feature_names_in_` and `n_features_in_` with the exact 16-feature contract. It requires integer classes `{0, 1}` and callable `predict` and `predict_proba` interfaces. Predicted labels must be 0 or 1. Probability output must have one row per input and two columns, contain finite values within `[0, 1]`, and sum to one within `1e-8`.

`classes_`, rather than an assumed probability-column position, maps `0` to **Bad** and `1` to **Good**. The return dictionary reports the forest's own predicted class plus both labelled probabilities. It does not add a threshold or approval/rejection rule. These are model outputs, **not calibrated lending probabilities or guarantees**.

`models/standard_scaler.pkl` is never imported, loaded, or called by `inference.py`. The saved forest was fitted on unscaled features; applying the Logistic Regression scaler would change the inputs it expects. Tests verify the loader receives only the forest path and that the fake estimator sees original unscaled numeric values.

## Automated test results

Command: `python -B -m unittest discover -s tests -v` from the project root, using the available Python 3.14.7 installation. The existing project `.venv/` remains incomplete and was not used or altered.

**Final result: 20 tests run; 20 passed; 0 failures; 0 errors; 0 skipped.** The first run had 19 passing tests; a load-error case was then added, and the final suite was rerun in full. The suite covers all required keys, missing and extra keys, account null/`Unknown`, invalid missing markers, exact categories, strict numeric types, Job codes, every mapping and dummy category, both all-zero reference groups, exact feature order and finite values, single/batch invariance, atomic invalid-batch rejection, unscaled model input, class-order-independent probability labels, incompatible model schemas/classes, malformed probabilities and predictions, and load errors.

Unit tests use synthetic records and a fake estimator. The integration smoke test loads the existing forest read-only, predicts for a synthetic already validated applicant, checks labelled probabilities, and confirms its file hash is unchanged. No real applicant-level dataset row was read or used in a test. The test suppresses NumPy deprecation warnings from loading the artifact; the [artifact audit](model-artifact-publication-audit.md) records those warnings.

## Preservation verification

| Protected file | SHA-256 before | SHA-256 after |
| --- | --- | --- |
| `models/random_forest_credit_model.pkl` | `9c1260524fb72ed568092eee8b3ad7daad89b5c91e2d700da470643d3e7263a2` | Same |
| `models/standard_scaler.pkl` | `114c4d27ac3f3da013fb2f5507775996e1e55555be5aacd341efdd8f93fb02bf` | Same |

The notebook (`928c97314e7d91076a3f62125368d1a129195ca683b202ebba8b1d4f0e3294c8`), dataset (`42be3b82a2e5073bd5ca23bce1d1c31426b78f72d20cd892f3aacaa2ba30a075`), `.gitignore` (`a90ab2dc3f777c2ebf0c8c6668f707d42188f2bb1dc61c03bc44e232d7afbaa5`), and `requirements.txt` (`00f4669a10507d67c23e74357ca7328c246155504d1b5fa66fe7cb2bfc95b4ae`) also retain their prior hashes. Accepted model results and methodology were not changed. Neither the notebook nor the dataset was executed or modified.

## Warnings and limitations

- Joblib/pickle artifacts must be loaded only from a trusted source. The working environment loads this artifact, but the incomplete `.venv/` is not a verified project environment. The existing artifact audit records NumPy 2.5 deprecation warnings during load; this implementation did not modify or convert the artifact.
- The `int64` representation check is technical. The approved policy deliberately supplies no business bounds for age, credit amount, or duration, so syntactically valid but implausible integers can still reach the model. Acceptance testing should review that consequence without silently changing the agreed contract.
- The tests establish deterministic encoding and mechanical integration with the existing artifact using synthetic inputs. They do not establish calibration, fairness, generalization to new populations, or fitness for real lending decisions.
- The local dataset and model artifacts remain ignored by Git. The new code, test, and report remain untracked. No stage, commit, push, or remote operation occurred.
