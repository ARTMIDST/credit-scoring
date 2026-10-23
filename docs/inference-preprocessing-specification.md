# Raw-applicant preprocessing specification

## Purpose and scope

This document specifies the input transformation required to call the **existing accepted baseline Random Forest** with a new applicant. It is the approved contract implemented by [inference.py](../inference.py), not a change to the trained model. The forest expects 16 numeric features in a fixed order and was fitted on **unscaled** data. The original CSV and retraining are unnecessary at prediction time if raw inputs are transformed exactly as specified here. The completed [inference implementation report](inference-implementation-report.md) records the implementation and tests.

The notebook is authoritative for transformations: cell 27 handles missing account values, cell 31 records observed categories, cell 33 maps `Risk`, `Sex`, and account categories, cell 34 creates dummy columns, cell 36 separates 16 predictors from the target, cell 38 fits a scaler for Logistic Regression, and cell 49 trains the baseline forest on unscaled `X_train`. The [artifact audit](model-artifact-publication-audit.md) records the forest's stored feature names and confirms that the scaler and forest share the same input schema. The [model README](../models/README.md), [project documentation](project_documentation.md), and [root README](../README.md) provide context and accepted results.

## Required raw input schema

One applicant must provide the following **nine predictor fields** with the exact source column names. `Risk` is the training target and is **not** an input. `Unnamed: 0` is a CSV export index removed by the notebook and is **not** an applicant attribute.

| Raw field | Observed notebook type | Contract for a valid value |
| --- | --- | --- |
| `Age` | `int64` | Finite integer in the source data's age unit; no string or boolean coercion. No business bounds were approved. |
| `Sex` | string | Exactly `female` or `male`. |
| `Job` | `int64` | Finite integer code restricted to 0, 1, 2, or 3 and passed through unchanged. The meaning of each code is not established here. |
| `Housing` | string | Exactly `free`, `own`, or `rent`. |
| `Saving accounts` | string, with missing values | Exactly `Unknown`, `little`, `moderate`, `quite rich`, or `rich` after the approved null handling below. |
| `Checking account` | string, with missing values | Exactly `Unknown`, `little`, `moderate`, or `rich` after the approved null handling below. |
| `Credit amount` | `int64` | Finite integer passed through unchanged; no business bounds were approved, and the unit is not established here. |
| `Duration` | `int64` | Finite integer passed through unchanged; no business bounds were approved, and the unit is not established here. |
| `Purpose` | string | Exactly `business`, `car`, `domestic appliances`, `education`, `furniture/equipment`, `radio/TV`, `repairs`, or `vacation/others`. |

These are the notebook's data types and categories, not a claim that all numeric values in those types are suitable lending inputs. The implementation validates one record before encoding and every row in a batch. Category matching is exact and case-sensitive; no whitespace trimming, capitalization changes, or aliases are approved.

## Exact transformations

1. **Account missing values:** The notebook fills null/NA values in `Saving accounts` and `Checking account` with the literal category `Unknown`. It does not impute any other field.
2. **Passthrough numeric fields:** `Age`, `Job`, `Credit amount`, and `Duration` retain their numeric values. `Job` is not one-hot encoded or rescaled in the forest path.
3. **Sex mapping:** `female` → `0`; `male` → `1`.
4. **Savings mapping:** `Unknown` → `0`; `little` → `1`; `moderate` → `2`; `quite rich` → `3`; `rich` → `4`.
5. **Checking mapping:** `Unknown` → `0`; `little` → `1`; `moderate` → `2`; `rich` → `3`.
6. **Housing dummies:** `free` is the dropped reference category: both `Housing_own` and `Housing_rent` are `0`. `own` sets only `Housing_own` to `1`; `rent` sets only `Housing_rent` to `1`.
7. **Purpose dummies:** `business` is the dropped reference category: all seven `Purpose_*` columns are `0`. Each other allowed purpose sets exactly its matching dummy to `1`, with the other six at `0`.
8. **Target:** The notebook maps `bad` → `0` and `good` → `1` during training. Do not encode or pass `Risk` for inference. The saved model's `predict_proba` columns correspond to `[bad, good]` in that order; choosing a decision threshold or using a probability in a lending process is outside this preprocessing contract.

The omitted housing and purpose categories follow the notebook's observed category lists and the saved model's exact feature names. Reproduce the resulting columns explicitly. **Do not** run `pd.get_dummies(..., drop_first=True)` independently on each new record or batch: the columns and dropped category would then depend on which categories happen to occur in that input.

## Exact encoded feature order

Build a numeric table with **exactly** these 16 columns, in this order, for every applicant. Names are case sensitive, including spaces and slashes.

| Position | Feature | Source or encoding |
| ---: | --- | --- |
| 1 | `Age` | Raw numeric value |
| 2 | `Sex` | `female=0`, `male=1` |
| 3 | `Job` | Raw numeric code |
| 4 | `Saving accounts` | Savings mapping above |
| 5 | `Checking account` | Checking mapping above |
| 6 | `Credit amount` | Raw numeric value |
| 7 | `Duration` | Raw numeric value |
| 8 | `Housing_own` | `1` only for `own` |
| 9 | `Housing_rent` | `1` only for `rent` |
| 10 | `Purpose_car` | `1` only for `car` |
| 11 | `Purpose_domestic appliances` | `1` only for `domestic appliances` |
| 12 | `Purpose_education` | `1` only for `education` |
| 13 | `Purpose_furniture/equipment` | `1` only for `furniture/equipment` |
| 14 | `Purpose_radio/TV` | `1` only for `radio/TV` |
| 15 | `Purpose_repairs` | `1` only for `repairs` |
| 16 | `Purpose_vacation/others` | `1` only for `vacation/others` |

The table must contain no extra dummy, target, export-index, or raw text column. The implementation asserts equality with the model's `feature_names_in_` before calling `predict` or `predict_proba`.

## Missing values and validation rules

**Established by the notebook:** only null/NA values in the two account fields become `Unknown`. In the original CSV loading path, pandas may parse a textual missing marker such as `NA` as null before `fillna` runs. A raw API or form input will not necessarily do that. The notebook does not safely handle a missing `Sex`, `Housing`, `Purpose`, `Age`, `Job`, `Credit amount`, or `Duration` value. It also does not safely handle an omitted column.

**Approved contract:** require all nine field keys. Permit an explicit null only in the two account fields and map it to `Unknown`; accept the exact `Unknown` string. Reject omitted account keys and null in every other field. Reject blank strings, textual `NA`, non-finite numbers, booleans, numeric strings, and decimals in numeric fields, unexpected categories, and case or whitespace variants. Reject every extra raw key, including `Risk` and `Unnamed: 0`. Restrict `Job` to integer codes 0, 1, 2, and 3. Do not silently impute beyond the two account fields, clamp, round, or remap values; do not invent bounds for `Age`, `Credit amount`, or `Duration`.

An unexpected `Sex` or account category would become a missing numeric value under the notebook's `.map` operation. An absent or unexpected `Housing` or `Purpose` category can be confused with the all-zero reference-category encoding. The implementation validates against the allowed sets **before** dummy construction and rejects unsupported values with field-specific errors. Trimming whitespace or changing case would be a new, unapproved rule.

## Single-record and batch consistency

- Encode each validated record with the same fixed mapping and 16-column order. Batch composition must not affect a row's encoding.
- A single-record request produces a one-row, 16-column numeric table. A batch of `N` valid records produces an `N`-row, 16-column table with exactly the same column names, order, and meanings.
- The encoding of a record alone must equal its row in any batch, regardless of the other records' categories or order. No fit operation, category discovery, per-batch `drop_first`, or scaler fitting is allowed at inference time.
- Validation errors identify the affected field and batch row. The approved batch policy is all-or-nothing: one invalid row rejects the entire batch before inference.
- Pass the unscaled table directly to the accepted baseline forest. The saved `StandardScaler` was fitted for Logistic Regression; scaling forest inputs would change their representation from what the forest saw during fitting.

## Resolved owner decisions

The owner approved these six policies for the completed implementation:

1. Require all nine keys. Accept only explicit null or exact `Unknown` for missing account values; reject omitted keys, empty strings, textual `NA`, and other missing markers.
2. Match categories exactly and case-sensitively, without trimming, capitalization changes, or aliases.
3. Require finite integers for numeric fields; reject booleans, numeric strings, decimals, and non-finite values. Restrict `Job` to 0–3 without inventing business bounds for `Age`, `Credit amount`, or `Duration`.
4. Reject all extra keys, including `Risk` and `Unnamed: 0`.
5. Reject unsupported categories with field-specific errors; never substitute them or encode them as reference categories.
6. Reject an entire batch if any row is invalid. Return the predicted `Bad` or `Good` class and both labelled model probabilities, without making a lending decision.

## Acceptance tests

These were the proposed acceptance checks for this specification. The completed [inference implementation report](inference-implementation-report.md) records the tests run and their results.

1. Assert that every valid encoded result has exactly the ordered 16-feature schema above and numeric, finite values. Compare column names directly with the loaded forest's `feature_names_in_` before inference.
2. Table-test every `Sex`, savings, and checking mapping, including explicit null → `Unknown` for both account fields.
3. Table-test all three housing categories and all eight purpose categories. `free` and `business` must yield their respective all-zero dummy groups; every other category must yield exactly one matching `1`.
4. Encode the same synthetic record alone and in several batches containing different categories and orders; the encoded row must be identical each time.
5. Reject omitted required fields, null in non-account fields, unknown category labels, invalid numeric types, non-finite values, and extra fields under the approved policy. Confirm failures do not silently become reference-category dummies or missing numeric encodings.
6. Confirm that the forest receives unscaled features and that the scaler is never called in the forest inference path.
7. Use non-applicant synthetic fixtures to verify that the encoded table can be passed to the saved forest and that `predict_proba` columns retain `[bad, good]` semantics. Do not use this as proof of real-world validity.

## Explicit limitations and implementation status

The model artifact itself contains no fitted raw-data transformer. The notebook's batch-wide dummy creation is not a reliable standalone encoder for a new applicant; the separate `inference.py` module implements this fixed contract. The StandardScaler is excluded and must not be used with the forest. The existing metrics come from a small historical dataset and do not establish fairness, calibration, or suitability for actual lending decisions. Model probabilities are not calibrated lending probabilities or guarantees.

**Implementation status: completed under the six resolved owner decisions.** The original specification's readiness verdict preceded the implementation; see the [implementation report](inference-implementation-report.md) for its verification and remaining limitations. This document does not establish production readiness or authorize real-world lending decisions.
