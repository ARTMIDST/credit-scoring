# Published baseline Random Forest

`random_forest_credit_model.pkl` is the accepted baseline 100-tree `RandomForestClassifier` (`random_state=42`). It was trained on the accepted 800-row training split using 16 encoded, **unscaled** features in a fixed order. The model's classes are `0 = Bad credit risk` and `1 = Good credit risk`.

| Held-out test metric | Accepted result |
| --- | ---: |
| Accuracy | 0.7550 |
| Precision | 0.7898 |
| Recall | 0.8857 |
| F1 | 0.8350 |
| ROC-AUC | 0.7407 |

The artifact is 3,097,641 bytes. Its SHA-256 is `9c1260524fb72ed568092eee8b3ad7daad89b5c91e2d700da470643d3e7263a2`. Verify this hash before loading a downloaded copy. The [artifact audit](../docs/model-artifact-publication-audit.md) records the fitted parameters, feature names, and compatibility check.

The artifact records scikit-learn 1.9.1 and was tested with Python 3.14.7, scikit-learn 1.9.1, joblib 1.5.3, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, and threadpoolctl 3.7.0. `requirements.txt` pins the direct project dependencies; it does not lock every transitive dependency. Other version combinations have not been validated. The audit observed NumPy 2.5 deprecation warnings during artifact loading, although loading and synthetic inference succeeded in the tested environment.

Use the root [inference module](../inference.py): `predict_applicant(record)` accepts one raw applicant mapping, and `predict_batch(records)` accepts a non-empty sequence of mappings. Each returns the predicted `Bad` or `Good` class and both labelled probabilities. The module validates and encodes inputs to the exact 16-feature order documented in the [preprocessing specification](../docs/inference-preprocessing-specification.md). The original dataset is not needed for inference with the published forest. The probabilities are model outputs, not calibrated lending probabilities or guarantees.

`standard_scaler.pkl` remains local and excluded from Git. It belongs to the separate Logistic Regression workflow and **must not** be applied to the Random Forest. Loading pickle/joblib files can execute arbitrary code: load only a trusted, hash-verified copy in a compatible environment. This artifact is for educational and portfolio demonstration, **not** automated lending decisions or a validated production credit decision system. Executing the notebook would retrain and overwrite local artifacts; it is not required to use the saved forest.
