# Model artifact publication audit (historical)

This document records the artifact audit before the approved inference implementation and publication decision. Its technical observations and audit-time limitations are retained as historical findings. The later decision superseded the recommendation to keep both artifacts excluded: only `models/random_forest_credit_model.pkl` is approved for publication. `models/standard_scaler.pkl` and `data/german_credit_data.csv` remain excluded. See the completed [inference implementation report](inference-implementation-report.md) and current [model README](../models/README.md) for the subsequent outcome.

## Decision

**Audit-time decision: conditional publication only.** Both saved objects loaded and could perform inference without the original CSV when given the exact 16 encoded, unscaled features in the trained order. At the time of this audit, the repository did **not** provide a reliable standalone path from a new applicant's raw fields to that input. Publishing these files as ready-to-use credit-scoring artifacts would have overstated their usability. The audit recommended keeping them ignored until the project owner reviewed the risks and approved an explicit inference interface and publication scope. That recommendation was superseded by the later, narrower approval described above.

This audit read the two local artifacts and notebook source, loaded the artifacts with the available Python interpreter, and used one synthetic all-zero *encoded* row for a mechanical API check. It did not read applicant-level CSV rows, execute the notebook, retrain, or save an artifact.

## Inventory and integrity

| File | Size | SHA-256 |
| --- | ---: | --- |
| `models/random_forest_credit_model.pkl` | 3,097,641 bytes | `9c1260524fb72ed568092eee8b3ad7daad89b5c91e2d700da470643d3e7263a2` |
| `models/standard_scaler.pkl` | 1,479 bytes | `114c4d27ac3f3da013fb2f5507775996e1e55555be5aacd341efdd8f93fb02bf` |

Both files existed and retained these hashes after loading. They are joblib/pickle artifacts with a protocol 5 prefix and serialized scikit-learn `1.9.1` metadata. The notebook's export cell writes a baseline forest and fitted scaler with `joblib.dump` to these filenames.

## Load and dependency compatibility

The available working interpreter is Python 3.14.7 at the system Python installation. The project's existing `.venv/` lacks `pyvenv.cfg` and `Scripts/python.exe`, so it is **not** the environment used for this check. Installed versions were scikit-learn 1.9.1, joblib 1.5.3, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, and threadpoolctl 3.7.0. The direct versions in `requirements.txt` agree with this working environment; that file does not lock every transitive dependency. The artifacts do not establish the exact NumPy or SciPy versions used when they were created.

Both `joblib.load` calls succeeded, with **zero** `InconsistentVersionWarning` instances. Loading produced 404 NumPy 2.5 `DeprecationWarning` instances about setting array shape. This is a compatibility warning to investigate before claiming support across environments; it did not prevent loading or the smoke test here. An in-memory synthetic DataFrame with the stored 16 column names produced a prediction of shape `(1,)` and probabilities of shape `(1, 2)`; probabilities were finite and summed to one. The scaler transformed the same synthetic schema to shape `(1, 16)`. This proves API operation for an already encoded input in this environment, not raw-applicant correctness or production reliability.

scikit-learn states that joblib is pickle based, loading can execute arbitrary code, and cross-version model loading is unsupported. Consumers should load only trusted, hash-verified copies in a compatible environment. See [scikit-learn's model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

## Random Forest artifact

Class: `sklearn.ensemble.RandomForestClassifier`. It is the accepted **baseline** forest, not the tuned GridSearchCV estimator. Fitted attributes include 100 trees, `n_features_in_ = 16`, `classes_ = [0, 1]`, `n_classes_ = 2`, and `n_outputs_ = 1`. The notebook maps `bad` to `0` and `good` to `1`; `predict_proba` columns therefore follow `[bad, good]` in that order. Its accepted held-out results remain 0.7550 accuracy, 0.8350 F1, and 0.7407 ROC-AUC. These are historical notebook results; this audit did not recalculate them from the dataset.

Complete `get_params(deep=False)` values:

```text
bootstrap=True; ccp_alpha=0.0; class_weight=None; criterion='gini'
max_depth=None; max_features='sqrt'; max_leaf_nodes=None; max_samples=None
min_impurity_decrease=0.0; min_samples_leaf=1; min_samples_split=2
min_weight_fraction_leaf=0.0; monotonic_cst=None; n_estimators=100
n_jobs=None; oob_score=False; random_state=42; verbose=0; warm_start=False
```

The model's stored feature names, in exact order, are:

```text
1  Age                         9  Housing_rent
2  Sex                        10  Purpose_car
3  Job                        11  Purpose_domestic appliances
4  Saving accounts            12  Purpose_education
5  Checking account           13  Purpose_furniture/equipment
6  Credit amount              14  Purpose_radio/TV
7  Duration                   15  Purpose_repairs
8  Housing_own                16  Purpose_vacation/others
```

The three largest stored impurity-based feature importances are `Credit amount` (0.24127612), `Age` (0.17760338), and `Duration` (0.16128245). They are descriptions of the fitted model, **not causal evidence** or a basis for individual credit decisions.

## StandardScaler artifact and its relationship to the forest

Class: `sklearn.preprocessing.StandardScaler`; parameters are `copy=True`, `with_mean=True`, and `with_std=True`. It is fitted on 800 training rows, has `n_features_in_ = 16`, and stores the same ordered `feature_names_in_` as the forest. Fitted arrays `mean_`, `var_`, and `scale_` each have 16 values:

| Feature | `mean_` | `var_` | `scale_` |
| --- | ---: | ---: | ---: |
| Age | 35.386250 | 129.292061 | 11.370667 |
| Sex | 0.687500 | 0.214844 | 0.463512 |
| Job | 1.905000 | 0.420975 | 0.648826 |
| Saving accounts | 1.177500 | 0.908494 | 0.953149 |
| Checking account | 0.975000 | 0.909375 | 0.953612 |
| Credit amount | 3185.841250 | 7668851.278548 | 2769.269087 |
| Duration | 20.827500 | 147.540244 | 12.146614 |
| Housing_own | 0.723750 | 0.199936 | 0.447142 |
| Housing_rent | 0.173750 | 0.143561 | 0.378894 |
| Purpose_car | 0.333750 | 0.222361 | 0.471552 |
| Purpose_domestic appliances | 0.010000 | 0.009900 | 0.099499 |
| Purpose_education | 0.055000 | 0.051975 | 0.227980 |
| Purpose_furniture/equipment | 0.187500 | 0.152344 | 0.390312 |
| Purpose_radio/TV | 0.281250 | 0.202148 | 0.449609 |
| Purpose_repairs | 0.025000 | 0.024375 | 0.156125 |
| Purpose_vacation/others | 0.010000 | 0.009900 | 0.099499 |

The notebook fits this scaler on `X_train`, uses it for the Logistic Regression path, and trains the Random Forest on **unscaled** `X_train`. The scaler must **not** be applied before the saved forest. No Logistic Regression artifact is present, so publishing the scaler alone does not supply a second usable classifier. It may be retained only with clear workflow context.

## New-applicant inference readiness

The observations in this section describe the repository **at audit time**, before `inference.py` was implemented. The current implementation is described in the [inference implementation report](inference-implementation-report.md).

The notebook supplies enough information to *design* a preprocessing function, but the exported objects do not include one. The relevant notebook steps are: drop the exported index column; replace missing `Saving accounts` and `Checking account` with `Unknown`; encode `Sex` as female `0` and male `1`; encode savings as `Unknown=0`, `little=1`, `moderate=2`, `quite rich=3`, `rich=4`; encode checking as `Unknown=0`, `little=1`, `moderate=2`, `rich=3`; and one-hot encode `Housing` and `Purpose` with `drop_first=True` and integer output. In the observed categories, `free` is the omitted housing category and `business` is the omitted purpose category. `Risk` is the target and must not be an inference input.

This is notebook logic applied to the full analysis DataFrame, not a fitted or fixed-schema transformer. Calling `pd.get_dummies(..., drop_first=True)` on one applicant or a batch with different categories can omit expected columns or change which category is dropped. `.map` returns missing values for unexpected category strings. At audit time, no code validated required fields, numeric types or ranges, unknown categories, absent values beyond the two account fields, or the exact 16-column order. There was no repository Python module, CLI, API, fitted encoder, or serialized `Pipeline`/`ColumnTransformer` for raw input. A model-only consumer could not reliably turn raw applicant fields into the saved forest's expected input without implementing and validating these steps. The original CSV is **not** needed at prediction time once a correct, fixed preprocessing contract exists; that contract has since been implemented in `inference.py`.

The model can mechanically predict from a correctly named and ordered 16-feature row without the CSV or retraining. That limited fact must not be presented as validated real-world credit inference. The dataset is small, the positive class is `good`, and the original work does not establish fairness, calibration, drift monitoring, or performance in other applicant populations.

## Publication control and recommendation

At audit time, `.gitignore` included `/models/**` and `!/models/README.md`, so Git ignored both `.pkl` files. The audit's earlier conditional recommendation contemplated exceptions for both artifacts. The later approved publication scope is narrower and supersedes that recommendation. The current exception structure is:

```gitignore
/models/**
!/models/README.md
!/models/random_forest_credit_model.pkl
```

Only the accepted baseline forest is approved for publication; the scaler and CSV remain excluded. The approved inference module supplies the fixed raw-input contract, subject to the limitations in the [implementation report](inference-implementation-report.md) and [model README](../models/README.md). Do not broadly unignore `models/` or present model outputs as validated real-world lending decisions. Dataset redistribution terms remain unverified. No ignore rule, artifact, or Git index entry was changed **during this historical audit**.
