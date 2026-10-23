# Project documentation

## Purpose and source material

This study predicts the `Risk` category for 1,000 credit applicants. The authoritative project records are the [original notebook](../notebooks/01_dataset_exploration.ipynb), with its analysis and saved plots, and the [formal report](Credit%20Scoring%20Report.docx). The dataset was downloaded from Kaggle; an incomplete first download was replaced with a second Kaggle download used for the completed analysis. The exact listing, original attribution, licence, and redistribution terms remain unverified, so the CSV stays local and excluded from Git.

## Dataset and preparation

The source CSV has an unnamed export index, nine input fields, and the `Risk` target. `Saving accounts` has 183 missing values and `Checking account` has 394. The notebook drops the export index, fills those missing account values with the explicit `Unknown` category, and retains plausible outliers. It finds no duplicate rows.

The notebook maps `good` to 1 and `bad` to 0. It maps `Sex` and the account categories to numbers, then one-hot encodes `Housing` and `Purpose` with `drop_first=True`, producing 16 model features. It makes a stratified 80/20 train/test split with `random_state=42`, yielding 800 training and 200 test records. `StandardScaler` is fitted on the training split and used for Logistic Regression. The Decision Tree and Random Forest use unscaled features.

## Models and evaluation

Logistic Regression (`max_iter=1000`, `random_state=42`), a Decision Tree (`random_state=42`), and a baseline Random Forest (100 estimators, `random_state=42`) were compared on the held-out test set. Accuracy, precision, recall, F1, confusion matrices, classification reports, and ROC-AUC were reported. In the reported precision, recall, and F1 scores, `good` is the positive class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6750 | 0.7273 | 0.8571 | 0.7869 | 0.6602 |
| Decision Tree | 0.6600 | 0.7535 | 0.7643 | 0.7589 | 0.5905 |
| Baseline Random Forest | 0.7550 | 0.7898 | 0.8857 | 0.8350 | 0.7407 |
| Tuned Random Forest | 0.7400 | 0.7750 | 0.8857 | 0.8267 | 0.7344 |

The baseline forest's confusion matrix, with rows and columns ordered `bad`, `good`, is `[[27, 33], [16, 124]]`. Thus, it correctly identifies 27 of 60 `bad` cases. The baseline forest achieved the strongest held-out results in the original model comparison and remains the selected final model.

The notebook also runs five-fold `GridSearchCV` on the training set, scoring accuracy over estimator counts 100/200, depths `None`/10/20, minimum split sizes 2/5, and minimum leaf sizes 1/2. The best training cross-validation score is 0.7450 with 200 estimators, depth 10, split size 2, and leaf size 1. Its held-out accuracy and ROC-AUC are lower than the baseline's, so tuning did not change the selection.

## Interpretation and limits

The notebook ranks `Credit amount`, `Age`, and `Duration` among the leading baseline forest feature importances. These are impurity-based model scores, **not causal effects** or a justification for individual lending decisions.

The data is small and favors `good` records 700 to 300. The evaluation uses a single held-out split and does not establish how performance changes across time, institutions, or applicant groups. The `bad` class detection rate is limited; false approvals could matter in a real credit setting. The report does not document fairness assessment, probability calibration, monitoring, or external validation. Use the work as an educational experiment, not as an automated real-world credit decision system.

## Reproduction and artifacts

Use a fresh Python 3.14 virtual environment and install the direct dependencies pinned in `requirements.txt`; the root README gives Windows PowerShell commands for Jupyter and kernel setup. Obtain the dataset separately from its verified source and place it at `data/german_credit_data.csv`. Open the notebook so its working directory is `notebooks/` and its `../data/` and `../models/` paths resolve. The notebook exports the selected baseline forest and fitted scaler to root `models/`. Running it retrains and overwrites local artifacts; the accepted results and outputs are already preserved in the notebook and report.

The local artifact pair records scikit-learn 1.9.1 and was verified against the earlier copy: model parameters, feature count and names, feature importances, predictions and probabilities, scaler attributes, and selected test metrics match. Only the selected pair is retained. Artifact files and the CSV remain excluded from Git pending any separate decision about distribution.
