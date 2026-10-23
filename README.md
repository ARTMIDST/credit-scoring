# Credit Scoring Prediction

An exploratory machine learning study of credit risk classification using 1,000 applicant records. The project compares Logistic Regression, a Decision Tree, and a Random Forest on the same held-out test set. The saved baseline Random Forest was selected because it led the reported baseline comparison.

## Results

The original notebook encodes `good` risk as 1 and `bad` risk as 0. The following test-set metrics therefore use `good` as the positive class. The test set contains 200 records (140 good, 60 bad).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6750 | 0.7273 | 0.8571 | 0.7869 | 0.6602 |
| Decision Tree | 0.6600 | 0.7535 | 0.7643 | 0.7589 | 0.5905 |
| Baseline Random Forest | **0.7550** | **0.7898** | **0.8857** | **0.8350** | **0.7407** |
| Tuned Random Forest | 0.7400 | 0.7750 | 0.8857 | 0.8267 | 0.7344 |

The tuned forest was an experiment and did not improve the held-out accuracy or ROC-AUC. The [formal report](docs/Credit%20Scoring%20Report.docx) and [notebook](notebooks/01_dataset_exploration.ipynb) retain the analysis, plots, derived outputs, and conclusions. Saved row-level previews were removed pending verification of dataset redistribution terms.

## Dataset Source

This project uses the German Credit dataset from Kaggle's [German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) listing by Leonardo Ferreira (`kabure`). The dataset file remains local and excluded from Git; this repository makes no licence or redistribution claim.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/` | Dataset instructions; the CSV is kept local and excluded from Git |
| `docs/` | Original formal report and [project documentation](docs/project_documentation.md) |
| `models/` | Published baseline Random Forest and [artifact documentation](models/README.md); the scaler remains local and excluded from Git |
| `notebooks/` | Authoritative exploratory notebook |

## Use the saved forest

The root [inference module](inference.py) provides `predict_applicant(record)` and `predict_batch(records)` for single and batch predictions. It validates the nine raw fields, applies the fixed encoding in the [preprocessing specification](docs/inference-preprocessing-specification.md), and returns a `Bad` or `Good` prediction plus both labelled model probabilities. The [model documentation](models/README.md) records the artifact hash, tested environment, and loading precautions. The dataset CSV is not needed to use the saved forest; model outputs are not lending decisions or calibrated guarantees.

## Reproduce the analysis

An incomplete first download was replaced with a second Kaggle download for the completed analysis. The CSV is excluded from Git, so a separately obtained copy must be present at `data/german_credit_data.csv` before the notebook runs. [Dataset details](data/README.md) provide the expected columns and counts.

In Windows PowerShell, run the following from the repository root with Python 3.14 available. Use a **new** environment: the existing `.venv/` is incomplete and should not be used.

```powershell
python -m venv .venv-new
.\.venv-new\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install notebook ipykernel
python -m ipykernel install --sys-prefix --name credit-scoring --display-name "Python (credit-scoring)"
python -m notebook
```

In Jupyter, open `notebooks/01_dataset_exploration.ipynb` from the repository root and select the `Python (credit-scoring)` kernel. Alternatively, navigate to `notebooks/` in PowerShell after activation and launch Jupyter there. The notebook expects its own directory as the kernel working directory; its relative paths are `../data/german_credit_data.csv` and `../models/`.

Running the notebook will train models and overwrite local artifacts in `models/`. Review the preserved outputs first if you only need the accepted results. The repository does not need retraining to read the report or load existing local artifacts.

The pinned versions were observed in the available working Python environment; they are not a claim that the incomplete `.venv/` works. See [models/README.md](models/README.md) for loading the existing artifacts.

## Scope and limitations

The 1,000-record dataset is small, contains missing account values, and has a 70/30 target split. The reported metrics reflect one stratified 80/20 split; they do not establish performance on other populations or future lending conditions. The selected model identifies only 27 of 60 `bad` cases in the test set. Feature importances describe how the forest used its encoded inputs and **are not causal evidence**. No fairness, calibration, drift, or deployment validation is documented. This is an experimental study, not a system for real-world credit decisions.

The Kaggle listing is linked above. This repository does not publish the dataset or assert its licence or redistribution permissions.
