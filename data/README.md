# Dataset

The dataset was downloaded from Kaggle. The first download was incomplete, so a replacement was downloaded from Kaggle and used for the completed analysis. The available project files do not identify the exact Kaggle listing or its licence and redistribution terms. The CSV remains excluded from Git; do not upload or commit it until the exact listing and permissions are verified.

To reproduce the notebook, obtain the dataset separately from its verified source and place it at `data/german_credit_data.csv`. The required structure and counts below can help check that a candidate file matches the local analysis copy; they do not establish its source or licence.

The replacement copy used for the accepted notebook outputs has 1,000 rows. Its columns are an unnamed sequential export index, `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`, and `Risk`. The notebook drops the export index before analysis. `Risk` contains 700 `good` and 300 `bad` records. The CSV has 183 missing `Saving accounts` values and 394 missing `Checking account` values.

The notebook expects to run from `notebooks/` and reads `../data/german_credit_data.csv`. See [project documentation](../docs/project_documentation.md) for preprocessing details.
