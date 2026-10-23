# Dataset

Source: the German Credit dataset from Kaggle's [German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) listing by Leonardo Ferreira (`kabure`). The first download was incomplete; a replacement Kaggle download was used for the completed analysis. The CSV remains excluded from Git. This repository does not assert a dataset licence or redistribution permission.

For local reproduction, place a separately obtained copy at `data/german_credit_data.csv`. The structure and counts below describe the copy used in the analysis.

The replacement copy used for the accepted notebook outputs has 1,000 rows. Its columns are an unnamed sequential export index, `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`, and `Risk`. The notebook drops the export index before analysis. `Risk` contains 700 `good` and 300 `bad` records. The CSV has 183 missing `Saving accounts` values and 394 missing `Checking account` values.

The notebook expects to run from `notebooks/` and reads `../data/german_credit_data.csv`. See [project documentation](../docs/project_documentation.md) for preprocessing details.
