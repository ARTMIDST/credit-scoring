# Saved artifacts

The local `models/` directory holds `random_forest_credit_model.pkl` and `standard_scaler.pkl`. Both are excluded from Git as generated artifacts. The selected copies came from the notebook's original save destination and record scikit-learn 1.9.1. They were verified to load in a working environment with scikit-learn 1.9.1.

The forest is the **baseline** 100-tree Random Forest, with `random_state=42`. It expects 16 encoded features in the same names and order used by the notebook. It was trained on unscaled features. The scaler was fitted on the training features for Logistic Regression; do not apply it to the saved forest merely because both artifacts are present.

Load only artifacts from a trusted source: pickle/joblib loading can execute code. To reproduce artifacts from scratch, follow the notebook instructions in the root README. Executing the notebook will overwrite local files in this directory.
