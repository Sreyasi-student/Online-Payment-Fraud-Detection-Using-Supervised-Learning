# Health Insurance Claims Fraud Detection using Supervised Learning

A binary classification pipeline that flags fraudulent healthcare insurance claims, built on the Kaggle **[Healthcare Fraud Detection Dataset](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset)**.

## Project structure

```
.
├── Healthcare_Fraud_detection_code_fixed.ipynb   # main notebook (EDA → cleaning → modeling → evaluation)
├── Project_Report.docx                            # write-up of methodology, fixes, and results
└── README.md
```

## What the notebook does

1. **Load data** — downloads the dataset via `kagglehub`.
2. **Missing values** — inspects and quantifies nulls per column.
3. **Feature cleanup**
   - Drops `Claim_ID` and `Provider_ID` (unique identifiers with no generalizable predictive value).
   - Extracts `Claim_Submission_Month` from `Claim_Submission_Date`, then drops the raw date.
   - Casts `Procedure_Code` to string so it's one-hot encoded rather than scaled.
4. **Train/test split** — 70/30 stratified split performed **before** any encoding, scaling, or resampling, to prevent test-set leakage.
5. **Imputation** — median (numeric) / mode (categorical) imputation, fit and applied separately on train and test.
6. **Preprocessing** — `ColumnTransformer` with `OneHotEncoder` for categorical features and `StandardScaler` for numeric features.
7. **Class imbalance** — `SMOTE` oversampling, applied only to the training data (and inside each cross-validation fold during tuning, never touching validation folds or the test set).
8. **Model training** — Logistic Regression, Decision Tree, Random Forest, XGBoost, and KNN as baselines.
9. **Hyperparameter tuning** — `RandomizedSearchCV` (with SMOTE inside an `imblearn` pipeline) for KNN, Random Forest, and XGBoost.
10. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, and AUC-PR (the primary metric, since fraud is a rare-class problem) on the held-out test set, plus an ROC curve comparing the two strongest models.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
kagglehub
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost kagglehub
```

You'll also need a Kaggle account/API token configured for `kagglehub` to download the dataset (`~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

## Running

Open the notebook and run all cells top to bottom (**Runtime → Restart and run all** in Colab / Jupyter). The pipeline is fully self-contained after the dataset download step.

## Changelog — fixes applied

This version resolves four issues found in the original notebook:

| # | Issue | Fix |
|---|---|---|
| 1 | `Provider_ID` was left in the feature set despite the notebook's own text saying it would be dropped, risking leakage of provider-specific fraud history from train into test. | Dropped alongside `Claim_ID`. |
| 2 | `rf_auc` / `xgb_auc` were referenced in print statements and the ROC plot but never defined — `NameError` on a clean run. | Computed via `roc_auc_score(y_test, proba)` before use. |
| 3 | `num_features1 = num_features.remove('Is_Fraud')` set `num_features1` to `None` (`list.remove()` mutates in place and returns `None`). | Split into `.copy()` + `.remove()` on separate lines. |
| 4 | The `ColumnTransformer` silently re-derived categorical/numeric feature lists by dtype, overriding the deliberate categorical treatment of `Chronic_Condition_Flag` and `Claim_Submission_Month`. | Reuses the manually curated `cat_features` / `num_features1` lists instead. |

**Note:** because fix #1 removes a feature the models were previously trained on, metrics from a fresh run will differ from (and are more trustworthy than) any results generated before this fix.
