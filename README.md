# Online Payment Fraud Detection using Supervised Learning

This project explores exploratory data analysis (EDA) and supervised machine learning
for detecting fraudulent online payment transactions on a tabular, severely imbalanced
dataset (\~5% fraud rate).

## Files

* `Fraud\_code\_final.ipynb` — Full workflow notebook: data loading, EDA, preprocessing,
model training, hyperparameter tuning, and evaluation.
* `Fraud Detection Dataset.csv` — Dataset used for analysis (expected in the same directory).

> \*\*Note:\*\* the trained model pickle files are not included in this repository due to size.

## Workflow

* **EDA** on 51,000 transactions using pandas, matplotlib, and seaborn to inspect
distributions, correlations, and class imbalance.
* **Preprocessing** — missing values imputed (median for numeric, mode for categorical),
categorical features one-hot encoded, numeric features standardized.
* **Leakage-safe split** — the train/test split is performed *before* any encoding,
scaling, or resampling. The preprocessor is fit on the training fold only and applied
(not re-fit) to the test fold, so no information from the test set leaks into training.
* **Class imbalance handling** — SMOTE is applied to the *training* fold only, after the
split. The test set is left untouched, real, and imbalanced, so evaluation reflects
real-world performance rather than inflated, leakage-driven metrics.
* **Models compared** — Logistic Regression, Decision Tree, K-Nearest Neighbors, Random
Forest, and XGBoost.
* **Hyperparameter tuning** — Random Forest and XGBoost tuned via `RandomizedSearchCV`
(3-fold cross-validation) over depth, estimator count, feature sampling, learning rate,
and subsampling ratios.
* **Evaluation** — Accuracy, Precision, Recall, F1-score, and ROC-AUC, computed on the
held-out, untouched test set.

## Results

|Model|Accuracy|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|-|
|Logistic Regression|56.3%|5.2%|45.3%|0.68|0.511|
|Decision Tree|89.2%|6.8%|9.4%|0.90|0.514|
|K-Nearest Neighbors|75.4%|4.8%|21.0%|0.82|0.496|
|Random Forest (tuned)|95.0%|41.4%|3.2%|0.06|0.535|
|**XGBoost (tuned)**|95.1%|100%\*|0.9%|0.02|0.500|

*\*XGBoost's near-perfect precision reflects that it predicts almost no transactions as
fraudulent, rather than genuine discriminative strength.*

**Best model by ROC-AUC: Random Forest (tuned), AUC = 0.535.**

### Honest takeaway

After correcting a data-leakage issue in the original pipeline (SMOTE and preprocessing
were previously fit on the full dataset before splitting, which artificially inflated
test-set performance), the corrected, leakage-free results show that none of the models
achieve strong separation between fraud and non-fraud transactions — ROC-AUC values
cluster close to 0.50 (equivalent to random guessing). High accuracy figures (\~95%) are
a byproduct of the \~95% non-fraud base rate, not genuine predictive skill, and recall on
the minority (fraud) class remains very low across all models.

This suggests the available features in this dataset may not carry enough signal to
reliably separate fraudulent from legitimate transactions, and is flagged here as an
explicit finding rather than presented as a high-performing model. Potential next steps
include richer feature engineering (e.g. transaction velocity, device/location
consistency checks), alternative resampling strategies, or acquiring additional
behavioral features.

