# Modeling Life Expectancy Using Health and Economic Indicators

Multiple linear regression (OLS), Ridge, Lasso, and Principal Component Regression (PCR) models built on the **Life Expectancy WHO Updated** dataset to explain and predict life expectancy from health and economic indicators.

## Contents

- `Life_Expectancy_Regression_Final.ipynb` — the analysis notebook (data loading, EDA, preprocessing, modeling, diagnostics, model comparison)

## Objectives

1. Load and inspect the data.
2. Perform a train/test split.
3. Fit multiple linear regression (OLS).
4. Check the major assumptions of linear regression.
5. Fit Ridge and Lasso regression.
6. Select the Ridge/Lasso regularization parameter (`alpha`) using cross-validation on the training set.
7. Compare OLS, Ridge, and Lasso (plus PCR) on the same untouched test set.

## Dataset

- Source file: `Life-Expectancy-Data-Updated.csv` (referred to in the notebook as the "clean Life Expectancy WHO Updated dataset")
- Original shape: 2,864 rows × 21 columns, with no missing values
- The notebook restricts the analysis to a single year, **`Year == 2003`**, giving 179 countries across 9 regions
- Columns: `Country`, `Region`, `Year`, `Infant_deaths`, `Under_five_deaths`, `Adult_mortality`, `Alcohol_consumption`, `Hepatitis_B`, `Measles`, `BMI`, `Polio`, `Diphtheria`, `Incidents_HIV`, `GDP_per_capita`, `Population_mln`, `Thinness_ten_nineteen_years`, `Thinness_five_nine_years`, `Schooling`, `Economy_status_Developed`, `Economy_status_Developing`, `Life_expectancy`

### Target
`Life_expectancy`

### Feature engineering
- `Country` is dropped (one-hot encoding ~200 countries would effectively turn the model into a country-fixed-effects model); `Region` is kept as the categorical geographic variable.
- `Economy_status_Developed` and `Economy_status_Developing` are collapsed into a single binary column, `Economy_status` (1 = Developing, 0 = Developed), to avoid perfect linear dependence with the intercept.
- `Under_five_deaths` is dropped due to high correlation with `Infant_deaths`, which is retained.

### Final predictors (14 numeric + 1 categorical)
Numeric: `Infant_deaths`, `Alcohol_consumption`, `Hepatitis_B`, `Measles`, `BMI`, `Polio`, `Diphtheria`, `Incidents_HIV`, `GDP_per_capita`, `Population_mln`, `Thinness_ten_nineteen_years`, `Thinness_five_nine_years`, `Schooling`, `Economy_status`
Categorical: `Region` (one-hot encoded, first category dropped)

## Methodology

1. **Train/test split** — 80/20 (`test_size=0.20`, `random_state=42`), yielding 143 training and 36 test observations. The test set is untouched until final evaluation.
2. **Preprocessing pipeline** (`ColumnTransformer`) — `StandardScaler` on numeric features, `OneHotEncoder(drop="first")` on `Region`; fit on the training data only.
3. **OLS baseline** — `LinearRegression` on the preprocessed features.
4. **Assumption checks on the OLS model** — linearity (residuals vs. fitted), normality (histogram, Q-Q plot, Shapiro-Wilk), homoskedasticity (Breusch-Pagan test), independence (Durbin-Watson), multicollinearity (VIF), and influential points (Cook's distance).
5. **Principal Component Regression (PCR)** — PCA fit on the training design matrix only, retaining components that explain 95% of training variance, followed by `LinearRegression` on the retained components.
6. **Ridge and Lasso** — `alpha` tuned via `GridSearchCV` over `np.logspace(-4, 4, 100)` using `KFold(n_splits=3, shuffle=True, random_state=42)` and `neg_root_mean_squared_error` scoring.
7. **Model comparison** — OLS, Ridge, Lasso, and PCR evaluated on the same held-out test set using R², RMSE, and MAE.

## Results

| Model | Test R² | Test RMSE | Test MAE | Tuned alpha |
|---|---|---|---|---|
| Ridge | 0.9260 | 2.6533 | 2.1618 | 3.3516 |
| OLS   | 0.9253 | 2.6654 | 2.0682 | — |
| Lasso | 0.9212 | 2.7377 | 2.1813 | 0.5214 |
| PCR   | 0.8798 | 3.3804 | 2.8571 | — (11 components, 96.02% variance retained) |

**Best model by test RMSE: Ridge.**

### OLS assumption-check summary
| Check | Result |
|---|---|
| Linearity | Assessed visually via residuals-vs-fitted plot |
| Normality (Shapiro-Wilk) | statistic 0.9816, p = 0.0511 → do not reject normality at 5% |
| Homoskedasticity (Breusch-Pagan) | LM stat 30.52, p = 0.1063 → no significant evidence of heteroskedasticity |
| Independence (Durbin-Watson) | 1.6965 → no substantial autocorrelation |
| Multicollinearity (VIF) | `Thinness_five_nine_years` (26.48), `Thinness_ten_nineteen_years` (24.18), `Polio` (15.24), `Diphtheria` (13.63) exceed 10 |
| Influential points (Cook's distance) | max 277.51; 10 observations above the 4/n threshold |

### Lasso variable selection
Of 22 encoded coefficients, Lasso shrinks 19 to exactly zero, retaining only `Infant_deaths`, `Incidents_HIV`, and `GDP_per_capita`.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
statsmodels
```

## How to run

1. Place `Life-Expectancy-Data-Updated.csv` in the path referenced by the notebook (`/content/...` if using Google Colab, or update `file_path` for a local run).
2. Run the notebook cells in order — the pipeline fits all preprocessing, PCA, and regularization hyperparameters on the training split only, so cells must execute sequentially.

## Notes / Limitations

- No missing-value imputation is performed; the source file has no missing values.
- The analysis in this notebook uses only the 2003 cross-section of the panel dataset (179 countries), not the full multi-year panel.
- Several features (`Polio`/`Diphtheria`, and the two thinness measures) show high VIF, indicating multicollinearity in the OLS coefficients; Ridge and Lasso are included in part to address this.
- Cook's distance flags 10 training observations as influential; these were not removed or investigated further within the notebook.
