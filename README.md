# Supervised Classification — Model Comparison and Evaluation

Rigorous comparison of three supervised classification models using repeated cross-validation and AUC-ROC as the primary metric, with a full evaluation pipeline on held-out test data.

## Context

MSc in Statistics & Data Science, University College Dublin — Statistical Machine Learning module. This project focuses on robust model selection methodology rather than any single algorithm, emphasising reproducible evaluation practices.

## What it does

1. **Data inspection** — class balance check; AUC-ROC chosen as primary metric to handle potential imbalance correctly. The positive class is set explicitly to ensure consistent two-class summary computation.
2. **Cross-validation framework** — repeated 5-fold CV (× 10 repeats = 50 resamples) via `caret`; all models evaluated under the same resampling scheme for fair comparison.
3. **Models trained and tuned**:
   - Logistic Regression (baseline)
   - SVM with RBF kernel (`kernlab`, cost and sigma tuned via grid search)
   - Random Forest (`randomForest`, mtry tuned)
4. **Model selection** — CV AUC-ROC distributions compared with boxplots; Random Forest selected as the best-performing model.
5. **Test evaluation** — final predictions (`y_hat`) and probability scores (`p_hat`) generated on the held-out test set.

## Tech stack

| Layer | Tools |
|---|---|
| Language | R |
| ML framework | `caret` |
| Algorithms | `randomForest`, `kernlab` |
| Evaluation | `ROCR`, `doParallel` |

## Key results

- Random Forest achieved the highest cross-validated AUC-ROC across all 50 resamples.
- Logistic Regression served as a strong interpretable baseline; SVM-RBF was competitive but more sensitive to kernel parameter choice.
- Parallel processing (`doParallel`) used to accelerate the 50-resample tuning loop.

## Data

The dataset (`classification.RData`) is not redistributed here. Place it in `data/` before running `src/classification.R`.
