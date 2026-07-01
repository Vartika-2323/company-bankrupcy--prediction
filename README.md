# company-bankrupcy--prediction
Validated the 1968 Altman Z-Score bankruptcy model against 6,819 modern companies and benchmarked it against a logistic regression challenger — AUC 0.91, with sensitivity and class-imbalance analysis.
# Company Bankruptcy Prediction — Altman Z-Score Validation & Benchmarking

Independent validation of the classic Altman Z-Score (1968) bankruptcy prediction model against modern real-world data, benchmarked against a logistic regression challenger.

## Overview
- **Dataset:** 6,819 companies, 96 financial variables, 220 bankruptcy cases (3.23% base rate)
- **Goal:** Test whether a 1968 financial distress model, built on 66 US manufacturing firms, still holds up on a modern, larger, more diverse dataset — and see if a data-driven model can beat it

## Approach
1. Replicated the Altman Z-Score formula (Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5) using 5 core financial ratios
2. Validated discriminatory power (AUC-ROC, Gini) and classification performance against actual bankruptcy outcomes
3. Ran sensitivity analysis: threshold optimization, coefficient stress testing, and variable omission testing
4. Built a Logistic Regression "challenger" model to benchmark against the Altman formula

## Key Results
| Model | AUC-ROC | Gini |
|---|---|---|
| Altman Z-Score (1968) | 0.910 | 0.819 |
| Logistic Regression (challenger) | 0.913 | 0.826 |

- **Net Income/Total Assets (X3)** is the most influential ratio — removing it drops AUC by 0.06
- The original 1968 distress threshold (1.81) is **miscalibrated for this dataset** — at that threshold, the model misses 100% of actual bankruptcies despite 96.8% overall accuracy, a classic class-imbalance trap
- An optimized threshold (3.80) improves real-world classification meaningfully
- The logistic regression challenger edges out Altman on AUC (+0.003) but sacrifices the interpretability that makes Z-Score usable in regulated credit contexts

## Key Insight
Overall accuracy is a misleading metric on imbalanced data (3.23% bankruptcy rate) — a model can score 96%+ accuracy while catching zero real bankruptcies. This project demonstrates why AUC, recall, and threshold calibration matter more than headline accuracy for risk models.

## Tools
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
