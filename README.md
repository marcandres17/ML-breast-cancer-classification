# Breast Cancer Classification from Blood Biomarkers

End-to-end classical ML project: predicting breast cancer status from routine blood analysis and anthropometric measurements, instead of the more common cell-morphology (FNA) features.

Built as a guided learning project, inspired by [OscarW99's applied-ml-series](https://github.com/OscarW99/applied-ml-series/blob/main/ClassicalML1_2_EndToEnd_BreastCancerClassification.ipynb), adapted to a different dataset.

## Dataset

[Breast Cancer Coimbra](https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra) (Patrício et al., 2018, UCI ML Repository). 116 patients (64 with breast cancer, 52 healthy controls), 9 predictors: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP-1. No missing values.

## Motivation

Most public breast cancer classification demos (including the notebook this project is based on) use cell-morphology features extracted from a fine-needle aspirate (FNA) — already an invasive diagnostic step. This project asks a different question: can routine bloodwork and anthropometric measurements (the kind collected in a standard checkup) carry a useful signal on their own? The Coimbra dataset's predictors — glucose, insulin, HOMA-IR, leptin, adiponectin, resistin, BMI, age — are tied to a real research thread on insulin resistance and adipokine signaling as breast cancer risk factors, which gives the project a genuine hypothesis to test rather than just fitting a model to whatever numbers are available. The goal here is methodology, not diagnosis: an honest walkthrough of an end-to-end classical ML pipeline (EDA → preprocessing → model comparison → tuning → evaluation), including where the data pushes back on convenient assumptions.

## Results

Three models were compared with 5-fold cross-validation (ROC AUC) on the training set (n=92):

| Model | CV AUC (mean ± std) |
|---|---|
| Logistic Regression | 0.855 ± 0.119 |
| Decision Tree | 0.623 ± 0.100 |
| Random Forest (default) | 0.828 ± 0.066 |

The unconstrained Decision Tree hit AUC=1.0 on the training set but collapsed under cross-validation — a textbook overfitting signature. Random Forest, tuned via `GridSearchCV` (`n_estimators=50`, `max_depth=5`, `max_features='sqrt'`, `min_samples_split=5`), improved to **CV AUC 0.848** and was selected as the final model.

**Test set (n=24, held out from the start):**
- Accuracy: **0.667**
- ROC AUC: **0.846** — nearly identical to the cross-validation estimate (0.848), suggesting the CV score wasn't overly optimistic.
- Confusion matrix: 9 true negatives, 2 false positives, 6 false negatives, 7 true positives.

**Feature importances** (from the tuned Random Forest): Glucose (0.26), Resistin (0.14), BMI (0.13), and Age (0.11) were the top drivers. Notably, Age and BMI ranked highly despite near-zero linear correlation with the target — the model is picking up non-linear relationships that a correlation heatmap alone would miss. HOMA and Insulin, despite stronger individual correlation with the target, ranked lower in importance — expected, since HOMA-IR is mathematically derived from glucose and insulin (`HOMA = Glucose × Insulin / 405`), so their signal partially overlaps.

## Limitations

- **Small sample size (n=116, test set n=24).** Every reported number here is noisy — a single misclassified test case shifts accuracy by ~4 percentage points. Results should be read as a methodology demonstration, not a validated clinical estimate.
- **Class imbalance in errors matters more than accuracy.** The model missed 6 of 13 actual cancer cases in the test set (false negatives). In a screening context, false negatives are the costlier error type — this model, as-is, is not tuned for that trade-off (e.g., via a lower decision threshold or a recall-oriented metric), which would be a natural next step.
- **Feature multicollinearity.** HOMA is a deterministic function of Glucose and Insulin, not an independent measurement — this likely understates their combined importance in the Random Forest's feature ranking.
- **Single-center data**, not externally validated — the model has not been tested on patients from a different hospital, population, or measurement protocol.
- **Not clinically validated.** This is a portfolio/methodology project, not a diagnostic tool.

## Reproducing

Requires Python 3.11+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, and `joblib`.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/breast_cancer_coimbra_classification.ipynb
```

Or, with conda:

```bash
conda env create -f environment.yml
conda activate coimbra-cancer-ml
jupyter notebook notebooks/breast_cancer_coimbra_classification.ipynb
```
