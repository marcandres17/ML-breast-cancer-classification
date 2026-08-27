# Guided ML Portfolio Project: Breast Cancer Coimbra Classification

## Goal

Adapt the structure of [OscarW99's end-to-end breast cancer classification notebook](https://github.com/OscarW99/applied-ml-series/blob/main/ClassicalML1_2_EndToEnd_BreastCancerClassification.ipynb) to a different dataset, as a portfolio piece. The user writes the code themselves; Claude acts as a guide, not an implementer — one step at a time, in chat, explaining the objective and relevant concepts without supplying the code.

## Dataset

[Breast Cancer Coimbra](https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra) (Patrício et al., 2018, UCI ML Repository, id 451). 116 patients (64 cancer, 52 healthy controls), 9 blood/anthropometric predictors (Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP-1), binary target, no missing values. Chosen over the original 30-feature morphology dataset for a distinct narrative (routine bloodwork vs. FNA imaging features) while staying in the same cancer-classification theme.

Deviations from the original notebook, and why:
- **UMAP → PCA 2D**: with only 9 features and 116 samples, UMAP's non-linear neighbor embedding is less appropriate/interpretable than in the original's 30-feature dataset; PCA is the more honest choice here and ties naturally into the scaling step.
- **cv=10 → cv=5** for cross-validation: with ~93 training samples, 10 folds leaves too few samples per fold; 5 folds is more stable.
- Imputation and one-hot-encoding sections stay **illustrative** (as in the original), since this dataset has no missing values or categorical features either.

## Repo structure

```
ML prediction cancer/
├── README.md
├── environment.yml            # conda env "coimbra-cancer-ml"
├── .gitignore
├── data/breast_cancer_coimbra.csv
├── notebooks/01_breast_cancer_coimbra_classification.ipynb
├── figures/                   # key exported plots for the README
└── models/                    # saved joblib pipeline
```

## Pedagogical approach

Step by step in chat. For each step: Claude states the objective and why it matters (no code, no exact function names unless asked), the user writes the code and shares it, Claude reviews/corrects/explains, then they move to the next step. This is a live collaborative session, not autonomous implementation — so the usual writing-plans/executing-plans skills don't apply here; this spec stands in as the roadmap instead.

## Roadmap (10 steps)

1. Load the dataset, inspect shape/dtypes
2. Initial EDA: histograms by class (9 features)
3. Stratified train/test split
4. Deeper EDA on train: PCA 2D, correlation heatmap, pairplot, boxplots
5. Preprocessing: illustrative imputation/encoding + real scaling
6. Train 3 models (Logistic Regression, Decision Tree, Random Forest) via `Pipeline` + cross-validation (`cv=5`)
7. Hyperparameter tuning with `GridSearchCV` on Random Forest
8. Feature importances
9. Final test-set evaluation (accuracy, AUC, confusion matrix, ROC) + interpret the small-n caveat
10. Save the model with `joblib` + fill in README results/limitations

## Success criteria

The project is done when: the notebook runs top to bottom, all 10 steps are implemented by the user and reviewed by Claude, the final model is saved to `models/`, and the README's Motivation/Results/Limitations sections are filled in.
