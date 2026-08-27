# Breast Cancer Classification from Blood Biomarkers

End-to-end classical ML project: predicting breast cancer status from routine blood analysis and anthropometric measurements, instead of the more common cell-morphology (FNA) features.

Built as a guided learning project, inspired by [OscarW99's applied-ml-series](https://github.com/OscarW99/applied-ml-series/blob/main/ClassicalML1_2_EndToEnd_BreastCancerClassification.ipynb), adapted to a different dataset.

## Dataset

[Breast Cancer Coimbra](https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra) (Patrício et al., 2018, UCI ML Repository). 116 patients (64 with breast cancer, 52 healthy controls), 9 predictors: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP-1. No missing values.

## Motivation

<!-- TODO: fill in after finishing the project -->

## Results

<!-- TODO: fill in accuracy/AUC on the test set, confusion matrix, key findings -->

## Limitations

<!-- TODO: fill in — small n, single-center data, not clinically validated, etc. -->

## Reproducing

```bash
conda env create -f environment.yml
conda activate coimbra-cancer-ml
jupyter notebook notebooks/01_breast_cancer_coimbra_classification.ipynb
```
