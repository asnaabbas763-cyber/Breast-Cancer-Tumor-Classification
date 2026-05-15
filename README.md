# Breast Cancer Tumor Classification Using Clinical Features

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Team Members & Course Details

| Detail | Info |
|---|---|
| **Course** | Machine Learning / Data Science |
| **Institution** | *Digital University Kerala* |
| **Team Members** | *Angel Treesa John, Nikhil, Asna Abbas* |
| **Student IDs** | *253127* |


---

## Problem Statement & Motivation

Breast cancer is one of the most prevalent and life-threatening cancers worldwide. Early and accurate detection of whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** is critical; it directly impacts treatment decisions and patient survival rates.

Traditional diagnostic methods rely heavily on expert radiologists and pathologists, which can be time-consuming and subject to human error. Machine learning offers a powerful, data-driven alternative that can assist clinicians in making faster and more reliable diagnoses.

**Goal:** Build and compare multiple machine learning models that classify breast tumors as malignant or benign using numerical clinical features derived from digitized images of cell nuclei - and deploy the best model as an interactive web application.

---

## Dataset Description

| Property | Details |
|---|---|
| **Name** | Wisconsin Breast Cancer (Diagnostic) Dataset (WBCD) |
| **Source** | UCI Machine Learning Repository |
| **Samples** | 568 |
| **Features** | 30 numerical features |
| **Target Variable** | `outcome` - M (Malignant = 1) / B (Benign = 0) |

### Feature Groups
Features are computed from digitized images of breast mass cell nuclei across three categories:

- **Mean values** - e.g., `radius_mean`, `texture_mean`, `area_mean`
- **Standard Error (SE)** - e.g., `radius_se`, `perimeter_se`
- **Worst values** - e.g., `radius_worst`, `concave_points_worst`

Key morphological features include: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

### Class Distribution

| Class | Label | Count (approx.) |
|---|---|---|
| Benign | 0 | ~357 (63%) |
| Malignant | 1 | ~212 (37%) |

> The dataset is slightly imbalanced but is suitable for classification tasks.

---

## Methodology Overview

The project follows a complete **ML lifecycle** across 8 stages:

### Stage 1 - Problem Definition & Literature Review
- Defined the classification objective: malignant vs. benign tumor detection
- Reviewed prior work on WBCD - models like LR, SVM, and Random Forest consistently achieve >90% accuracy

### Stage 2 - Data Collection & Understanding
- Loaded the WBCD dataset with 568 samples and 30 features
- Understood feature semantics: mean, SE, and worst measurements per nucleus

### Stage 3 - Data Preprocessing & Cleaning
- Removed the `id` column (non-predictive)
- Encoded the target variable: M → 1, B → 0
- Verified zero missing values and zero duplicate rows

### Stage 4 - Exploratory Data Analysis (EDA)
- Plotted class distribution via count plots
- Analyzed feature distributions using histograms
- Detected outliers using boxplots
- Performed correlation analysis (heatmap) - found strong correlation between radius, perimeter, and area
- Visualized class separation via scatter plots (`radius_mean` vs `area_mean`)

### Stage 5 - Feature Engineering & Scaling
- Applied **StandardScaler** for normalization (critical for LR and SVM)
- Used an 80/20 stratified train-test split preserving class ratios

### Stage 6 - Model Building & Training
Three models were trained and compared:
1. **Logistic Regression** - `max_iter=1000`, scaled features
2. **Support Vector Machine (SVM)** - RBF kernel, `probability=True`
3. **Random Forest** - `n_estimators=100`

All models and the scaler were saved as `.pkl` files using `joblib`.

### Stage 7 - Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Tools: Confusion Matrix, ROC Curves, 5-Fold Stratified Cross-Validation
- **Recall** was prioritized - a missed malignant case (False Negative) is far more dangerous than a False Positive in medical diagnosis

### Stage 8 - Model Interpretation & Explainability
- **Random Forest Feature Importances** - global feature ranking
- **Logistic Regression Coefficients** - direction of feature influence
- **Permutation Importance** - true impact on F1 score for all 3 models
- **SHAP (SHapley Additive exPlanations)** - summary plots and single-prediction waterfall charts

![ROC Curve](plots/ROC%20curve.png)

![Confusion Metrics](plots/confusion%20metrics.png)

---

## Results Summary

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9762** | **0.9762** | **0.9762** | **0.9974** |
| SVM (RBF Kernel) | **0.9912** | **1.0000** | **0.9762** | **0.9880** | **0.9967** |
| Random Forest | **0.9912** | **1.0000** | **0.9762** | **0.9880** | **0.9942** |

> *(Fill in your actual metric values from the notebook output)*

### Key Findings
- **Best Model by Accuracy & ROC-AUC:** Logistic Regression (ROC-AUC = **0.9974**)
- **Most Important Features** (consistent across all models):
  1. `concave_points_worst`
  2. `perimeter_worst`
  3. `radius_worst`
- **Cross-Validation:** 5-Fold Stratified CV confirms model stability
- In a medical context, **Recall** is the most critical metric — the model minimizes dangerous False Negatives

---

## Application Screenshots

> *(Add screenshots of your deployed Streamlit app here)*

| Input Form | Prediction Result |
|---|---|
| ![App Input](screenshots/app_input.png) | ![App Output](screenshots/app_output.png) |

---

## Live Deployment

**Streamlit App:** [Click here to use the app](https://your-streamlit-app-url.streamlit.app)

> *(Replace with your actual Streamlit deployment link)*





Key libraries used:
