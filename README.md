# 🎗️ Breast Cancer Tumor Classification Using Clinical Features

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👥 Team Members & Course Details

| Field | Details |
|---|---|
| **Project Title** | Breast Cancer Tumor Classification Using Clinical Features |
| **Team Members** | Angel Treesa John, Asna Abbas, Nikhil AS |
| **Course** | Predictive Analytics |
| **Institution** | Digital University kerala |
| **Submission Date** | 16-05-2026 |

---

## 🧩 Problem Statement & Motivation

Breast cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate detection of malignant tumors is critical - it directly determines whether a patient receives timely treatment, which significantly improves survival rates.

This project builds a machine learning pipeline to **classify breast tumors as Malignant (cancerous) or Benign (non-cancerous)** using clinical features derived from digitized images of cell nuclei. Rather than relying on a single model, we compare three well-established algorithms - Logistic Regression, Support Vector Machine, and Random Forest - and evaluate them across multiple metrics to identify the best approach for this high-stakes medical classification task.

The motivation is two-fold:
- **Clinical impact:** Assist medical practitioners with a data-driven second opinion tool.
- **Research benchmark:** The Wisconsin Breast Cancer Dataset is a canonical benchmark, making results directly comparable with existing literature.

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Name** | Wisconsin Breast Cancer (Diagnostic) Dataset (WBCD) |
| **Source** | UCI Machine Learning Repository / Google Drive |
| **Samples** | 568 rows (one per tumor biopsy) |
| **Features** | 30 numerical features (+ ID + target) |
| **Target Variable** | `outcome` — `M` = Malignant (1), `B` = Benign (0) |

### Feature Groups

Features are computed from digitized images of fine needle aspirate (FNA) of breast masses. For each of 10 base measurements, three statistics are recorded:

| Group | Description |
|---|---|
| **Mean values** | `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, ... |
| **Standard Error (SE)** | `radius_se`, `texture_se`, `perimeter_se`, `area_se`, ... |
| **Worst values** | `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, ... |

### Class Distribution

| Class | Label | Count (approx.) |
|---|---|---|
| Benign | 0 | ~357 (63%) |
| Malignant | 1 | ~212 (37%) |

> The dataset is slightly imbalanced but suitable for classification without resampling. Recall is prioritized as the primary metric given the medical context.

---

## 🔬 Methodology Overview

The project follows a complete Machine Learning lifecycle across **8 stages**:

### Stage 1 - Problem Definition & Literature Review
Defined the binary classification objective and reviewed prior art on WBCD. Literature consistently reports >90% accuracy with appropriate preprocessing. Key morphological predictors identified: radius, texture, perimeter, area, and concavity.

### Stage 2 - Data Collection & Understanding
Loaded the dataset from Google Drive. Explored the shape (568 × 32), column types, and the meaning of each feature group (mean, SE, worst).

### Stage 3 - Data Preprocessing & Cleaning
- Dropped the non-informative `id` column.
- Encoded the target: `M → 1`, `B → 0`.
- Confirmed zero missing values and zero duplicate rows.
- Verified correct data types for all columns.

### Stage 4 - Exploratory Data Analysis (EDA)
- **Class distribution:** Countplot to visualize slight imbalance.
- **Feature distributions:** Histograms across all 30 features (mostly normal/slightly skewed).
- **Outlier detection:** Boxplots — some outliers present but not extreme enough to distort analysis.
- **Correlation analysis:** Heatmap revealed high collinearity between radius, perimeter, and area features.
- **Key relationships:** Scatterplot of `radius_mean` vs `area_mean` showed clear class separation.

### Stage 5 - Feature Engineering & Selection
Features were retained as-is given the strong domain-validated predictive power. StandardScaler applied for Logistic Regression and SVM. Random Forest used the same scaled data for consistency.

### Stage 6 - Model Building & Training
Three models were trained on an **80/20 stratified split** (preserving class ratio):

| Model | Key Config |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `random_state=42` |
| **SVM** | `kernel='rbf'`, `probability=True`, `random_state=42` |
| **Random Forest** | `n_estimators=100`, `random_state=42` |

All models and the scaler were serialized as `.pkl` files using `joblib`.

### Stage 7 - Model Evaluation
Models were evaluated on the held-out test set using five metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and a 5-fold Stratified Cross-Validation. Confusion matrices and ROC curves were plotted for all three models.

## ROC Curve
![ROC Curve](roc_curve.png)

## Confusion Matrix
![Confusion Matrix](confusion_metrics.png)

### Stage 8 - Model Interpretation & Explainability
- **Random Forest Feature Importances** - global feature ranking.
- **Logistic Regression Coefficients** - direction and magnitude of each feature's influence.
- **Permutation Importance** - true impact measured on the test F1 score for all three models.
- **SHAP (SHapley Additive exPlanations)** - beeswarm summary plots (RF + LR), bar importance plots, and a waterfall plot for a single prediction.

---

## 📈 Results Summary

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.98** | 0.97 | 0.97 | 0.97 | **0.9974** |
| SVM (RBF Kernel) | 0.97 | 0.96 | 0.96 | 0.96 | 0.997 |
| Random Forest | 0.96 | 0.95 | 0.95 | 0.95 | 0.995 |

> These results are based on the latest model evaluation.

### 5-Fold Cross-Validation (F1-Score)

| Model | Mean F1 | Std |
|---|---|---|
| Logistic Regression | 0.97 | 0.01 |
| SVM | 0.96 | 0.01 |
| Random Forest | 0.95 | 0.01 |

### Key Findings

- **Best Overall Model:** Logistic Regression - highest Accuracy and ROC-AUC (0.9974).
- **Most Critical Metric:** Recall - a False Negative (missed malignancy) is far more dangerous than a False Positive in clinical settings.
- **Top Predictive Features (consistent across all models):**
  1. `concave_points_worst`
  2. `perimeter_worst`
  3. `radius_worst`

---

## 🖥️ Screenshots

> Add screenshots of your Streamlit app below. Suggested sections to capture:
> - Input form / feature sliders
> - Prediction output 

---

## 🚀 Live Deployment

The app is deployed on **Streamlit Community Cloud** and accessible at:

🔗 **[https://your-app-name.streamlit.app](https://breast-cancer-tumor-classification-5psqkbblq5szdblfeuw4uu.streamlit.app/)**

> *(Replace the link above with your actual Streamlit deployment URL)*

---

## 📁 Project Structure

```
breast-cancer-classification/
│
├── Project_Breast_Cancer_Tumor_Classification.ipynb   # Full ML pipeline notebook
├── app.py                                             # Streamlit deployment app
├── requirements.txt                                   # Python dependencies
│
├── logistic_regression_model.pkl                      # Trained LR model
├── svm_model.pkl                                      # Trained SVM model
├── random_forest_model.pkl                            # Trained RF model
├── scaler.pkl                                         # Fitted StandardScaler
│
└── README.md                                          # Project documentation
```

---
