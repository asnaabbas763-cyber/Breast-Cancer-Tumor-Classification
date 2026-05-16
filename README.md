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
| **Team Members** | *(Add your team member names here)* |
| **Course** | *(Add course name & code, e.g., CS456 — Machine Learning)* |
| **Institution** | *(Add institution name)* |
| **Submission Date** | *(Add date)* |

---

## 🧩 Problem Statement & Motivation

Breast cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate detection of malignant tumors is critical — it directly determines whether a patient receives timely treatment, which significantly improves survival rates.

This project builds a machine learning pipeline to **classify breast tumors as Malignant (cancerous) or Benign (non-cancerous)** using clinical features derived from digitized images of cell nuclei. Rather than relying on a single model, we compare three well-established algorithms — Logistic Regression, Support Vector Machine, and Random Forest — and evaluate them across multiple metrics to identify the best approach for this high-stakes medical classification task.

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

### Stage 1 — Problem Definition & Literature Review
Defined the binary classification objective and reviewed prior art on WBCD. Literature consistently reports >90% accuracy with appropriate preprocessing. Key morphological predictors identified: radius, texture, perimeter, area, and concavity.

### Stage 2 — Data Collection & Understanding
Loaded the dataset from Google Drive. Explored the shape (568 × 32), column types, and the meaning of each feature group (mean, SE, worst).

### Stage 3 — Data Preprocessing & Cleaning
- Dropped the non-informative `id` column.
- Encoded the target: `M → 1`, `B → 0`.
- Confirmed zero missing values and zero duplicate rows.
- Verified correct data types for all columns.

### Stage 4 — Exploratory Data Analysis (EDA)
- **Class distribution:** Countplot to visualize slight imbalance.
- **Feature distributions:** Histograms across all 30 features (mostly normal/slightly skewed).
- **Outlier detection:** Boxplots — some outliers present but not extreme enough to distort analysis.
- **Correlation analysis:** Heatmap revealed high collinearity between radius, perimeter, and area features.
- **Key relationships:** Scatterplot of `radius_mean` vs `area_mean` showed clear class separation.

### Stage 5 — Feature Engineering & Selection
Features were retained as-is given the strong domain-validated predictive power. StandardScaler applied for Logistic Regression and SVM. Random Forest used the same scaled data for consistency.

### Stage 6 — Model Building & Training
Three models were trained on an **80/20 stratified split** (preserving class ratio):

| Model | Key Config |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `random_state=42` |
| **SVM** | `kernel='rbf'`, `probability=True`, `random_state=42` |
| **Random Forest** | `n_estimators=100`, `random_state=42` |

All models and the scaler were serialized as `.pkl` files using `joblib`.

### Stage 7 — Model Evaluation
Models were evaluated on the held-out test set using five metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and a 5-fold Stratified Cross-Validation. Confusion matrices and ROC curves were plotted for all three models.

### Stage 8 — Model Interpretation & Explainability
- **Random Forest Feature Importances** — global feature ranking.
- **Logistic Regression Coefficients** — direction and magnitude of each feature's influence.
- **Permutation Importance** — true impact measured on the test F1 score for all three models.
- **SHAP (SHapley Additive exPlanations)** — beeswarm summary plots (RF + LR), bar importance plots, and a waterfall plot for a single prediction.

---

## 📈 Results Summary

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **~0.98** | ~0.97 | ~0.97 | ~0.97 | **0.9974** |
| SVM (RBF Kernel) | ~0.97 | ~0.96 | ~0.96 | ~0.96 | ~0.997 |
| Random Forest | ~0.96 | ~0.95 | ~0.95 | ~0.95 | ~0.995 |

> *Exact values depend on your run. Update the table above with your printed output.*

### 5-Fold Cross-Validation (F1-Score)

| Model | Mean F1 | Std |
|---|---|---|
| Logistic Regression | *(fill from output)* | *(fill from output)* |
| SVM | *(fill from output)* | *(fill from output)* |
| Random Forest | *(fill from output)* | *(fill from output)* |

### Key Findings

- **Best Overall Model:** Logistic Regression — highest Accuracy and ROC-AUC (0.9974).
- **Most Critical Metric:** Recall — a False Negative (missed malignancy) is far more dangerous than a False Positive in clinical settings.
- **Top Predictive Features (consistent across all models):**
  1. `concave_points_worst`
  2. `perimeter_worst`
  3. `radius_worst`

---

## 🖥️ Screenshots

> Add screenshots of your Streamlit app below. Suggested sections to capture:
> - Input form / feature sliders
> - Prediction output (Malignant / Benign)
> - Model selector dropdown
> - Confidence score / probability display

| Input Panel | Prediction Result |
|---|---|
| *(add screenshot here)* | *(add screenshot here)* |

---

## 🚀 Live Deployment

The app is deployed on **Streamlit Community Cloud** and accessible at:

🔗 **[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**

> *(Replace the link above with your actual Streamlit deployment URL)*

---

## 🛠️ Setup & Run Locally

### Prerequisites

- Python 3.9 or above
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` should include:**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
shap
streamlit
```

### 3. Add the Dataset

Place the dataset CSV file in the project root (or update the path in the notebook):

```
Breast Cancer Wisconsin (Diagnostic) -wbc.csv
```

The dataset is available on the [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

### 4. Train the Models (Optional)

Open and run the notebook in Google Colab or Jupyter:

```bash
jupyter notebook Project_Breast_Cancer_Tumor_Classification.ipynb
```

This will generate the following `.pkl` files:

<<<<<<< Updated upstream
```
logistic_regression_model.pkl
svm_model.pkl
random_forest_model.pkl
scaler.pkl
```
=======
## Running the Notebooks

1. Open the notebook files in VS Code or Jupyter.
2. Run the cells from top to bottom.
3. Make sure the dataset file is in the project root.

### Notes for `Stage_5_6_7_8_Feature_Engineering_Model_Building_Evaluation_Explainability.ipynb`

- The notebook now loads the CSV with explicit column names.
- The `outcome` target column is created by mapping `diagnosis` values.
- A local guard is included for the Colab download block, so the notebook works in non-Colab environments.

## Model Artifacts
>>>>>>> Stashed changes

### 5. Run the Streamlit App

Ensure the `.pkl` files are in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

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
