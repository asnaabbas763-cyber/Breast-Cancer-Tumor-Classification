# Breast Cancer Tumor Classification

This repository contains a breast cancer classification project using the Wisconsin Breast Cancer Diagnostic dataset. The project includes two main Jupyter notebooks for data exploration, model training, evaluation, explainability, and deployment preparation.

## Repository Structure

```text
Breast Cancer Wisconsin (Diagnostic) -wbc (1) (1) (1).csv
README.md
Requirements.txt
app.py
Problem Definition & Literature Review _ Data Collection & Data Understanding _ Data Preprocessing & Cleaning _ Exploratory Data Analysis.ipynb
Stage_5_6_7_8_Feature_Engineering_Model_Building_Evaluation_Explainability.ipynb
Models/
plots/
INDIVIDUAL PROFILES/
```

## Notebooks

- `Problem Definition & Literature Review _ Data Collection & Data Understanding _ Data Preprocessing & Cleaning _ Exploratory Data Analysis.ipynb`
  - EDA, preprocessing, feature analysis, and dataset understanding.
- `Stage_5_6_7_8_Feature_Engineering_Model_Building_Evaluation_Explainability.ipynb`
  - Model training, evaluation, explainability, and saving model artifacts.

## Dataset

The dataset file is included in the repository root:

```text
Breast Cancer Wisconsin (Diagnostic) -wbc (1) (1) (1).csv
```

### Important dataset note

The CSV file does not include header names, so the notebook loads it using `header=None` and explicitly sets the column names. The target column is mapped from:

- `M` → `1` (Malignant)
- `B` → `0` (Benign)

The notebooks also drop the non-predictive `id` column before training.

## Installation

1. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```

## Running the Notebooks

1. Open the notebook files in VS Code or Jupyter.
2. Run the cells from top to bottom.
3. Make sure the dataset file is in the project root.

### Notes for `Stage_5_6_7_8_Feature_Engineering_Model_Building_Evaluation_Explainability.ipynb`

- The notebook now loads the CSV with explicit column names.
- The `outcome` target column is created by mapping `diagnosis` values.
- A local guard is included for the Colab download block, so the notebook works in non-Colab environments.

## Model Artifacts

After running the training notebook, the following files are generated:

- `logistic_regression_model.pkl`
- `svm_model.pkl`
- `random_forest_model.pkl`
- `scaler.pkl`

These files are saved locally in the notebook folder.

## Key Findings

- The project compares Logistic Regression, SVM (RBF), and Random Forest.
- The model evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves, and stratified cross-validation.
- Model explainability uses feature importances, logistic regression coefficients, permutation importance, and SHAP.

## How to Use

1. Run the first notebook to explore and preprocess the data.
2. Run the second notebook to train the models and evaluate them.
3. Optionally use `app.py` to build a Streamlit or Flask-based interface.

## Recommended Python Version

- Python 3.8 or newer

## Contact

For questions or improvements, review the notebook outputs and adjust the preprocessing, model parameters, or explainability analysis.

### Key libraries used

- `pandas`
 - `numpy`
 - `scikit-learn`
 - `matplotlib`
 - `shap`
 - `joblib`
