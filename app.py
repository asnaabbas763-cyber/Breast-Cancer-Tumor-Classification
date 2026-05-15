"""
Stage 9: Deployment — Breast Cancer Tumor Classifier
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Tumor Classifier",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #C0392B;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-malignant {
        background: linear-gradient(135deg, #FDECEA, #FADBD8);
        border-left: 6px solid #C0392B;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-benign {
        background: linear-gradient(135deg, #EAF7F0, #D5F5E3);
        border-left: 6px solid #1E8449;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .stButton>button {
        width: 100%;
        background-color: #C0392B;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.7rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #A93226;
    }
    .info-box {
        background: #EBF5FB;
        border: 1px solid #AED6F1;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.9rem;
        color: #1A5276;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        scaler   = joblib.load('scaler.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        svm_model= joblib.load('svm_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        return scaler, lr_model, svm_model, rf_model, True
    except Exception as e:
        return None, None, None, None, False

scaler, lr_model, svm_model, rf_model, models_loaded = load_models()

MODEL_MAP = {
    "Logistic Regression": lr_model,
    "SVM (RBF Kernel)":    svm_model,
    "Random Forest":       rf_model
}

FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean','concave_points_mean',
    'symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
    'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
    'concave_points_se','symmetry_se','fractal_dimension_se','radius_worst',
    'texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave_points_worst',
    'symmetry_worst','fractal_dimension_worst'
]

# Typical healthy ranges for slider defaults (WBCD dataset medians approx.)
FEATURE_DEFAULTS = {
    'radius_mean': (6.98, 28.11, 13.37),
    'texture_mean': (9.71, 39.28, 19.28),
    'perimeter_mean': (43.79, 188.5, 86.24),
    'area_mean': (143.5, 2501.0, 551.1),
    'smoothness_mean': (0.053, 0.163, 0.096),
    'compactness_mean': (0.019, 0.345, 0.104),
    'concavity_mean': (0.0, 0.427, 0.089),
    'concave_points_mean': (0.0, 0.201, 0.049),
    'symmetry_mean': (0.106, 0.304, 0.181),
    'fractal_dimension_mean': (0.05, 0.097, 0.063),
    'radius_se': (0.112, 2.873, 0.405),
    'texture_se': (0.360, 4.885, 1.217),
    'perimeter_se': (0.757, 21.98, 2.866),
    'area_se': (6.802, 542.2, 40.34),
    'smoothness_se': (0.002, 0.031, 0.007),
    'compactness_se': (0.002, 0.135, 0.025),
    'concavity_se': (0.0, 0.396, 0.032),
    'concave_points_se': (0.0, 0.053, 0.012),
    'symmetry_se': (0.008, 0.079, 0.020),
    'fractal_dimension_se': (0.001, 0.030, 0.004),
    'radius_worst': (7.93, 36.04, 16.27),
    'texture_worst': (12.02, 49.54, 25.68),
    'perimeter_worst': (50.41, 251.2, 107.26),
    'area_worst': (185.2, 4254.0, 880.6),
    'smoothness_worst': (0.071, 0.223, 0.132),
    'compactness_worst': (0.027, 1.058, 0.254),
    'concavity_worst': (0.0, 1.252, 0.272),
    'concave_points_worst': (0.0, 0.291, 0.115),
    'symmetry_worst': (0.157, 0.664, 0.290),
    'fractal_dimension_worst': (0.055, 0.208, 0.084)
}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🎗️ Breast Cancer Tumor Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Wisconsin Breast Cancer Dataset · Logistic Regression · SVM · Random Forest</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
⚕️ <b>Medical Disclaimer:</b> This tool is for <b>educational and research purposes only</b>.
It is <b>not</b> a substitute for professional medical diagnosis. Always consult a qualified clinician.
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Model files not found. Make sure `scaler.pkl`, `logistic_regression_model.pkl`, `svm_model.pkl`, and `random_forest_model.pkl` are in the same directory as `app.py`.")
    st.info("Run the training notebook first to generate the `.pkl` files, then place them alongside this app.")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — Model selection + CSV upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model_name = st.selectbox(
        "Choose Model",
        list(MODEL_MAP.keys()),
        help="Logistic Regression has the best ROC-AUC (0.9974) in our evaluation."
    )

    st.markdown("---")
    st.subheader("📂 Batch Prediction (CSV)")
    uploaded_file = st.file_uploader(
        "Upload a CSV with 30 features",
        type=["csv"],
        help="CSV must have exactly 30 numeric columns matching the WBCD feature order."
    )
    st.markdown("---")
    st.markdown("""
**Model Performance Summary**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Reg | 0.9825 | 0.9974 |
| SVM | 0.9912 | 0.9967 |
| Random Forest | 0.9912 | 0.9942 |

*Evaluated on 20% held-out test set.*
    """)

# ─────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Manual Input", "📂 Batch CSV", "📊 Model Info"])

# ══════════════════════════════════════════════
# TAB 1 — Manual input
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Enter Cell Nucleus Measurements")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("**📐 Mean Values** — average of cell nuclei")
    with col_info2:
        st.markdown("**📏 SE Values** — standard error of measurements")
    with col_info3:
        st.markdown("**⚠️ Worst Values** — largest/worst measurements")

    st.markdown("---")

    # Group features into Mean / SE / Worst
    groups = {
        "📐 Mean Values": [f for f in FEATURE_NAMES if f.endswith('_mean')],
        "📏 SE Values":   [f for f in FEATURE_NAMES if f.endswith('_se')],
        "⚠️ Worst Values":[f for f in FEATURE_NAMES if f.endswith('_worst')]
    }

    input_values = {}

    for group_name, features in groups.items():
        with st.expander(group_name, expanded=(group_name == "📐 Mean Values")):
            cols = st.columns(2)
            for i, feat in enumerate(features):
                mn, mx, default = FEATURE_DEFAULTS[feat]
                step = round((mx - mn) / 1000, 6)
                with cols[i % 2]:
                    input_values[feat] = st.number_input(
                        feat.replace('_', ' ').title(),
                        min_value=float(mn),
                        max_value=float(mx) * 1.5,
                        value=float(default),
                        step=float(step),
                        format="%.5f",
                        key=feat
                    )

    # ── Predict button ──
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button("🔍 Predict Tumor Type")

    if predict_btn:
        # Build feature vector
        input_array = np.array([[input_values[f] for f in FEATURE_NAMES]])
        input_scaled = scaler.transform(input_array)

        model = MODEL_MAP[selected_model_name]
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = probabilities[prediction] * 100

        st.markdown("---")
        st.subheader(f"🔎 Prediction Result — {selected_model_name}")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
<div class="result-malignant">
<h2>🔴 MALIGNANT</h2>
<p style="font-size:1.1rem;">The model predicts the tumor is <b>malignant (cancerous)</b>.</p>
<p><b>Confidence:</b> {confidence:.1f}%</p>
<p style="color:#922B21; font-weight:600;">⚠️ Please consult a medical professional immediately.</p>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-benign">
<h2>🟢 BENIGN</h2>
<p style="font-size:1.1rem;">The model predicts the tumor is <b>benign (non-cancerous)</b>.</p>
<p><b>Confidence:</b> {confidence:.1f}%</p>
<p style="color:#1E8449; font-weight:600;">✅ Regular monitoring is still recommended.</p>
</div>
""", unsafe_allow_html=True)

        with res_col2:
            # Probability bar chart
            fig, ax = plt.subplots(figsize=(4, 2.5))
            labels = ['Benign', 'Malignant']
            colors = ['#1E8449', '#C0392B']
            bars = ax.barh(labels, probabilities * 100, color=colors, height=0.5, edgecolor='white')
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{prob*100:.1f}%', va='center', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)", fontsize=10)
            ax.set_title("Class Probabilities", fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── All models comparison ──
        st.markdown("#### 🤖 All Models Comparison")
        comp_cols = st.columns(3)
        for i, (mname, mobj) in enumerate(MODEL_MAP.items()):
            pred_i = mobj.predict(input_scaled)[0]
            prob_i = mobj.predict_proba(input_scaled)[0]
            conf_i = prob_i[pred_i] * 100
            label  = "🔴 Malignant" if pred_i == 1 else "🟢 Benign"
            with comp_cols[i]:
                st.markdown(f"""
<div class="metric-card">
<p style="font-size:0.85rem; color:#666; margin-bottom:0.2rem;"><b>{mname}</b></p>
<p style="font-size:1.4rem; margin: 0.2rem 0;">{label}</p>
<p style="font-size:0.9rem; color:#333;">Conf: {conf_i:.1f}%</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — Batch CSV prediction
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Batch Prediction from CSV File")

    st.info("""
**CSV Format Requirements:**
- Must contain exactly **30 numeric columns** in the WBCD feature order.
- Column names should match: `radius_mean`, `texture_mean`, ..., `fractal_dimension_worst`
- No `id` or `outcome` column needed.
    """)

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)

            # Drop id/outcome if present
            for col in ['id', 'outcome']:
                if col in df_upload.columns:
                    df_upload.drop(col, axis=1, inplace=True)

            if df_upload.shape[1] != 30:
                st.error(f"Expected 30 feature columns, found {df_upload.shape[1]}. Please check your CSV.")
            else:
                # Rename if needed
                if list(df_upload.columns) != FEATURE_NAMES:
                    df_upload.columns = FEATURE_NAMES

                st.write(f"**Loaded {len(df_upload)} samples.** Preview:")
                st.dataframe(df_upload.head(), use_container_width=True)

                if st.button("🔍 Run Batch Prediction"):
                    X_scaled = scaler.transform(df_upload.values)
                    model = MODEL_MAP[selected_model_name]
                    preds = model.predict(X_scaled)
                    probs = model.predict_proba(X_scaled)

                    results_df = df_upload.copy()
                    results_df['Prediction'] = ['Malignant' if p == 1 else 'Benign' for p in preds]
                    results_df['Benign_Prob (%)']    = (probs[:, 0] * 100).round(2)
                    results_df['Malignant_Prob (%)'] = (probs[:, 1] * 100).round(2)
                    results_df['Confidence (%)']     = (np.max(probs, axis=1) * 100).round(2)

                    st.success(f"✅ Batch prediction complete using **{selected_model_name}**")

                    # Summary
                    n_mal = (preds == 1).sum()
                    n_ben = (preds == 0).sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Samples", len(preds))
                    c2.metric("🔴 Malignant", n_mal)
                    c3.metric("🟢 Benign", n_ben)

                    # Pie chart
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    ax2.pie([n_ben, n_mal], labels=['Benign', 'Malignant'],
                            colors=['#1E8449', '#C0392B'], autopct='%1.1f%%',
                            startangle=90, textprops={'fontsize': 11})
                    ax2.set_title("Prediction Distribution", fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    st.dataframe(results_df[['Prediction', 'Benign_Prob (%)', 'Malignant_Prob (%)', 'Confidence (%)']],
                                 use_container_width=True)

                    # Download
                    csv_out = results_df.to_csv(index=False)
                    st.download_button("⬇️ Download Results as CSV", csv_out,
                                       "breast_cancer_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
**To test batch prediction:**
1. Download the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from UCI.
2. Remove the `id` and `outcome` columns.
3. Upload the CSV here.
        """)

# ══════════════════════════════════════════════
# TAB 3 — Model info
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Model Performance & Dataset Information")

    st.markdown("#### Dataset: Wisconsin Breast Cancer (Diagnostic) Dataset")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Samples", "568")
    col_b.metric("Features", "30")
    col_c.metric("Benign", "357 (63%)")
    col_d.metric("Malignant", "212 (37%)")

    st.markdown("---")
    st.markdown("#### Model Comparison Table")

    perf_data = {
        "Model":     ["Logistic Regression", "SVM (RBF Kernel)", "Random Forest"],
        "Accuracy":  [0.9825, 0.9912, 0.9912],
        "Precision": [0.9762, 1.0000, 1.0000],
        "Recall":    [0.9762, 0.9762, 0.9762],
        "F1-Score":  [0.9762, 0.9880, 0.9880],
        "ROC-AUC":   [0.9974, 0.9967, 0.9942]
    }
    perf_df = pd.DataFrame(perf_data).set_index("Model")
    st.dataframe(perf_df.style.highlight_max(axis=0, color='#D5F5E3')
                              .format("{:.4f}"),
                 use_container_width=True)

    st.markdown("""
> ⚠️ **Why Recall matters most in medical diagnosis:**
> A **False Negative** (predicting Benign when the tumor is actually Malignant) is far more
> dangerous than a False Positive. Our models achieve **Recall = 0.9762** for malignant cases,
> meaning very few malignant tumors are missed.
    """)

    st.markdown("---")
    st.markdown("#### Feature Groups")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown("**Mean Values (10)**")
        st.code('\n'.join([f for f in FEATURE_NAMES if '_mean' in f]))
    with feat_col2:
        st.markdown("**SE Values (10)**")
        st.code('\n'.join([f for f in FEATURE_NAMES if '_se' in f]))
    with feat_col3:
        st.markdown("**Worst Values (10)**")
        st.code('\n'.join([f for f in FEATURE_NAMES if '_worst' in f]))

    st.markdown("---")
    st.markdown("""
#### Most Important Features (All Models Consistent)
1. `concave_points_worst`
2. `perimeter_worst`
3. `radius_worst`
4. `concave_points_mean`
5. `area_worst`

These **worst-value** morphological features consistently emerge as the strongest
predictors of malignancy across Logistic Regression, SVM, and Random Forest.
    """)

    st.markdown("---")
    st.markdown("#### About This Project")
    st.markdown("""
**Breast Cancer Tumor Classification Using Clinical Features**

- **Institution:** Digital University Kerala — School of Digital Science
- **Team:** Angel Treesa John, Nikhil, Asna Abbas
- **Course:** Predictive Analytics (2025–26)
- **Dataset:** [UCI ML Repository — WBCD](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Pipeline:** Problem Definition → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Explainability (SHAP) → Deployment
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🎗️ Breast Cancer Tumor Classifier · Digital University Kerala · Predictive Analytics 2025–26 · "
    "Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)