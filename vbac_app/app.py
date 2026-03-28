"""
VBAC Prediction App — Home Page
"""
import streamlit as st

st.set_page_config(
    page_title="VBAC Prediction ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("VBAC Prediction — Machine Learning Platform")
st.markdown("**Global Validation Initiative | Lis Maternity Hospital, Tel Aviv Sourasky Medical Center**")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome

    This platform allows any maternity hospital to:

    1. **Upload** their own delivery dataset (Excel or CSV)
    2. **Map** columns to standardized VBAC variables
    3. **Explore** the data with descriptive statistics
    4. **Train** two machine learning models:
       - **Model A** — Antenatal features only (available before labor)
       - **Model B** — Antenatal + Intrapartum features (available during labor)
    5. **Evaluate** model performance (AUC, sensitivity, specificity, PPV, NPV)
    6. **Calculate** individual patient VBAC probability

    ---

    ### Study Background

    This tool was developed as part of a single-center study at **Lis Maternity Hospital**
    (Tel Aviv Sourasky Medical Center, 2015–2024), IRB approval **0474-24**.

    Three algorithms are compared: **Logistic Regression**, **Random Forest**, and **Gradient Boosting**,
    with SMOTE oversampling for class imbalance and Youden's J threshold optimization.

    The Grobman MFMU nomogram is included as a benchmark.

    ---

    ### How to Use

    Use the **sidebar** to navigate through the steps in order:
    1. Upload & Map Data
    2. Exploratory Analysis
    3. Train Models
    4. View Results
    5. Patient Calculator
    """)

with col2:
    st.markdown("""
    ### Quick Stats (Lis Cohort)

    | | |
    |---|---|
    | Total deliveries | 1,627 |
    | VBAC success | 88.5% |
    | Repeat CS | 11.5% |
    | Study period | 2015–2024 |
    | Antenatal features | 12 |
    | Intrapartum features | 7 |

    ---

    ### Reference Paper

    Tako E, Attali E, et al.
    *Machine Learning for VBAC Prediction.*
    PAPER 2026 (submitted).

    ---

    ### Contact

    Dr. Einat Tako
    Lis Maternity Hospital
    Tel Aviv Sourasky Medical Center
    """)

st.markdown("---")
st.caption("Built with Streamlit · For research purposes only · Not a clinical decision-making tool")
