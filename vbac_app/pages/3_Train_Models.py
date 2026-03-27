"""
Page 3 — Train Models
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import train_and_evaluate, ANTENATAL_VARS, INTRAPARTUM_VARS

st.set_page_config(page_title="Train Models", page_icon="🤖", layout="wide")
st.title("Step 3 — Train Machine Learning Models")

if not st.session_state.get("mapping_done"):
    st.warning("Please complete Step 1 (Upload & Map) first.")
    st.stop()

df_ant = st.session_state["df_ant"]
df_modelB = st.session_state["df_modelB"]
y = st.session_state["y"]

# ── Configuration ──────────────────────────────────────────────────────────────
st.subheader("Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    st.caption(f"Training: {int((1-test_size)*len(y)):,} | Test: {int(test_size*len(y)):,}")

with col2:
    algorithms = st.multiselect(
        "Algorithms to train",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

with col3:
    use_smote = st.checkbox("Use SMOTE oversampling", value=True,
                            help="Recommended when outcome is imbalanced (>80% majority class)")
    random_seed = st.number_input("Random seed", value=42, min_value=0)

st.markdown("---")

# ── Model selection ────────────────────────────────────────────────────────────
st.subheader("Models to Train")

train_A = st.checkbox("Model A — Antenatal only (15 features)", value=True)
train_B = st.checkbox("Model B — Antenatal + Intrapartum (20 features)", value=True)

ant_features = list(ANTENATAL_VARS.keys())
inp_features = list(INTRAPARTUM_VARS.keys())

# Only use features that exist in the data
ant_features = [f for f in ant_features if f in df_ant.columns]
all_features = [f for f in ant_features + inp_features if f in df_modelB.columns]

st.markdown(f"- **Model A** will use: {len(ant_features)} antenatal features")
st.markdown(f"- **Model B** will use: {len(all_features)} total features")

st.markdown("---")

# ── Grobman benchmark ──────────────────────────────────────────────────────────
with st.expander("Grobman MFMU Nomogram Benchmark (optional)"):
    st.markdown("""
    The Grobman nomogram uses: maternal age, BMI, prior vaginal delivery, prior VBAC, and indication for prior CS.
    Race/ethnicity coefficients are excluded (not applicable outside the US).

    If your dataset contains these variables (already mapped above), the benchmark will be computed automatically.
    """)
    run_grobman = st.checkbox("Include Grobman benchmark", value=True)

st.markdown("---")

# ── Train button ───────────────────────────────────────────────────────────────
if not algorithms:
    st.warning("Select at least one algorithm.")
    st.stop()

if not (train_A or train_B):
    st.warning("Select at least one model (A or B).")
    st.stop()

if st.button("Start Training", type="primary"):

    results = {}

    # Train/test split
    X_A = df_ant[ant_features]
    X_B = df_modelB[all_features]

    X_A_train, X_A_test, y_train, y_test = train_test_split(
        X_A, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    X_B_train, X_B_test, _, _ = train_test_split(
        X_B, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    total_runs = (len(algorithms) if train_A else 0) + (len(algorithms) if train_B else 0)
    progress = st.progress(0)
    status = st.empty()
    run = 0

    for algo in algorithms:
        if train_A:
            status.text(f"Training Model A — {algo}...")
            res = train_and_evaluate(X_A_train, X_A_test, y_train, y_test, algo, ant_features)
            results[f"A_{algo}"] = res
            run += 1
            progress.progress(run / total_runs)

        if train_B:
            status.text(f"Training Model B — {algo}...")
            res = train_and_evaluate(X_B_train, X_B_test, y_train, y_test, algo, all_features)
            results[f"B_{algo}"] = res
            run += 1
            progress.progress(run / total_runs)

    # Grobman benchmark
    if run_grobman:
        status.text("Computing Grobman benchmark...")
        try:
            from utils import compute_grobman
            grobman_auc = compute_grobman(df_ant, y)
            st.session_state["grobman_auc"] = grobman_auc
        except Exception:
            st.session_state["grobman_auc"] = None

    status.text("Training complete!")
    progress.progress(1.0)

    # Store results
    st.session_state["results"] = results
    st.session_state["y_test"] = y_test
    st.session_state["ant_features"] = ant_features
    st.session_state["all_features"] = all_features
    st.session_state["training_done"] = True

    st.success("All models trained successfully!")

    # Quick summary table
    st.markdown("### Quick Results Summary")
    summary_rows = []
    for key, res in results.items():
        model, algo = key.split("_", 1)
        summary_rows.append({
            "Model": f"Model {model}",
            "Algorithm": algo,
            "CV AUC (mean ± SD)": f"{res['cv_auc_mean']:.3f} ± {res['cv_auc_std']:.3f}",
            "Test AUC": f"{res['test_auc']:.3f}",
            "Sensitivity": f"{res['sensitivity']:.3f}",
            "Specificity": f"{res['specificity']:.3f}",
            "PPV": f"{res['ppv']:.3f}",
            "NPV": f"{res['npv']:.3f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("**Proceed to Step 4 (Results) for detailed plots using the sidebar.**")

elif st.session_state.get("training_done"):
    st.info("Models already trained. Navigate to Step 4 for results, or re-run training with different settings.")

    summary_rows = []
    for key, res in st.session_state["results"].items():
        model, algo = key.split("_", 1)
        summary_rows.append({
            "Model": f"Model {model}",
            "Algorithm": algo,
            "CV AUC": f"{res['cv_auc_mean']:.3f} ± {res['cv_auc_std']:.3f}",
            "Test AUC": f"{res['test_auc']:.3f}",
            "Sensitivity": f"{res['sensitivity']:.3f}",
            "Specificity": f"{res['specificity']:.3f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
