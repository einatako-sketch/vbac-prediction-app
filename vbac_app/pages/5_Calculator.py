"""
Page 5 — Individual Patient VBAC Calculator
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import ANTENATAL_VARS, INTRAPARTUM_VARS, CS_INDICATION_LABELS, START_MODE_LABELS

st.set_page_config(page_title="VBAC Calculator", page_icon="🧮", layout="wide")
st.title("Step 5 — Individual Patient VBAC Calculator")

st.markdown("""
Enter the patient's clinical data below to receive a VBAC probability estimate.

> **Disclaimer:** This tool is for research purposes only and does not replace clinical judgment.
""")

if not st.session_state.get("training_done"):
    st.warning("Please complete Step 3 (Train Models) first.")
    st.stop()

results = st.session_state["results"]
all_vars = {**ANTENATAL_VARS, **INTRAPARTUM_VARS}

# ── Model and algorithm selection ──────────────────────────────────────────────
st.subheader("Select Model")

available_models = list(results.keys())
selected_model = st.selectbox(
    "Choose model and algorithm",
    available_models,
    format_func=lambda k: f"Model {k.split('_', 1)[0]} ({['Antenatal only', 'Antenatal + Intrapartum'][k.startswith('B')]}) — {k.split('_', 1)[1]}"
)

model_letter = selected_model.split("_")[0]
is_model_B = model_letter == "B"

opt_thresh = results[selected_model]["opt_thresh"]
st.info(f"Optimal threshold (Youden's J): **{opt_thresh:.3f}** — scores above this predict VBAC success.")

st.markdown("---")

# ── Patient input form ─────────────────────────────────────────────────────────
st.subheader("Patient Data Entry")

patient_data = {}

# Antenatal inputs
st.markdown("#### Antenatal Variables")
col1, col2, col3 = st.columns(3)

col_idx = 0
ant_cols = [col1, col2, col3]

for var, meta in ANTENATAL_VARS.items():
    col = ant_cols[col_idx % 3]
    col_idx += 1

    with col:
        if meta["type"] == "numeric":
            val = st.number_input(
                meta["label"],
                value=None,
                placeholder="Leave blank if unknown",
                key=f"calc_{var}"
            )
            patient_data[var] = float(val) if val is not None else np.nan

        elif meta["type"] == "binary":
            val = st.selectbox(
                meta["label"],
                options=["Unknown", "No (0)", "Yes (1)"],
                key=f"calc_{var}"
            )
            patient_data[var] = {"Unknown": np.nan, "No (0)": 0.0, "Yes (1)": 1.0}[val]

        elif meta["type"] == "categorical":
            if var == "cs_indication":
                options = {"Unknown": np.nan, **{f"{v} — {l}": float(v) for v, l in CS_INDICATION_LABELS.items()}}
            elif var == "start_mode":
                options = {"Unknown": np.nan, **{f"{v} — {l}": float(v) for v, l in START_MODE_LABELS.items()}}
            else:
                options = {"Unknown": np.nan}

            sel = st.selectbox(meta["label"], list(options.keys()), key=f"calc_{var}")
            patient_data[var] = options[sel]

# Intrapartum inputs (Model B only)
if is_model_B:
    st.markdown("#### Intrapartum Variables")
    col1, col2, col3 = st.columns(3)
    inp_cols = [col1, col2, col3]
    col_idx = 0

    for var, meta in INTRAPARTUM_VARS.items():
        col = inp_cols[col_idx % 3]
        col_idx += 1

        with col:
            if meta["type"] == "numeric":
                val = st.number_input(
                    meta["label"],
                    value=None,
                    placeholder="Leave blank if unknown",
                    key=f"calc_{var}"
                )
                patient_data[var] = float(val) if val is not None else np.nan

            elif meta["type"] == "binary":
                val = st.selectbox(
                    meta["label"],
                    options=["Unknown", "No (0)", "Yes (1)"],
                    key=f"calc_{var}"
                )
                patient_data[var] = {"Unknown": np.nan, "No (0)": 0.0, "Yes (1)": 1.0}[val]

st.markdown("---")

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Calculate VBAC Probability", type="primary"):

    feature_list = (
        st.session_state.get("all_features", list(all_vars.keys()))
        if is_model_B
        else st.session_state.get("ant_features", list(ANTENATAL_VARS.keys()))
    )

    input_row = {f: patient_data.get(f, np.nan) for f in feature_list}
    X_input = pd.DataFrame([input_row])

    pipe = results[selected_model]["pipeline"]
    cal_model = results[selected_model].get("cal_model")
    raw_prob = pipe.predict_proba(X_input)[0, 1]
    if cal_model is not None:
        prob = cal_model.predict_proba(np.array([[raw_prob]]))[0, 1]
    else:
        prob = raw_prob
    prediction = "VBAC Success" if prob >= opt_thresh else "Repeat CS"

    # Display result
    st.markdown("---")
    st.subheader("Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("VBAC Probability", f"{prob:.1%}")
    col2.metric("Prediction", prediction)
    col3.metric("Threshold", f"{opt_thresh:.3f}")

    # Visual gauge
    fig, ax = plt.subplots(figsize=(8, 2.5))
    cmap = plt.get_cmap("RdYlGn")
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
    ax.axvline(opt_thresh, color="navy", lw=3, linestyle="--", label=f"Threshold ({opt_thresh:.2f})")
    color = "#155724" if prob >= opt_thresh else "#721c24"
    ax.axvline(prob, color=color, lw=4, label=f"Patient ({prob:.2f})")
    ax.text(prob, 0.5, f"  {prob:.1%}", va="center", ha="left" if prob < 0.8 else "right",
            fontsize=14, fontweight="bold", color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("VBAC Probability (calibrated)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("VBAC Probability Gauge", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Interpretation
    if prob >= 0.90:
        risk_level = "Very High"
        color_msg = "success"
        message = "This patient has a very high probability of VBAC success. TOLAC is strongly supported."
    elif prob >= 0.75:
        risk_level = "High"
        color_msg = "success"
        message = "This patient has a high probability of VBAC success."
    elif prob >= opt_thresh:
        risk_level = "Moderate-High"
        color_msg = "info"
        message = "This patient is above the optimal threshold — VBAC success is predicted, but with moderate confidence."
    elif prob >= 0.50:
        risk_level = "Moderate-Low"
        color_msg = "warning"
        message = "This patient is below the optimal threshold — repeat CS is predicted, but the probability is borderline."
    else:
        risk_level = "Low"
        color_msg = "error"
        message = "This patient has a low probability of VBAC success. Careful counseling is advised."

    if color_msg == "success":
        st.success(f"**{risk_level} VBAC probability.** {message}")
    elif color_msg == "info":
        st.info(f"**{risk_level} VBAC probability.** {message}")
    elif color_msg == "warning":
        st.warning(f"**{risk_level} VBAC probability.** {message}")
    else:
        st.error(f"**{risk_level} VBAC probability.** {message}")

    with st.expander("View entered patient data"):
        input_summary = {all_vars.get(k, {}).get("label", k): v for k, v in input_row.items()}
        st.dataframe(pd.DataFrame.from_dict(input_summary, orient="index", columns=["Value"]))

    st.markdown("---")
    st.caption("This prediction is based on a machine learning model trained on your institution's data. Probabilities are calibrated using Platt scaling. Results should be interpreted by qualified clinicians in the context of the full clinical picture.")
