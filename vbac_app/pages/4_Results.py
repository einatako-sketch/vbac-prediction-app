"""
Page 4 — Results & Visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import ANTENATAL_VARS, INTRAPARTUM_VARS

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")
st.title("Step 4 — Results & Visualization")

if not st.session_state.get("training_done"):
    st.warning("Please complete Step 3 (Train Models) first.")
    st.stop()

results = st.session_state["results"]
y_test = st.session_state["y_test"]
grobman_auc = st.session_state.get("grobman_auc")
all_vars = {**ANTENATAL_VARS, **INTRAPARTUM_VARS}

# ── Performance table ──────────────────────────────────────────────────────────
st.subheader("Model Performance")

rows = []
for key, res in results.items():
    model, algo = key.split("_", 1)
    rows.append({
        "Model": f"Model {model}",
        "Algorithm": algo,
        "CV AUC (mean)": round(res["cv_auc_mean"], 3),
        "CV AUC (SD)": round(res["cv_auc_std"], 3),
        "Test AUC": round(res["test_auc"], 3),
        "Sensitivity": round(res["sensitivity"], 3),
        "Specificity": round(res["specificity"], 3),
        "PPV": round(res["ppv"], 3),
        "NPV": round(res["npv"], 3),
        "Threshold": round(res["opt_thresh"], 3),
    })

df_perf = pd.DataFrame(rows)
st.dataframe(df_perf, use_container_width=True, hide_index=True)

if grobman_auc is not None:
    st.info(f"Grobman MFMU Benchmark AUC (without race/ethnicity): **{grobman_auc:.3f}**")

# Download table
csv = df_perf.to_csv(index=False)
st.download_button("Download performance table (CSV)", csv, "vbac_model_performance.csv", "text/csv")

st.markdown("---")

# ── ROC Curves ────────────────────────────────────────────────────────────────
st.subheader("ROC Curves")

# Separate Model A and Model B
model_A_keys = [k for k in results if k.startswith("A_")]
model_B_keys = [k for k in results if k.startswith("B_")]

colors = {
    "Logistic Regression": "#3498db",
    "Random Forest": "#2ecc71",
    "Gradient Boosting": "#e67e22",
}

ncols = 1 + (1 if model_B_keys else 0)
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
if ncols == 1:
    axes = [axes]

for ax, (title, keys) in zip(axes, [("Model A — Antenatal", model_A_keys), ("Model B — Antenatal + Intrapartum", model_B_keys)]):
    if not keys:
        continue
    for key in keys:
        algo = key.split("_", 1)[1]
        res = results[key]
        auc = res["test_auc"]
        ax.plot(res["fpr"], res["tpr"],
                color=colors.get(algo, "gray"),
                lw=2,
                label=f"{algo} (AUC={auc:.3f})")

    if grobman_auc is not None:
        ax.axhline(y=grobman_auc, color="purple", linestyle=":", lw=1.5,
                   label=f"Grobman AUC≈{grobman_auc:.3f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Reference")
    ax.set_xlabel("1 − Specificity (FPR)", fontsize=11)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# ── Feature importance ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Feature Importance")

imp_keys = [k for k, r in results.items() if r.get("importance") is not None]
if imp_keys:
    selected_key = st.selectbox(
        "Select model/algorithm",
        imp_keys,
        format_func=lambda k: f"Model {k.split('_', 1)[0]} — {k.split('_', 1)[1]}"
    )

    imp = results[selected_key]["importance"].sort_values(ascending=True)
    labels = [all_vars.get(f, {}).get("label", f)[:35] for f in imp.index]

    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.4)))
    bars = ax.barh(labels, imp.values,
                   color=["#e74c3c" if v == imp.max() else "#3498db" for v in imp.values])
    ax.set_xlabel("Importance" if "feature_importances_" in str(type(imp)) else "Coefficient |value|")
    ax.set_title(f"Feature Importance — {selected_key.replace('_', ' — Model ', 1)}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
else:
    st.info("Feature importance not available for the selected algorithm.")

# ── Calibration plot ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Calibration Plot")

calib_key = st.selectbox(
    "Select model for calibration",
    list(results.keys()),
    format_func=lambda k: f"Model {k.split('_', 1)[0]} — {k.split('_', 1)[1]}",
    key="calib_select"
)

y_prob = results[calib_key]["y_prob"]

# Bin predictions
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
fraction_pos = []
mean_pred = []

for i in range(n_bins):
    mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
    if mask.sum() > 0:
        fraction_pos.append(y_test.values[mask].mean())
        mean_pred.append(y_prob[mask].mean())

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(mean_pred, fraction_pos, "o-", color="#e67e22", lw=2, markersize=7,
        label=f"Model {calib_key.split('_', 1)[0]} — {calib_key.split('_', 1)[1]}")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of VBAC successes")
ax.set_title("Calibration Plot", fontweight="bold")
ax.legend()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, use_container_width=False)
plt.close()

# ── Download all results ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Export")

if st.button("Generate Summary Report (CSV)"):
    all_rows = []
    for key, res in results.items():
        model, algo = key.split("_", 1)
        row = {
            "Model": f"Model {model}",
            "Algorithm": algo,
            "CV_AUC_mean": res["cv_auc_mean"],
            "CV_AUC_SD": res["cv_auc_std"],
            "Test_AUC": res["test_auc"],
            "Sensitivity": res["sensitivity"],
            "Specificity": res["specificity"],
            "PPV": res["ppv"],
            "NPV": res["npv"],
            "Optimal_threshold": res["opt_thresh"],
        }
        all_rows.append(row)

    report_df = pd.DataFrame(all_rows)
    st.download_button(
        "Download full report",
        report_df.to_csv(index=False),
        "vbac_results_report.csv",
        "text/csv"
    )

st.markdown("**Proceed to Step 5 (Calculator) to predict for individual patients using the sidebar.**")
