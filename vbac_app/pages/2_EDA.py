"""
Page 2 — Exploratory Data Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import ANTENATAL_VARS, INTRAPARTUM_VARS

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("Step 2 — Exploratory Data Analysis")

if not st.session_state.get("mapping_done"):
    st.warning("Please complete Step 1 (Upload & Map) first.")
    st.stop()

df_modelB = st.session_state["df_modelB"]
y = st.session_state["y"]
df_ant = st.session_state["df_ant"]

all_vars = {**ANTENATAL_VARS, **INTRAPARTUM_VARS}

# ── Overview ───────────────────────────────────────────────────────────────────
st.subheader("Dataset Overview")

n_total = len(y)
n_vbac = int(y.sum())
n_cs = n_total - n_vbac

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total cases", f"{n_total:,}")
col2.metric("VBAC success", f"{n_vbac:,} ({100*n_vbac/n_total:.1f}%)")
col3.metric("Repeat CS", f"{n_cs:,} ({100*n_cs/n_total:.1f}%)")
col4.metric("Features available", int(df_modelB.notna().any().sum()))

# ── Outcome distribution ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Outcome Distribution")

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["VBAC Success", "Repeat CS"], [n_vbac, n_cs],
       color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
ax.set_ylabel("Count")
for i, v in enumerate([n_vbac, n_cs]):
    ax.text(i, v + 5, f"{v:,}\n({100*v/n_total:.1f}%)", ha="center", fontsize=9)
ax.set_ylim(0, max(n_vbac, n_cs) * 1.2)
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, use_container_width=False)
plt.close()

# ── Table 1 ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Table 1 — Baseline Characteristics by Outcome")

df_combined = df_modelB.copy()
df_combined["outcome"] = y.values

rows = []
for var, meta in all_vars.items():
    if var not in df_combined.columns:
        continue
    col_data = df_combined[var]
    vbac_data = df_combined.loc[df_combined["outcome"] == 1, var].dropna()
    cs_data = df_combined.loc[df_combined["outcome"] == 0, var].dropna()

    if meta["type"] in ["numeric"]:
        vbac_num = pd.to_numeric(vbac_data, errors='coerce').dropna()
        cs_num = pd.to_numeric(cs_data, errors='coerce').dropna()
        rows.append({
            "Variable": meta["label"],
            f"VBAC (n={n_vbac:,})": f"{vbac_num.median():.1f} [{vbac_num.quantile(0.25):.1f}–{vbac_num.quantile(0.75):.1f}]" if len(vbac_num) > 0 else "N/A",
            f"Repeat CS (n={n_cs:,})": f"{cs_num.median():.1f} [{cs_num.quantile(0.25):.1f}–{cs_num.quantile(0.75):.1f}]" if len(cs_num) > 0 else "N/A",
            "Type": "Median [IQR]"
        })
    elif meta["type"] in ["binary", "categorical"]:
        # Show most common value or yes %
        if meta["type"] == "binary":
            vbac_num = pd.to_numeric(vbac_data, errors='coerce').dropna()
            cs_num = pd.to_numeric(cs_data, errors='coerce').dropna()
            v_pct = float(vbac_num.mean()) * 100 if len(vbac_num) > 0 else np.nan
            c_pct = float(cs_num.mean()) * 100 if len(cs_num) > 0 else np.nan
            rows.append({
                "Variable": meta["label"],
                f"VBAC (n={n_vbac:,})": f"{v_pct:.1f}%" if not np.isnan(v_pct) else "N/A",
                f"Repeat CS (n={n_cs:,})": f"{c_pct:.1f}%" if not np.isnan(c_pct) else "N/A",
                "Type": "% Yes"
            })
        else:
            # Show distribution
            v_mode = vbac_data.mode()[0] if len(vbac_data) > 0 else "N/A"
            c_mode = cs_data.mode()[0] if len(cs_data) > 0 else "N/A"
            rows.append({
                "Variable": meta["label"],
                f"VBAC (n={n_vbac:,})": f"mode={v_mode}",
                f"Repeat CS (n={n_cs:,})": f"mode={c_mode}",
                "Type": "Categorical"
            })

table1 = pd.DataFrame(rows).set_index("Variable")
st.dataframe(table1, use_container_width=True)

# ── Variable distributions ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Variable Distributions by Outcome")

numeric_vars = [v for v, m in all_vars.items() if m["type"] == "numeric" and v in df_combined.columns]

if numeric_vars:
    selected_var = st.selectbox(
        "Select variable to plot",
        numeric_vars,
        format_func=lambda v: all_vars[v]["label"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    vbac_vals = pd.to_numeric(df_combined.loc[df_combined["outcome"] == 1, selected_var], errors='coerce').dropna()
    cs_vals = pd.to_numeric(df_combined.loc[df_combined["outcome"] == 0, selected_var], errors='coerce').dropna()

    ax.hist(vbac_vals, bins=30, alpha=0.6, color="#2ecc71", label=f"VBAC (n={len(vbac_vals):,})", density=True)
    ax.hist(cs_vals, bins=30, alpha=0.6, color="#e74c3c", label=f"Repeat CS (n={len(cs_vals):,})", density=True)
    ax.set_xlabel(all_vars[selected_var]["label"])
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Correlation heatmap ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Correlation Matrix (Antenatal Features)")

numeric_ant_vars = [v for v, m in ANTENATAL_VARS.items() if v in df_ant.columns]

if len(numeric_ant_vars) >= 2:
    corr = df_ant[numeric_ant_vars].apply(pd.to_numeric, errors='coerce').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5,
        xticklabels=[ANTENATAL_VARS.get(v, {}).get("label", v)[:20] for v in numeric_ant_vars],
        yticklabels=[ANTENATAL_VARS.get(v, {}).get("label", v)[:20] for v in numeric_ant_vars],
        ax=ax
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Missing data ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Missing Data")

missing = df_modelB.isnull().mean().sort_values(ascending=False) * 100
missing = missing[missing > 0]

if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.4)))
    colors = ["#e74c3c" if v > 20 else "#f39c12" if v > 5 else "#3498db" for v in missing.values]
    ax.barh(
        [all_vars.get(v, {}).get("label", v)[:40] for v in missing.index],
        missing.values,
        color=colors
    )
    ax.set_xlabel("Missing (%)")
    ax.axvline(5, color="#f39c12", linestyle="--", alpha=0.7, label=">5% threshold")
    ax.axvline(20, color="#e74c3c", linestyle="--", alpha=0.7, label=">20% threshold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
else:
    st.success("No missing data in the mapped variables.")

st.markdown("---")
st.markdown("**Proceed to Step 3 (Train Models) using the sidebar.**")
