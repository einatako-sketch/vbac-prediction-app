"""
Page 1 — Upload Data & Map Columns
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import ANTENATAL_VARS, INTRAPARTUM_VARS, derive_prev_vaginal

st.set_page_config(page_title="Upload & Map", page_icon="📂", layout="wide")
st.title("Step 1 — Upload Data & Map Columns")

st.markdown("""
Upload your hospital's delivery dataset. The file should contain one row per delivery attempt
(women with one prior cesarean who attempted TOLAC).

**Accepted formats:** Excel (.xlsx, .xls) or CSV (.csv)
""")

# ── File upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your dataset",
    type=["xlsx", "xls", "csv"],
    help="One row per delivery. Include outcome (delivery mode) and as many predictor variables as available."
)

if uploaded is None:
    st.info("Please upload a file to continue.")
    st.stop()

# Load file
@st.cache_data
def load_file(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    else:
        return pd.read_excel(f)

with st.spinner("Loading file..."):
    df_raw = load_file(uploaded)

st.success(f"Loaded: **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**")

with st.expander("Preview raw data (first 5 rows)"):
    st.dataframe(df_raw.head())

st.markdown("---")

# ── Column mapping ─────────────────────────────────────────────────────────────
st.subheader("Map Your Columns")
st.markdown("""
For each required variable, select the corresponding column in your dataset.
Leave as **"-- not available --"** if the variable is not in your data
(it will be imputed with the median during training).
""")

cols = ["-- not available --"] + list(df_raw.columns)

def col_select(label, key, help_text=""):
    return st.selectbox(label, cols, key=key, help=help_text)

# ── Outcome ────────────────────────────────────────────────────────────────────
st.markdown("### Outcome Variable")
outcome_col = col_select(
    "Delivery outcome column",
    "outcome_col",
    "Column containing delivery mode. Will be recoded: vaginal/vacuum = VBAC success (1), CS = failure (0)."
)

if outcome_col != "-- not available --":
    unique_vals = sorted(df_raw[outcome_col].dropna().unique())
    st.markdown(f"Unique values in outcome column: `{unique_vals}`")

    st.markdown("**Define VBAC success values** (all others will be treated as CS/failure):")
    vbac_success_vals = st.multiselect(
        "Values that represent VBAC success (vaginal / vacuum delivery)",
        options=unique_vals,
        default=[v for v in unique_vals if v in [2, 3, "2", "3", "vaginal", "Vaginal", "vacuum", "Vacuum"]],
        key="vbac_success_vals"
    )

st.markdown("---")

# ── Antenatal variables ────────────────────────────────────────────────────────
st.markdown("### Antenatal Variables (Model A + B)")

antenatal_map = {}
for var, meta in ANTENATAL_VARS.items():
    if var == "prev_vaginal_cs":
        continue
    antenatal_map[var] = col_select(
        f"{meta['label']}",
        f"ant_{var}",
        help_text=f"Type: {meta['type']}"
    )

# Derived variable: prev_vaginal_before_cs
st.markdown("#### Derived Variable: Prior Vaginal Delivery Before Index CS")
st.markdown("""
This variable is calculated as: **Parity − Prior CS count − Prior VBACs**.
You can either:
- Let us calculate it automatically (requires parity, CS count, and prior VBACs columns), or
- Map a pre-existing column directly.
""")

derive_auto = st.checkbox("Calculate automatically from parity, CS count, and prior VBACs", value=True)

if derive_auto:
    parity_col_deriv = col_select(
        "Parity column (total number of deliveries)",
        "parity_col_deriv",
        "Total number of prior deliveries (including cesareans and vaginal)"
    )
    cs_count_col = col_select(
        "Prior CS count column (for derivation)",
        "cs_count_col",
        "Number of prior cesarean deliveries"
    )
    antenatal_map["prev_vaginal_cs"] = "DERIVED"
else:
    antenatal_map["prev_vaginal_cs"] = col_select(
        "Prior vaginal delivery before index CS (direct column)",
        "ant_prev_vaginal_cs"
    )

st.markdown("---")

# ── Intrapartum variables ──────────────────────────────────────────────────────
st.markdown("### Intrapartum Variables (Model B only)")

intrapartum_map = {}
for var, meta in INTRAPARTUM_VARS.items():
    intrapartum_map[var] = col_select(
        f"{meta['label']}",
        f"inp_{var}",
        help_text=f"Type: {meta['type']}"
    )

st.markdown("---")

# ── Filters ────────────────────────────────────────────────────────────────────
st.subheader("Dataset Filters")
st.markdown("Apply filters to select only TOLAC attempts (exclude planned repeat CS, etc.)")

apply_filters = st.checkbox("Apply filters before training", value=True)

filter_col = None
filter_exclude = []

if apply_filters:
    filter_col = col_select(
        "Filter column (e.g. labor onset / delivery mode category)",
        "filter_col",
        "Select a column to filter rows"
    )
    if filter_col != "-- not available --":
        unique_filter = sorted(df_raw[filter_col].dropna().unique())
        filter_exclude = st.multiselect(
            "Exclude rows where this column equals:",
            options=unique_filter,
            help="e.g. exclude planned CS (value=2) and unspecified (value=0)"
        )

st.markdown("---")

# ── Save mapping and process ───────────────────────────────────────────────────
if st.button("Apply Mapping & Process Dataset", type="primary"):

    if outcome_col == "-- not available --":
        st.error("You must map the outcome column.")
        st.stop()

    if not vbac_success_vals:
        st.error("You must select at least one value representing VBAC success.")
        st.stop()

    df = df_raw.copy()

    # Apply filters
    if apply_filters and filter_col != "-- not available --" and filter_exclude:
        before = len(df)
        df = df[~df[filter_col].isin(filter_exclude)]
        st.info(f"Filtered out {before - len(df):,} rows. Remaining: {len(df):,}")

    # Create outcome
    df["VBAC_outcome"] = df[outcome_col].isin(vbac_success_vals).astype(int)

    # Build feature DataFrames
    def extract_feature(df, var, mapped_col, rename):
        if mapped_col == "-- not available --" or mapped_col is None:
            return pd.Series(np.nan, index=df.index, name=rename)
        return df[mapped_col].rename(rename)

    # Antenatal features
    ant_series = []
    for var, meta in ANTENATAL_VARS.items():
        if var == "prev_vaginal_cs":
            if antenatal_map.get("prev_vaginal_cs") == "DERIVED":
                parity_col = st.session_state.get("parity_col_deriv", "-- not available --")
                cs_col = st.session_state.get("cs_count_col", "-- not available --")
                vbac_col_mapped = antenatal_map.get("prev_vbac", "-- not available --")

                if all(c != "-- not available --" for c in [parity_col, cs_col, vbac_col_mapped]):
                    derived = derive_prev_vaginal(df, parity_col, cs_col, vbac_col_mapped)
                    ant_series.append(derived.rename("prev_vaginal_cs"))
                else:
                    ant_series.append(pd.Series(np.nan, index=df.index, name="prev_vaginal_cs"))
            else:
                mapped = antenatal_map.get("prev_vaginal_cs", "-- not available --")
                ant_series.append(extract_feature(df, var, mapped, "prev_vaginal_cs"))
        else:
            mapped = antenatal_map.get(var, "-- not available --")
            ant_series.append(extract_feature(df, var, mapped, var))

    df_ant = pd.concat(ant_series, axis=1)

    # Intrapartum features
    inp_series = []
    for var in INTRAPARTUM_VARS:
        mapped = intrapartum_map.get(var, "-- not available --")
        inp_series.append(extract_feature(df, var, mapped, var))

    df_inp = pd.concat(inp_series, axis=1)

    # Combine
    df_modelB = pd.concat([df_ant, df_inp], axis=1)
    y = df["VBAC_outcome"]

    # Store in session state
    st.session_state["df_ant"] = df_ant
    st.session_state["df_inp"] = df_inp
    st.session_state["df_modelB"] = df_modelB
    st.session_state["y"] = y
    st.session_state["df_filtered"] = df
    st.session_state["mapping_done"] = True

    # Summary
    n_total = len(y)
    n_vbac = y.sum()
    n_cs = n_total - n_vbac

    st.success(f"Dataset processed successfully!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cases", f"{n_total:,}")
    col2.metric("VBAC success", f"{n_vbac:,} ({100*n_vbac/n_total:.1f}%)")
    col3.metric("Repeat CS", f"{n_cs:,} ({100*n_cs/n_total:.1f}%)")

    # Missing data summary
    st.markdown("#### Missing Data Summary")
    missing = df_modelB.isnull().mean().round(3) * 100
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Variable", "Missing (%)"]
    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing data detected.")

    st.markdown("**Proceed to Step 2 (EDA) or Step 3 (Train Models) using the sidebar.**")

elif st.session_state.get("mapping_done"):
    st.info("Mapping already applied. Use the sidebar to navigate to the next step.")
