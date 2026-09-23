"""
pages/1_📊_Data_Analysis.py
Exploratory Data Analysis page for the Student Performance dataset.

Streamlit automatically turns any file inside a `pages/` folder into an
extra page in the app's sidebar navigation.

Requires data/student_performance.csv, produced by
train_student_performance_model.py.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = "data/student_performance.csv"

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="centered")

st.title("📊 Data Analysis")



@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


df = load_data()

if df is None:
    st.error(
        f"No dataset found at `{DATA_PATH}`. "
        "Run `python train_student_performance_model.py` first — it saves "
        "a copy of the raw data for this page to use."
    )
    st.stop()

numeric_cols = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Performance Index",
]

# --- Filters (sidebar) -----------------------------------------------------
st.sidebar.header("🔍 Filter the data")

hours_range = st.sidebar.slider(
    "Hours Studied", int(df["Hours Studied"].min()), int(df["Hours Studied"].max()),
    (int(df["Hours Studied"].min()), int(df["Hours Studied"].max())),
)
sleep_range = st.sidebar.slider(
    "Sleep Hours", int(df["Sleep Hours"].min()), int(df["Sleep Hours"].max()),
    (int(df["Sleep Hours"].min()), int(df["Sleep Hours"].max())),
)
activity_filter = st.sidebar.multiselect(
    "Extracurricular Activities",
    options=df["Extracurricular Activities"].unique().tolist(),
    default=df["Extracurricular Activities"].unique().tolist(),
)

df_f = df[
    df["Hours Studied"].between(*hours_range)
    & df["Sleep Hours"].between(*sleep_range)
    & df["Extracurricular Activities"].isin(activity_filter)
]

st.sidebar.caption(f"{len(df_f):,} of {len(df):,} rows match your filters")

if df_f.empty:
    st.warning("No rows match the current filters. Adjust them in the sidebar.")
    st.stop()

# From here on, df_f (the filtered data) drives every chart, so the whole
# page updates live as filters change.

# --- Overview -----------------------------------------------------------
st.divider()
st.subheader("Dataset overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows (filtered)", f"{len(df_f):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Avg. Performance Index", f"{df_f['Performance Index'].mean():.1f}")

with st.expander("Preview raw data"):
    st.dataframe(df_f.head(20), use_container_width=True)

with st.expander("Summary statistics"):
    st.dataframe(df_f[numeric_cols].describe().round(2), use_container_width=True)

st.download_button(
    "⬇️ Download filtered data as CSV",
    df_f.to_csv(index=False).encode("utf-8"),
    file_name="student_performance_filtered.csv",
    mime="text/csv",
)

# --- Distributions --------------------------------------------------------
st.divider()
st.subheader("Feature distributions")

selected_col = st.selectbox("Choose a feature to view its distribution", numeric_cols)
fig_hist = px.histogram(
    df_f, x=selected_col, nbins=30, title=f"Distribution of {selected_col}"
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Correlation heatmap ---------------------------------------------------
st.divider()
st.subheader("Correlation with Performance Index")

corr = df_f[numeric_cols].corr()
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation matrix",
)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Relationship explorer -------------------------------------------------
st.divider()
st.subheader("Feature vs. Performance Index")

feature_x = st.selectbox(
    "Choose a feature to plot against Performance Index",
    [c for c in numeric_cols if c != "Performance Index"],
)
fig_scatter = px.scatter(
    df_f,
    x=feature_x,
    y="Performance Index",
    color="Extracurricular Activities",
    opacity=0.4,
    trendline="ols",
    title=f"{feature_x} vs. Performance Index",
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Extracurricular activities comparison --------------------------------
st.divider()
st.subheader("Extracurricular Activities")

fig_box = px.box(
    df_f,
    x="Extracurricular Activities",
    y="Performance Index",
    color="Extracurricular Activities",
    title="Performance Index by Extracurricular Activities",
)
st.plotly_chart(fig_box, use_container_width=True)

# --- Pairwise relationships -------------------------------------------------
st.divider()
st.subheader("Pairwise relationships")
st.caption("How every numeric feature relates to every other, at a glance.")

fig_matrix = px.scatter_matrix(
    df_f,
    dimensions=numeric_cols,
    color="Extracurricular Activities",
    opacity=0.3,
    height=700,
)
fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
st.plotly_chart(fig_matrix, use_container_width=True)

# --- Sleep habits bucket analysis -------------------------------------------
st.divider()
st.subheader("Performance by sleep habits")

sleep_bins = [0, 4, 6, 8, 10, 12]
sleep_labels = ["0–4h", "4–6h", "6–8h", "8–10h", "10–12h"]
df_f = df_f.assign(
    sleep_bucket=pd.cut(df_f["Sleep Hours"], bins=sleep_bins, labels=sleep_labels, include_lowest=True)
)
sleep_avg = (
    df_f.groupby("sleep_bucket", observed=True)["Performance Index"]
    .mean()
    .reset_index()
)
fig_sleep = px.bar(
    sleep_avg,
    x="sleep_bucket",
    y="Performance Index",
    title="Average Performance Index by sleep-hours bucket",
    labels={"sleep_bucket": "Sleep Hours"},
)
st.plotly_chart(fig_sleep, use_container_width=True)

# --- Statistical significance test ------------------------------------------
st.divider()
st.subheader("Does extracurricular participation actually matter?")

from scipy import stats

group_yes = df_f.loc[df_f["Extracurricular Activities"] == "Yes", "Performance Index"]
group_no = df_f.loc[df_f["Extracurricular Activities"] == "No", "Performance Index"]

if len(group_yes) > 1 and len(group_no) > 1:
    t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. — participates", f"{group_yes.mean():.2f}")
    c2.metric("Avg. — doesn't", f"{group_no.mean():.2f}")
    c3.metric("p-value", f"{p_value:.4f}")

    if p_value < 0.05:
        st.success(
            "The difference is statistically significant (p < 0.05) — "
            "extracurricular participation is associated with a real "
            "difference in Performance Index in this dataset."
        )
    else:
        st.info(
            "The difference is not statistically significant (p ≥ 0.05) "
            "at the current filter settings."
        )
else:
    st.info("Not enough data in both groups (under current filters) to run this test.")

