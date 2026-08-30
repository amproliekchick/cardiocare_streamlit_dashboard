"""
CardioCare Heart Disease Risk Analytics Dashboard
--------------------------------------------------
Interactive Streamlit prototype built for the BMDS2003 Data Science assignment
(Section 7.0 Deployment). It covers every guideline listed in the assignment brief:

 1) Data input          -> sidebar CSV uploader (falls back to the bundled dataset)
 2) Data exploration     -> "Data Explorer" tab: filter / sort / preview the dataset
 3) Feature importance   -> "Feature Importance" tab: Random Forest importances +
                             Logistic Regression coefficients, as heatmaps and bar charts
 4) Interactive charts   -> Plotly charts throughout that respond to widget input
 5) Visualisation comps  -> histograms, box plots, pie charts, heatmaps, confusion matrices
 6) User interaction     -> sliders, dropdowns, checkboxes, number inputs, a live patient form
 7) Real-time updates    -> "Live Monitoring" tab simulates a streaming feed of new patients
 8) Customisation        -> sidebar theme / layout controls that restyle the charts

Run locally with:  streamlit run app.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioCare | Heart Disease Risk Dashboard",
    page_icon="\u2764\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
TARGET = "Heart Disease Status"

CONTINUOUS_VARS = ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
                    "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]
CATEGORICAL_VARS = ["Gender", "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
                     "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
                     "Alcohol Consumption", "Stress Level", "Sugar Consumption"]

ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2}
BINARY_MAP_COLS = ["Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure",
                    "Low HDL Cholesterol", "High LDL Cholesterol"]
REQUIRED_COLUMNS = CONTINUOUS_VARS + CATEGORICAL_VARS + [TARGET]


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "heart_disease.csv")


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same imputation strategy used in the notebook: median for continuous, mode for categorical."""
    df = df.copy()
    for col in CONTINUOUS_VARS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_VARS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    bundle = joblib.load(APP_DIR / "rf_model.joblib")
    return bundle["model"], bundle["scaler"], bundle["features"]


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(APP_DIR / "model_metrics.json") as f:
        return json.load(f)


def encode_for_model(row: dict) -> pd.DataFrame:
    """Encode a single patient's raw feature dict the same way df_encoded was built."""
    encoded = {}
    for col, val in row.items():
        if col == "Gender":
            encoded[col] = 1 if val == "Male" else 0
        elif col in BINARY_MAP_COLS:
            encoded[col] = 1 if val == "Yes" else 0
        elif col in ["Exercise Habits", "Alcohol Consumption", "Stress Level", "Sugar Consumption"]:
            encoded[col] = ORDINAL_MAP[val]
        else:
            encoded[col] = val
    return pd.DataFrame([encoded])


model, scaler, FEATURES = load_model()
metrics = load_metrics()

# --------------------------------------------------------------------------------------
# Sidebar: data input + customisation (guidelines 1 & 8)
# --------------------------------------------------------------------------------------
st.sidebar.title("\u2764\ufe0f CardioCare Dashboard")
st.sidebar.caption("Heart Disease Risk Analytics \u2014 BMDS2003 Deployment Prototype")

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload a heart disease CSV (optional)", type=["csv"],
    help="Must contain the same 20 feature columns as the CardioCare dataset. "
         "If you don't upload a file, the bundled 10,000-patient dataset is used.",
)
if uploaded_file is not None:
    try:
        candidate_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"❌ Couldn't read **{uploaded_file.name}** as a CSV ({e}). Using the bundled dataset instead.")
        base_df = load_default_data()
    else:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in candidate_df.columns]
        if missing_cols:
            shown = ", ".join(missing_cols[:5]) + ("…" if len(missing_cols) > 5 else "")
            st.sidebar.error(
                f"❌ **{uploaded_file.name}** doesn't match the expected format — missing column(s): "
                f"{shown}. Using the bundled dataset instead."
            )
            base_df = load_default_data()
        else:
            base_df = candidate_df
            st.sidebar.success(f"✅ **{uploaded_file.name}** uploaded — {len(base_df):,} records loaded.")
else:
    base_df = load_default_data()
    st.sidebar.caption(f"Using the bundled dataset ({len(base_df):,} records).")

if "manual_records" not in st.session_state:
    st.session_state.manual_records = []

with st.sidebar.expander("➕ Add a patient record manually"):
    with st.form("manual_entry_form", clear_on_submit=True):
        m_age = st.number_input("Age", 18, 100, 50)
        m_gender = st.selectbox("Gender", ["Male", "Female"])
        m_bp = st.number_input("Blood Pressure", 80, 220, 130)
        m_chol = st.number_input("Cholesterol Level", 100, 400, 200)
        m_exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"], index=1)
        m_smoking = st.selectbox("Smoking", ["No", "Yes"])
        m_family = st.selectbox("Family Heart Disease", ["No", "Yes"])
        m_diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        m_bmi = st.number_input("BMI", 15.0, 45.0, 25.0, 0.1)
        m_hbp = st.selectbox("High Blood Pressure", ["No", "Yes"])
        m_low_hdl = st.selectbox("Low HDL Cholesterol", ["No", "Yes"])
        m_high_ldl = st.selectbox("High LDL Cholesterol", ["No", "Yes"])
        m_alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"], index=1)
        m_stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
        m_sleep = st.number_input("Sleep Hours", 3.0, 12.0, 7.0, 0.1)
        m_sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"], index=1)
        m_trig = st.number_input("Triglyceride Level", 50, 600, 150)
        m_fbs = st.number_input("Fasting Blood Sugar", 60, 300, 100)
        m_crp = st.number_input("CRP Level", 0.0, 20.0, 3.0, 0.1)
        m_homo = st.number_input("Homocysteine Level", 3.0, 25.0, 10.0, 0.1)
        m_status = st.selectbox("Heart Disease Status (if known)", ["No", "Yes"])
        add_record = st.form_submit_button("Add record to dataset")

    if add_record:
        st.session_state.manual_records.append({
            "Age": m_age, "Gender": m_gender, "Blood Pressure": m_bp, "Cholesterol Level": m_chol,
            "Exercise Habits": m_exercise, "Smoking": m_smoking, "Family Heart Disease": m_family,
            "Diabetes": m_diabetes, "BMI": m_bmi, "High Blood Pressure": m_hbp,
            "Low HDL Cholesterol": m_low_hdl, "High LDL Cholesterol": m_high_ldl,
            "Alcohol Consumption": m_alcohol, "Stress Level": m_stress, "Sleep Hours": m_sleep,
            "Sugar Consumption": m_sugar, "Triglyceride Level": m_trig, "Fasting Blood Sugar": m_fbs,
            "CRP Level": m_crp, "Homocysteine Level": m_homo, "Heart Disease Status": m_status,
        })
        st.success("Added — it's now included in every tab below.")

    if st.session_state.manual_records:
        st.caption(f"{len(st.session_state.manual_records)} manually entered record(s) this session.")
        if st.button("Clear manual records"):
            st.session_state.manual_records = []
            st.rerun()

if st.session_state.manual_records:
    raw_df = pd.concat([base_df, pd.DataFrame(st.session_state.manual_records)], ignore_index=True)
else:
    raw_df = base_df
df = clean_data(raw_df)

st.sidebar.header("Customisation")
color_theme = st.sidebar.selectbox(
    "Chart color theme", ["Blues", "Reds", "Viridis", "Teal", "Purples", "Oranges"], index=0
)
chart_template = st.sidebar.radio("Chart style", ["plotly_white", "plotly_dark", "seaborn"], index=0)
compact_view = st.sidebar.checkbox("Compact layout (hide long descriptions)", value=False)

st.sidebar.divider()
st.sidebar.metric("Records loaded", f"{len(df):,}")
st.sidebar.metric("Columns", f"{df.shape[1]}")
if TARGET in df.columns:
    pos_rate = (df[TARGET] == "Yes").mean()
    st.sidebar.metric("Heart disease prevalence", f"{pos_rate:.1%}")

st.title("CardioCare Health System \u2014 Heart Disease Risk Analytics")
st.caption(
    "Interactive companion to Section 6.0 (Results and Discussion) of the BMDS2003 report. "
    "Explore the data, inspect model performance and feature importance, try the risk-prediction "
    "prototype, and watch a simulated real-time patient feed."
)

tab_explore, tab_eda, tab_importance, tab_models, tab_predict, tab_live = st.tabs(
    ["\U0001F4CA Data Explorer", "\U0001F50E EDA", "\U0001F3AF Feature Importance",
     "\U0001F9EA Model Comparison", "\U0001FA7A Risk Prediction", "\U0001F534 Live Monitoring"]
)

# ========================================================================================
# TAB 1 - Data Explorer (guideline 2: filter, sort, visualise subsets)
# ========================================================================================
with tab_explore:
    st.subheader("Explore and filter the patient dataset")
    if not compact_view:
        st.write(
            "Use the controls below to filter the dataset interactively, then sort and preview "
            "the resulting subset. All charts on this page update live as you change the filters."
        )

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()),
                               (int(df["Age"].min()), int(df["Age"].max())))
    with fcol2:
        gender_filter = st.multiselect("Gender", sorted(df["Gender"].dropna().unique().tolist()),
                                        default=sorted(df["Gender"].dropna().unique().tolist()))
    with fcol3:
        status_filter = st.multiselect(TARGET, sorted(df[TARGET].dropna().unique().tolist()),
                                        default=sorted(df[TARGET].dropna().unique().tolist()))
    with fcol4:
        smoking_filter = st.multiselect("Smoking", sorted(df["Smoking"].dropna().unique().tolist()),
                                         default=sorted(df["Smoking"].dropna().unique().tolist()))

    search_col = st.selectbox("Sort by column", df.columns.tolist(), index=0)
    sort_desc = st.checkbox("Descending order", value=False)

    filtered = df[
        df["Age"].between(*age_range)
        & df["Gender"].isin(gender_filter)
        & df[TARGET].isin(status_filter)
        & df["Smoking"].isin(smoking_filter)
    ].sort_values(by=search_col, ascending=not sort_desc)

    st.dataframe(filtered, use_container_width=True, height=320)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records after filtering.")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        num_col = st.selectbox("Histogram variable", CONTINUOUS_VARS, index=0, key="hist_var")
        fig = px.histogram(filtered, x=num_col, color=TARGET, barmode="overlay",
                            nbins=30, template=chart_template,
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Distribution of {num_col} by {TARGET}")
        st.plotly_chart(fig, use_container_width=True)
    with ecol2:
        cat_col = st.selectbox("Category breakdown", CATEGORICAL_VARS, index=0, key="cat_var")
        pie_df = filtered[cat_col].value_counts().reset_index()
        pie_df.columns = [cat_col, "count"]
        fig2 = px.pie(pie_df, names=cat_col, values="count", template=chart_template,
                       color_discrete_sequence=getattr(px.colors.sequential, color_theme, px.colors.sequential.Blues))
        fig2.update_layout(title=f"Share of patients by {cat_col}")
        st.plotly_chart(fig2, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("\U0001F4E5 Download filtered data as CSV", csv_bytes, "filtered_patients.csv", "text/csv")

# ========================================================================================
# TAB 2 - EDA (guideline 5: histograms, box plots, correlation heatmap)
# ========================================================================================
with tab_eda:
    st.subheader("Exploratory data analysis")

    df_encoded = df.copy()
    for col in BINARY_MAP_COLS:
        df_encoded[col] = df_encoded[col].map({"No": 0, "Yes": 1})
    df_encoded["Gender"] = df_encoded["Gender"].map({"Female": 0, "Male": 1})
    for col in ["Exercise Habits", "Alcohol Consumption", "Stress Level", "Sugar Consumption"]:
        df_encoded[col] = df_encoded[col].map(ORDINAL_MAP)
    df_encoded[TARGET] = df_encoded[TARGET].map({"No": 0, "Yes": 1})

    st.markdown("**Box plots for continuous variables** (spot outliers / spread by class)")
    box_vars = st.multiselect("Variables to plot", CONTINUOUS_VARS, default=CONTINUOUS_VARS[:4])
    if box_vars:
        melted = df.melt(id_vars=[TARGET], value_vars=box_vars, var_name="Variable", value_name="Value")
        fig_box = px.box(melted, x="Variable", y="Value", color=TARGET, template=chart_template,
                          color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("**Correlation heatmap** (encoded features vs. target \u2014 see Section 3.4)")
    corr = df_encoded.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", template=chart_template,
                          color_continuous_scale=color_theme, zmin=-1, zmax=1)
    fig_corr.update_layout(height=650, title="Correlation matrix of all encoded features")
    st.plotly_chart(fig_corr, use_container_width=True)

    if not compact_view:
        max_abs_corr = corr[TARGET].drop(TARGET).abs().max()
        st.info(
            f"The strongest absolute correlation any feature has with **{TARGET}** in the currently "
            f"loaded data is **{max_abs_corr:.3f}** \u2014 consistent with Section 3.4 of the report, "
            "which found no feature with a meaningfully strong linear relationship to the target."
        )

# ========================================================================================
# TAB 3 - Feature Importance (guideline 3)
# ========================================================================================
with tab_importance:
    st.subheader("What drives the model's predictions?")

    rf_imp = metrics["models"]["Random Forest"]["feature_importances"]
    lr_coef = metrics["models"]["Logistic Regression"]["coefficients"]

    imp_col1, imp_col2 = st.columns(2)
    with imp_col1:
        imp_df = pd.DataFrame(sorted(rf_imp.items(), key=lambda x: -x[1]), columns=["Feature", "Importance"])
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", template=chart_template,
                          color="Importance", color_continuous_scale=color_theme)
        fig_imp.update_layout(title="Random Forest \u2014 impurity-based feature importance",
                               yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)
    with imp_col2:
        coef_df = pd.DataFrame(sorted(lr_coef.items(), key=lambda x: x[1]), columns=["Feature", "Coefficient"])
        fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", template=chart_template,
                           color="Coefficient", color_continuous_scale="RdBu")
        fig_coef.update_layout(title="Logistic Regression \u2014 standardised coefficients")
        st.plotly_chart(fig_coef, use_container_width=True)

    heat_df = pd.DataFrame({"RF Importance": rf_imp, "|LR Coefficient|": {k: abs(v) for k, v in lr_coef.items()}})
    heat_df = heat_df.loc[imp_df["Feature"]]
    fig_heat = px.imshow(heat_df.T, text_auto=".3f", aspect="auto", template=chart_template,
                          color_continuous_scale=color_theme)
    fig_heat.update_layout(title="Feature importance heatmap (Random Forest vs. Logistic Regression)")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.warning(
        "\u26A0\uFE0F **Interpretation caution (see Section 6.2 of the report):** every feature's raw "
        "correlation with Heart Disease Status is below 0.02, so these importance rankings mostly "
        "reflect each feature's numeric scale and cardinality rather than a genuine clinical "
        "relationship. They should not be treated as validated clinical risk factors."
    )

# ========================================================================================
# TAB 4 - Model Comparison (guideline 4: interactive performance charts)
# ========================================================================================
with tab_models:
    st.subheader("Comparing the four tuned models (Section 5.0 of the report)")

    model_names = list(metrics["models"].keys())
    metric_choice = st.selectbox("Metric to compare", ["test_accuracy", "test_auc", "test_log_loss", "cv_mean"],
                                  format_func=lambda x: {"test_accuracy": "Test Accuracy", "test_auc": "Test ROC-AUC",
                                                          "test_log_loss": "Test Log Loss",
                                                          "cv_mean": "10-Fold CV Mean Accuracy"}[x])
    comp_df = pd.DataFrame({
        "Model": model_names,
        "Value": [metrics["models"][m][metric_choice] for m in model_names],
    })
    fig_cmp = px.bar(comp_df, x="Model", y="Value", color="Model", template=chart_template,
                      color_discrete_sequence=px.colors.qualitative.Set2, text_auto=".3f")
    fig_cmp.update_layout(title=f"Model comparison \u2014 {metric_choice}", showlegend=False)
    if metric_choice == "test_auc":
        fig_cmp.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random guessing (AUC = 0.50)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("**Precision / Recall / F1 by class** (class imbalance makes this more informative than accuracy)")
    metric_type = st.radio("Metric", ["precision", "recall", "f1"], horizontal=True)
    pr_data = {m: metrics["models"][m][metric_type] for m in model_names}
    pr_df = pd.DataFrame(pr_data).T
    fig_pr = px.imshow(pr_df, text_auto=".2f", template=chart_template, color_continuous_scale=color_theme,
                        zmin=0, zmax=1)
    fig_pr.update_layout(title=f"{metric_type.capitalize()} heatmap by model and class")
    st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("**Confusion matrix**")
    model_pick = st.selectbox("Select a model", model_names, key="cm_model")
    cm = np.array(metrics["models"][model_pick]["confusion_matrix"])
    fig_cm = px.imshow(cm, text_auto=True, template=chart_template, color_continuous_scale=color_theme,
                        x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"])
    fig_cm.update_layout(title=f"Confusion matrix \u2014 {model_pick}")
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("Best hyperparameters found for each model"):
        st.json({m: metrics["models"][m]["best_params"] for m in model_names})

# ========================================================================================
# TAB 5 - Risk Prediction prototype (guideline 6: widgets + practical deployment objective)
# ========================================================================================
with tab_predict:
    st.subheader("Try the deployed prototype: Random Forest risk prediction")
    st.caption(
        "This mirrors the assignment's Practical Deployment objective (Section 1.2): enter a "
        "patient's characteristics and receive a heart disease risk prediction from the tuned "
        "Random Forest model selected in Section 5.5."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, 0.1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_pressure = st.slider("Blood Pressure (mmHg)", 90, 200, 130)
        with c2:
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
            age = st.number_input("Age", min_value=18, max_value=100, value=50)
            homocysteine = st.slider("Homocysteine Level (\u00b5mol/L)", 4.0, 20.0, 10.0, 0.1)
        with c3:
            high_ldl = st.selectbox("High LDL Cholesterol", ["No", "Yes"])
            family_history = st.selectbox("Family Heart Disease", ["No", "Yes"])
            sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"], index=1)
        crp = st.slider("CRP Level (mg/L)", 0.0, 15.0, 3.0, 0.1)

        submitted = st.form_submit_button("Predict heart disease risk", use_container_width=True)

    if submitted:
        patient = {
            "BMI": bmi, "Gender": gender, "Blood Pressure": blood_pressure, "Stress Level": stress,
            "Age": age, "Homocysteine Level": homocysteine, "High LDL Cholesterol": high_ldl,
            "Family Heart Disease": family_history, "Sugar Consumption": sugar, "CRP Level": crp,
        }
        encoded = encode_for_model(patient)[FEATURES]
        scaled = scaler.transform(encoded)
        proba = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]

        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            label = "Heart Disease Likely" if pred == 1 else "Heart Disease Unlikely"
            st.metric("Prediction", label)
            st.metric("Predicted probability of disease", f"{proba[1]:.1%}")
        with rcol2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={"text": "Predicted Risk (%)"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "firebrick" if proba[1] > 0.5 else "seagreen"},
                       "steps": [{"range": [0, 50], "color": "#e8f5e9"},
                                 {"range": [50, 100], "color": "#ffebee"}]},
            ))
            fig_gauge.update_layout(template=chart_template, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.error(
            "\u26A0\uFE0F **Clinical disclaimer:** as documented in Section 6.1 of the report, this Random "
            "Forest model achieves a test ROC-AUC of only \u2248 0.52 (random guessing = 0.50) and correctly "
            "recalls under 1% of true disease cases. This prototype demonstrates the deployment pipeline "
            "required by the assignment; it is **not** validated for clinical use and should not inform "
            "real patient care until retrained on data with a demonstrated predictive signal."
        )

# ========================================================================================
# TAB 6 - Live Monitoring simulation (guideline 7: real-time updates)
# ========================================================================================
with tab_live:
    st.subheader("Simulated real-time patient monitoring")
    st.caption(
        "Streamlit dashboards can refresh as new data arrives from a live source (e.g. a hospital "
        "feed or streaming API). No live feed is available for this assignment, so this tab "
        "simulates one by streaming rows from the dataset a few at a time \u2014 press **Start** to watch "
        "the running risk-prediction distribution update in real time."
    )

    if "stream_idx" not in st.session_state:
        st.session_state.stream_idx = 0
    if "stream_log" not in st.session_state:
        st.session_state.stream_log = []

    lcol1, lcol2, lcol3 = st.columns(3)
    batch_size = lcol1.slider("Patients per update", 1, 20, 5)
    n_batches = lcol2.slider("Number of updates", 1, 20, 5)
    start = lcol3.button("\u25B6\ufe0f Start simulated stream", use_container_width=True)

    placeholder = st.empty()

    if start:
        st.session_state.stream_log = []
        stream_source = df.sample(frac=1.0, random_state=None).reset_index(drop=True)
        for b in range(n_batches):
            start_i = (st.session_state.stream_idx + b * batch_size) % len(stream_source)
            batch = stream_source.iloc[start_i: start_i + batch_size]
            if batch.empty:
                continue
            encoded_batch = batch.copy()
            for col in BINARY_MAP_COLS:
                encoded_batch[col] = encoded_batch[col].map({"No": 0, "Yes": 1})
            encoded_batch["Gender"] = encoded_batch["Gender"].map({"Female": 0, "Male": 1})
            for col in ["Exercise Habits", "Alcohol Consumption", "Stress Level", "Sugar Consumption"]:
                encoded_batch[col] = encoded_batch[col].map(ORDINAL_MAP)
            scaled_batch = scaler.transform(encoded_batch[FEATURES])
            batch_proba = model.predict_proba(scaled_batch)[:, 1]

            for p in batch_proba:
                st.session_state.stream_log.append(p)

            with placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Patients streamed", len(st.session_state.stream_log))
                m2.metric("Mean predicted risk", f"{np.mean(st.session_state.stream_log):.1%}")
                m3.metric("High-risk flags (>50%)", int(np.sum(np.array(st.session_state.stream_log) > 0.5)))

                line_color = getattr(px.colors.sequential, color_theme, px.colors.sequential.Blues)[3]
                fig_stream = px.line(y=st.session_state.stream_log, markers=True, template=chart_template,
                                      labels={"index": "Patient (streaming order)", "y": "Predicted risk"})
                fig_stream.update_traces(line_color=line_color, marker_color=line_color)
                fig_stream.update_layout(title="Running predicted-risk stream", yaxis_range=[0, 1])
                st.plotly_chart(fig_stream, use_container_width=True, key=f"stream_chart_{b}")

            time.sleep(0.4)

        st.session_state.stream_idx += n_batches * batch_size
        st.success(f"Streamed {len(st.session_state.stream_log)} simulated patients.")
    elif st.session_state.stream_log:
        with placeholder.container():
            st.info("Showing the last simulated run. Press **Start** again to stream a new batch.")
            fig_stream = px.line(y=st.session_state.stream_log, markers=True, template=chart_template)
            fig_stream.update_layout(title="Running predicted-risk stream (last run)", yaxis_range=[0, 1])
            st.plotly_chart(fig_stream, use_container_width=True)
    else:
        placeholder.info("Press **Start simulated stream** to begin.")

st.divider()
st.caption(
    "BMDS2003 Data Science \u2014 CardioCare Health System heart disease risk assignment. "
    "Dashboard built with Streamlit + Plotly. Model: Random Forest (tuned, Section 4.1 / 5.5 of the report)."
)
