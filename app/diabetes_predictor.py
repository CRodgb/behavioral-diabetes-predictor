# diabetes_predictor.py
'''
1. Loads xgb_pipeline_model2.joblib
2. Uses the composite dataset to drive options
3. Requests a small, human-readable subset of inputs
4. Fills missing columns with defaults
5. Predicts risk and explains it with SHAP
'''
# ----------------------------
import sys
from pathlib import Path
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.markdown("""
<style>

/* -----------------------------------------------------
   GLOBAL BASE STYLING
------------------------------------------------------*/

html, body {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* -----------------------------------------------------
   MAIN PAGE BACKGROUND
------------------------------------------------------*/
.main {
    background-color: #f7f9fc !important;
}

/* -----------------------------------------------------
   SIDEBAR: DARK NAVY THEME
------------------------------------------------------*/
[data-testid="stSidebar"] {
    background-color: #071a2f !important;   /* DARK NAVY */
    color: #E8ECEF !important;
    padding: 1.2rem;
}

[data-testid="stSidebar"] * {
    color: #E8ECEF !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

/* Links inside sidebar */
[data-testid="stSidebar"] a {
    color: #90c7ff !important;
    text-decoration: none !important;
}
[data-testid="stSidebar"] a:hover {
    text-decoration: underline !important;
}

/* -----------------------------------------------------
   SIDEBAR TOGGLE BUTTON FIX – SINGLE ICON
------------------------------------------------------*/

/* 1) Hide the main header hamburger (top-right of page) */
button[kind="header"] {
    display: none !important;
}

/* 2) Clean up the sidebar's own toggle and replace text with icon */
[data-testid="collapsedControl"] span {
    font-size: 0 !important;      /* hide default "keyboard" text */
    line-height: 0 !important;
}

/* Add our custom hamburger icon */
[data-testid="collapsedControl"] span::before {
    content: "☰" !important;
    font-size: 23px !important;
    color: #E8ECEF !important;
    line-height: 1 !important;
    font-weight: 600 !important;
}

/* Hover effect */
[data-testid="collapsedControl"]:hover span::before {
    color: #ffffff !important;
}


/* -----------------------------------------------------
   CARDS / CONTAINERS
------------------------------------------------------*/
.stButton>button {
    border-radius: 6px !important;
    background-color: #0f4c81 !important;
    color: white !important;
    border: 0 !important;
}

.stButton>button:hover {
    background-color: #1261a3 !important;
}

/* Input widgets (more modern look) */
.stSelectbox, .stNumberInput input {
    border-radius: 6px !important;
}

/* -----------------------------------------------------
   DATAFRAMES (cleaner look)
------------------------------------------------------*/
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* -----------------------------------------------------
   SCROLLBAR STYLE
------------------------------------------------------*/
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #0b2239;
}
::-webkit-scrollbar-thumb {
    background: #375a7f;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4f7aad;
}

/* -----------------------------------------------------
   EXPANDER STYLE
------------------------------------------------------*/
.streamlit-expanderHeader {
    font-weight: 600 !important;
}

/* -----------------------------------------------------
   FORM CARD (patient profile)
------------------------------------------------------*/
.patient-card {
    border: 1px solid #e0e0e0;
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    background-color: white;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}


</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Project imports
# ---------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.defaults import get_training_df, compute_defaults, get_choices  # noqa: E402

BASE_DIR = ROOT_DIR
MODEL_PATH = BASE_DIR / "models" / "xgb_pipeline_model2.joblib"


# ---------------------------------------------------
# Caching: model + training data + defaults
# ---------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_training_and_defaults():
    df = get_training_df()
    defaults = compute_defaults()
    return df, defaults


pipeline = load_pipeline()
train_df, defaults = load_training_and_defaults()

preprocessor = pipeline.named_steps.get("preprocessor", None)
model = pipeline.named_steps.get("model", None)

# ---------------------------------------------------
# Global settings
# ---------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
)

CLASS_MAP = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}

# UI layout groups: 3 inputs per row
BEHAVIORAL_GROUPS = [
    ["Physical_Activity", "Alcohol_Risk_Level", "Average_Drinks_Per_Day"],
    ["Heavy_Drinking_Flag", "Current_Smoking_Frequency", "Ever_Smoked_100_Cigarettes"],
    ["Mentally_Unhealthy_Days_Count", "Physically_Unhealthy_Days_Count", "Unhealthy_Days_Total"],
]

ANTHRO_GROUPS = [
    ["Body_Mass_Index", "BMI_Category", "Weight_Pounds"],
]

DEMO_GROUPS = [
    ["Biological_Sex", "Age_Code", "Race_Ethnicity_Group"],
    ["Education_Level", "Household_Income_Category", "State_Name"],
]

# Columns that should be numeric in UI
NUMERIC_INPUT_COLS = {
    "Average_Drinks_Per_Day",
    "Mentally_Unhealthy_Days_Count",
    "Physically_Unhealthy_Days_Count",
    "Unhealthy_Days_Total",
    "Body_Mass_Index",
    "Weight_Pounds",
    "Age_Code"
}

# ----------------- SHAP label helpers -----------------

# Human-readable labels for specific (feature, code) combos
FEATURE_VALUE_LABELS = {
    # -------- Smoking ---------
    ("Smoking_Status_Category", "1.0"): "Smoking status: Every day",
    ("Smoking_Status_Category", "2.0"): "Smoking status: Some days",
    ("Smoking_Status_Category", "3.0"): "Smoking status: Former smoker",
    ("Smoking_Status_Category", "4.0"): "Smoking status: Never smoked",

    ("Ever_Smoked_100_Cigarettes", "1.0"): "Ever smoked ≥100 cigarettes (Yes)",
    ("Ever_Smoked_100_Cigarettes", "2.0"): "Ever smoked ≥100 cigarettes (No)",

    ("Current_Smoking_Frequency", "1.0"): "Smoking frequency: Daily",
    ("Current_Smoking_Frequency", "2.0"): "Smoking frequency: Some days",
    ("Current_Smoking_Frequency", "3.0"): "Smoking frequency: Not at all",

    # -------- Alcohol ---------
    ("Alcohol_Risk_Level", "0.0"): "Alcohol risk: None",
    ("Alcohol_Risk_Level", "1.0"): "Alcohol risk: Moderate",
    ("Alcohol_Risk_Level", "2.0"): "Alcohol risk: Heavy",

    ("Heavy_Drinking_Flag", "0.0"): "Heavy drinking: No",
    ("Heavy_Drinking_Flag", "1.0"): "Heavy drinking: Yes",

    # -------- BMI ---------
    ("BMI_Category", "1.0"): "BMI: Underweight",
    ("BMI_Category", "2.0"): "BMI: Normal weight",
    ("BMI_Category", "3.0"): "BMI: Overweight",
    ("BMI_Category", "4.0"): "BMI: Obese",

    ("At_Risk_BMI", "1.0"): "At-risk BMI: High obesity",
    ("At_Risk_BMI", "0.0"): "At-risk BMI: No",

    # # -------- Age Groups (BRFSS style; adjust to your composite codes) ---------
    # ("Age_Code", "1.0"): "Age 18–24",
    # ("Age_Code", "2.0"): "Age 25–34",
    # ("Age_Code", "3.0"): "Age 35–44",
    # ("Age_Code", "4.0"): "Age 45–54",
    # ("Age_Code", "5.0"): "Age 55–64",
    # ("Age_Code", "6.0"): "Age 65+",

    # -------- Race/Ethnicity ---------
    ("Race_Ethnicity_Group", "1.0"): "Race: White non-Hispanic",
    ("Race_Ethnicity_Group", "2.0"): "Race: Black non-Hispanic",
    ("Race_Ethnicity_Group", "3.0"): "Race: Other race",
    ("Race_Ethnicity_Group", "4.0"): "Race: Multiracial",
    ("Race_Ethnicity_Group", "5.0"): "Ethnicity: Hispanic",

    # -------- Sex ---------
    ("Biological_Sex", "1.0"): "Biological sex: Male",
    ("Biological_Sex", "2.0"): "Biological sex: Female",

    # -------- Income ---------
    ("Household_Income_Category", "1.0"): "< $10,000",
    ("Household_Income_Category", "2.0"): "$10k–15k",
    ("Household_Income_Category", "3.0"): "$15k–20k",
    ("Household_Income_Category", "4.0"): "$20k–25k",
    ("Household_Income_Category", "5.0"): "$25k–35k",
    ("Household_Income_Category", "6.0"): "$35k–50k",
    ("Household_Income_Category", "7.0"): "$50k–75k",
    ("Household_Income_Category", "8.0"): "$75k–100k",
    ("Household_Income_Category", "9.0"): "$100k–150k",
    ("Household_Income_Category", "10.0"): "$150k–200k",
    ("Household_Income_Category", "11.0"): "$200k+",

    # -------- Education ---------
    ("Education_Level", "1.0"): "Education: Never attended",
    ("Education_Level", "2.0"): "Education: Elementary",
    ("Education_Level", "3.0"): "Education: Some high school",
    ("Education_Level", "4.0"): "Education: High school graduate",
    ("Education_Level", "5.0"): "Education: Some college",
    ("Education_Level", "6.0"): "Education: College graduate",

    # -------- Physical Activity ---------
    ("Physical_Activity", "1.0"): "Physically active (past 30 days)",
    ("Physical_Activity", "2.0"): "No physical activity (past 30 days)",
}


def prettify_feature_name(raw_name: str) -> str:
    """
    Turn pipeline feature names like 'cat__Smoking_Status_Category_1.0'
    into human-readable labels like 'Smoking status: Every day'.
    """
    name = raw_name

    # 1) Strip common prefixes from ColumnTransformer/OneHot
    for prefix in ("num__", "cat__", "remainder__", "passthrough__"):
        if name.startswith(prefix):
            name = name[len(prefix):]

    # 2) Sometimes you get "At_Risk_BMI.1_1.0" etc. – we'll just work with what's left
    # Split on '_' and see if the last piece looks like a numeric code (like '1.0', '25.0')
    parts = name.split("_")
    last = parts[-1]

    is_code = last.replace(".", "", 1).isdigit()

    if is_code and len(parts) > 1:
        base = "_".join(parts[:-1])
        code = last

        # Make a nicer base label (spaces instead of underscores)
        base_pretty = base.replace("__", "_").replace("_", " ")

        # Check if we have a specific mapping
        key = (base_pretty, code)
        if key in FEATURE_VALUE_LABELS:
            return FEATURE_VALUE_LABELS[key]

        # Fallback generic label e.g. "Smoking status category = 1.0"
        return f"{base_pretty.capitalize()} = {code}"

    # 3) No code – just replace underscores with spaces
    return name.replace("__", "_").replace("_", " ").capitalize()


# ---------------------------------------------------
# Labels & help text
# ---------------------------------------------------
LABELS = {
    "Physical_Activity": "Physical activity in past 30 days",
    "Alcohol_Risk_Level": "Alcohol risk level",
    "Average_Drinks_Per_Day": "Average drinks per day (on drinking days)",
    "Heavy_Drinking_Flag": "Heavy drinking status",
    "Current_Smoking_Frequency": "Current smoking frequency",
    "Ever_Smoked_100_Cigarettes": "Ever smoked ≥100 cigarettes",
    "Mentally_Unhealthy_Days_Count": "Mentally unhealthy days (past 30 days)",
    "Physically_Unhealthy_Days_Count": "Physically unhealthy days (past 30 days)",
    "Unhealthy_Days_Total": "Total unhealthy days (physical + mental)",
    "Body_Mass_Index": "Body Mass Index (BMI)",
    "BMI_Category": "BMI category",
    "Weight_Pounds": "Weight (lbs)",
    "Biological_Sex": "Biological sex",
    "Age_Code": "Age group",
    "Race_Ethnicity_Group": "Race / ethnicity",
    "Education_Level": "Education level",
    "Household_Income_Category": "Household income",
    "State_Name": "State / territory",
}

HELP_TEXTS = {
    "Physical_Activity": "Any physical activity in the past 30 days? Yes = 1, No = 2",
    "Alcohol_Risk_Level": "Alcohol_Risk_Level: 0=none, 1=moderate, 2=heavy",
    "Average_Drinks_Per_Day": "During the past 30 days, on the days when you drank, about how many drinks did you drink on the average?  (A 40 ounce beer would count as 3 drinks, or a cocktail drink with 2 shots would count as 2 drinks.)",
    "Heavy_Drinking_Flag": "1 - 76 = No. of times, 0 = None",
    "Current_Smoking_Frequency": "1 = Every day, 2 = Some Days, 3 = Not at all",
    "Ever_Smoked_100_Cigarettes": "1 = Yes, 2 = No",
    "Mentally_Unhealthy_Days_Count": "Number of days mental health was not good during past 30 days.",
    "Physically_Unhealthy_Days_Count": "Number of days physical health was not good during past 30 days.",
    "Unhealthy_Days_Total": "Total unhealthy days (physical + mental), capped at 30.",
    "Body_Mass_Index": "BMI (1 - 9999)",
    "BMI_Category": "1 = Underweight, 2 = Normal Weight, 3 = Overweight, 4 = Obese",
    "Weight_Pounds": "Body weight in pounds.",
    "Biological_Sex": "1 = Male, 2 = Female",
    "Age_Code": "Age between 18 and 80",
    "Race_Ethnicity_Group": "1 = White, 2 = Black, 3 = Other, 4 = Multiracial, 5 = Hispanic, nan = Don't know",
    "Education_Level": "1 = Never Attended, 2 = Elementary, 3 = Some High School, 4 = High School Graduate, 5 = Some college or technical school, 6 = College Graduate, nan = Not sure",
    "Household_Income_Category": "1 = Less than $10,000, 2 = $10,000 to < $15,000, 3 = $15,000 to < $20,000, 4 = $20,000 to < $25,000, 5 = $25,000 to < $35,000, 6 = $35,000 to < $50,000, 7 = $50,000 to < $75,000, 8 = 75,000 to < $100,000, 9 = $100,000 to < $150,000, 10 = $150,000 to < $200,000, 11 = $200,000 or More, nan = Don't Know",
    "State_Name": "State or territory",
}

# ---------------------------------------------------
# Helper functions: numeric ranges, inputs, SHAP, PDF
# ---------------------------------------------------
def get_numeric_bounds(col: str):
    """Use robust 1–99% quantiles to avoid outliers driving UI limits."""
    s = pd.to_numeric(train_df[col], errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    q1, q99 = s.quantile([0.01, 0.99])
    min_v = float(q1)
    max_v = float(q99)
    if min_v >= max_v:
        max_v = min_v + 1.0
    return min_v, max_v


def clamp_default(col: str, min_v: float, max_v: float):
    """Clamp default value into [min_v, max_v] so Streamlit doesn't crash."""
    raw = defaults.get(col, None)
    if raw is None:
        return float((min_v + max_v) / 2.0)
    try:
        val = float(raw)
        if np.isnan(val):
            raise ValueError
    except Exception:
        return float((min_v + max_v) / 2.0)
    return float(max(min_v, min(max_v, val)))


def render_numeric_input(col_name: str, container):
    min_v, max_v = get_numeric_bounds(col_name)
    default_val = clamp_default(col_name, min_v, max_v)

    return container.number_input(
        LABELS.get(col_name, col_name),
        min_value=float(min_v),
        max_value=float(max_v),
        value=float(default_val),
        step=0.1,
        help=HELP_TEXTS.get(col_name, ""),
    )


def render_categorical_input(col_name: str, container):
    choices = get_choices(col_name)
    default_val = defaults.get(col_name, None)
    if not choices:
        choices = sorted(train_df[col_name].dropna().unique().tolist())
    if default_val in choices:
        index = choices.index(default_val)
    else:
        index = 0

    return container.selectbox(
        LABELS.get(col_name, col_name),
        options=choices,
        index=index,
        help=HELP_TEXTS.get(col_name, ""),
    )


def render_grouped_inputs(groups, title: str, subtitle: str):
    st.markdown(f"### {title}")
    st.caption(subtitle)

    ui_values = {}

    for group in groups:
        cols = st.columns(len(group))
        for feat, col_container in zip(group, cols):
            if feat not in train_df.columns:
                continue

            if feat in NUMERIC_INPUT_COLS:
                ui_values[feat] = render_numeric_input(feat, col_container)
            else:
                ui_values[feat] = render_categorical_input(feat, col_container)

    st.markdown("---")
    return ui_values


def build_model_input(ui_values: dict) -> pd.DataFrame:
    """Create single-row DataFrame with all columns expected by the pipeline."""
    row = {}
    for col in train_df.columns:
        if col in ui_values:
            row[col] = ui_values[col]
        else:
            row[col] = defaults.get(col, np.nan)
    return pd.DataFrame([row])


@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(model)




def build_report_text(
    inputs_dict: dict,
    prediction_label: str,
    prob_df: pd.DataFrame,
    shap_df_display: pd.DataFrame,
) -> str:
    """Build a text version of the report (used to draw on PDF)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("Diabetes Risk Report")
    lines.append("")
    lines.append(f"Generated on: {now}")
    lines.append("")
    lines.append("1. Predicted Risk")
    lines.append(f"   - Predicted class: {prediction_label}")
    lines.append("")
    lines.append("2. Class Probabilities")
    for _, row in prob_df.iterrows():
        cls = row["Class"]
        p = row["Probability"]
        lines.append(f"   - {cls}: {p:.3f}")
    lines.append("")
    lines.append("3. Patient Inputs")
    for key, val in inputs_dict.items():
        label = LABELS.get(key, key)
        lines.append(f"   - {label}: {val}")
    lines.append("")
    lines.append("4. Top Factors Influencing This Prediction")
    lines.append("   (Based on SHAP values for the Diabetes class)")
    lines.append("")
    top_shap = shap_df_display.head(10)
    for _, row in top_shap.iterrows():
        feat = row["Feature"]
        sv = row["SHAP value"]
        lines.append(f"   - {feat}: SHAP value = {sv:.3f}")
    lines.append("")
    lines.append(
        "Note: This report is generated by a machine learning model trained on BRFSS 2024 data "
        "and is for educational/demo purposes only. It is not a medical diagnosis."
    )

    return "\n".join(lines)


def build_pdf_bytes(report_text: str) -> bytes:
    """Turn the text report into a simple multi-page PDF."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x_margin = 50
    y = height - 50
    line_height = 14

    for line in report_text.split("\n"):
        if y < 50:  # new page if at bottom
            c.showPage()
            y = height - 50
        c.drawString(x_margin, y, line)
        y -= line_height

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ---------------------------------------------------
# Sidebar: project meta, links
# ---------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.write(
        "This application is developed based on the 2024 CDC Behavioral Risk Factor Surveillance System."
    )
    st.markdown("**Dataset:** [CDC BRFSS 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html)")
    st.markdown("**Tech stack:** XGBoost · SHAP · Streamlit · Python")

    # st.markdown("---")
    # st.markdown("**Links**")
    st.markdown("**GitHub:** [Behavioral Diabetes Predictor](https://github.com/CRodgb/behavioral-diabetes-predictor)")
    # st.markdown("- 📄 Paper: *add link here*")

    st.markdown("---")
    st.markdown("<p style='color: red;'>⚠️ Disclaimer: This application is designed for educational and research purposes only and should not be used for clinical decision-making.</p>", unsafe_allow_html=True)


# ---------------------------------------------------
# Main layout
# ---------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)

# 3-step workflow banner
st.markdown(
    """
<div style="
    background-color:#f8f9fa;
    border:1px solid #dee2e6;
    border-radius:0.5rem;
    padding:0.75rem 1rem;
    margin-bottom:1rem;
    text-align: center;
">
<p>An explainable multiclass Machine Learning application for diabetes risk prediction and quantify behavioral contributions. <br>
This tool estimates <b>3-class diabetes status</b> (No Diabetes / Prediabetes / Diabetes)</p>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown("---")

left, center, right = st.columns([1, 2, 1])

# ---------------------------------------------------
# Patient Profile form (center card)
# ---------------------------------------------------
with center:
    with st.container():
        st.markdown(
            """
<div style="
    border:1px solid #e0e0e0;
    border-radius:0.75rem;
    padding:1.25rem 1.5rem;
    background-color:#ffffff;
    box-shadow:0 1px 3px rgba(0,0,0,0.08);
">
<h3 style="margin-top:0;text-align:center;">🩺 Patient Profile</h3>
<p style="margin-bottom:0.5rem;">
Fill in lifestyle, body, and demographic information.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")  # small spacer

    with st.form("risk_form"):
        # 1. Behavioral
        behav_values = render_grouped_inputs(
            BEHAVIORAL_GROUPS,
            "1. 🏃 Lifestyle / Behavioral Factors",
            "Physical activity, alcohol use, smoking, and health-related days.",
        )

        # 2. Anthropometric
        anthro_values = render_grouped_inputs(
            ANTHRO_GROUPS,
            "2. ⚕️ Body & Health Profile",
            "BMI, weight, and related categories.",
        )

        # 3. Demographics
        demo_values = render_grouped_inputs(
            DEMO_GROUPS,
            "3. 👤 Demographic Context",
            "Sex, age, race/ethnicity, education, income, and state.",
        )

        all_values = {**behav_values, **anthro_values, **demo_values}

        submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

# ---------------------------------------------------
# Prediction + SHAP + PDF + Advanced mode
# ---------------------------------------------------
if submitted:
    X_user = build_model_input(all_values)

    with st.spinner("Running model..."):
        proba = pipeline.predict_proba(X_user)[0]
        pred_class = int(pipeline.predict(X_user)[0])

    risk_label = CLASS_MAP.get(pred_class, str(pred_class))

    # Risk card color
    if pred_class == 0:
        bg = "#5bc852"
        border = "#c3e6cb"
    elif pred_class == 1:
        bg = "#d6c245"
        border = "#ffeeba"
    else:
        bg = "#e2747d"
        border = "#f5c6cb"

    st.markdown("## Prediction Result")

    st.markdown(
        f"""
<div style="
    background-color:{bg};
    border:1px solid {border};
    border-radius:0.5rem;
    padding:0.75rem 1rem;
    margin-bottom:0.75rem;
">
<b>Predicted category:</b> {risk_label}<br/>
</div>
""",
        unsafe_allow_html=True,
    )

    # Class probabilities table
    prob_df = pd.DataFrame(
        {"Class": [CLASS_MAP[i] for i in range(len(proba))], "Probability": proba * 100}
    )
    st.write("**Class probabilities:**")
    st.dataframe(prob_df.style.format({"Probability": "{:.1f}%"}), width="stretch")

    # SHAP placeholder
    shap_display_df = pd.DataFrame({"Feature": [], "SHAP value": []})

    # ---------- SHAP local explanation ----------
    if preprocessor is not None and model is not None:
        X_trans = preprocessor.transform(X_user)
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(X_trans)

        # Multi-class handling: pick diabetes class index 2 if available
        if isinstance(shap_values, list):
            class_idx = 2 if len(shap_values) > 2 else len(shap_values) - 1
            sv_row = np.array(shap_values[class_idx][0])
        else:
            sv_row = np.array(shap_values[0])

        sv_row = sv_row.ravel()

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(len(sv_row))]

        feature_names = np.asarray(feature_names)
        min_len = min(len(feature_names), len(sv_row))
        feature_names = feature_names[:min_len]
        sv_row = sv_row[:min_len]

        shap_df = pd.DataFrame(
            {"Feature_raw": feature_names, "SHAP value": sv_row}
        ).sort_values("SHAP value", key=lambda s: s.abs(), ascending=False)

        shap_df["Feature"] = shap_df["Feature_raw"].apply(prettify_feature_name)
        shap_display_df = shap_df[["Feature", "SHAP value"]]

        # Basic explanation section (always visible)
        st.markdown("## Why did the model predict this?")

        top_basic = shap_display_df.head(5)
        st.write("Top factors (summary):")
        st.dataframe(
            top_basic.style.format({"SHAP value": "{:.3f}"}),
            width="stretch",
        )

        # Advanced details toggle
        with st.expander("Show advanced explanation", expanded=False):
            st.markdown("### Detailed SHAP Impact (Top 15)")
            top_15 = shap_display_df.head(15)
            st.dataframe(
                top_15.style.format({"SHAP value": "{:.3f}"}),
                width="stretch",
            )

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(top_15["Feature"][::-1], top_15["SHAP value"][::-1])
            ax.set_xlabel("SHAP value (impact on Diabetes class log-odds)")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            with st.expander("How this model works"):
                st.markdown(
                    """
- **Model:** XGBoost (multi-class)  
- **Target:** 3-class diabetes status (No diabetes / Prediabetes / Diabetes)  
- **Inputs:** Behavioral (activity, alcohol, smoking, unhealthy days),  
  anthropometric (BMI, weight), and demographic (sex, age, race, education, income, state).  
- **Training data:** Composite features engineered from BRFSS 2024 responses.  
- **Interpretability:** SHAP values show how each feature pushes the prediction 
  towards or away from the Diabetes class for this individual.
"""
                )

    else:
        st.warning(
            "Could not access internal preprocessor/model to compute SHAP explanations."
        )

    # ---------- Build PDF report ----------
    report_text = build_report_text(all_values, risk_label, prob_df, shap_display_df)
    pdf_bytes = build_pdf_bytes(report_text)

    st.markdown("## Download Report")
    st.write(
        "Download a PDF report containing your inputs, the model's prediction, "
        "class probabilities, and the top contributing factors."
    )

    st.download_button(
        label="📕 Download PDF report",
        data=pdf_bytes,
        file_name="diabetes_risk_report.pdf",
        mime="application/pdf",
    )

# ---------- Footer / About ----------
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Copyright 2025. All rights reserved.</h5>", unsafe_allow_html=True)
