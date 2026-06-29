import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a0f1e 100%);
        color: #f1f5f9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Cards */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #162032 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1rem;
    }
    .risk-card-low {
        border-left: 5px solid #10b981;
        background: linear-gradient(135deg, #0d2818 0%, #1e293b 100%);
    }
    .risk-card-high {
        border-left: 5px solid #f43f5e;
        background: linear-gradient(135deg, #2a0d1a 0%, #1e293b 100%);
    }

    /* Metric cards */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-box .value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-box .label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #334155;
    }

    /* Risk badge */
    .badge-low {
        display: inline-block;
        background: #065f46;
        color: #6ee7b7;
        border: 1px solid #10b981;
        border-radius: 24px;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .badge-high {
        display: inline-block;
        background: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #f43f5e;
        border-radius: 24px;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #334155;
        margin: 1.5rem 0;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
    }

    /* Warnings */
    .disclaimer {
        background: #1c1a0a;
        border: 1px solid #854d0e;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #fde68a;
        font-size: 0.82rem;
    }

    /* Info boxes */
    .info-box {
        background: #0a1628;
        border: 1px solid #1e40af;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #93c5fd;
        font-size: 0.85rem;
    }

    /* Probability bar container */
    .prob-bar-bg {
        background: #1e293b;
        border-radius: 999px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin: 0.5rem 0 1rem 0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "best_model_stack.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_xgb():
    path = "best_model_xgb.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()
xgb_model = load_xgb()

# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS (must match training order exactly)
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income"
]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>Powered by an ensemble Stacking Classifier trained on 248,000+ CDC BRFSS survey responses</p>
    <p style="margin-top:0.3rem; font-size:0.85rem; color:#64748b;">
        By <strong style="color:#a78bfa;">Saikat Sarkar</strong>
        &nbsp;·&nbsp; 🎓 IIT Jodhpur
        &nbsp;·&nbsp; 📊 Data Science & AI
    </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `best_model_stack.pkl` not found. Please train the model and save it first.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Patient Health Profile")
    st.markdown("---")

    # ── Cardiovascular Health ──
    st.markdown("#### 🫀 Cardiovascular Health")
    HighBP = st.radio("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    HighChol = st.radio("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    CholCheck = st.radio("Cholesterol Check (last 5 yrs)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    HeartDiseaseorAttack = st.radio("Heart Disease / Heart Attack", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    Stroke = st.radio("Ever Had a Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

    st.markdown("---")

    # ── Physical Health ──
    st.markdown("#### 🏃 Physical Health & Lifestyle")
    BMI = st.slider("BMI (Body Mass Index)", min_value=10, max_value=98, value=28, step=1)
    PhysActivity = st.radio("Physical Activity (last 30 days)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    DiffWalk = st.radio("Difficulty Walking / Climbing Stairs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    PhysHlth = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
    MentHlth = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)

    st.markdown("---")

    # ── Diet & Habits ──
    st.markdown("#### 🥦 Diet & Habits")
    Fruits = st.radio("Fruits (≥ 1/day)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    Veggies = st.radio("Vegetables (≥ 1/day)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    Smoker = st.radio("Smoker (≥ 100 cigarettes lifetime)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

    st.markdown("---")

    # ── Healthcare Access ──
    st.markdown("#### 🏥 Healthcare Access")
    AnyHealthcare = st.radio("Any Healthcare Coverage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    NoDocbcCost = st.radio("Couldn't Afford Doctor (last 12 months)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    GenHlth = st.select_slider(
        "General Health (Self-Rated)",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x]
    )

    st.markdown("---")

    # ── Demographics ──
    st.markdown("#### 👤 Demographics")
    Sex = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)
    Age = st.select_slider(
        "Age Group",
        options=list(range(1, 14)),
        value=7,
        format_func=lambda x: {
            1:"18–24", 2:"25–29", 3:"30–34", 4:"35–39",
            5:"40–44", 6:"45–49", 7:"50–54", 8:"55–59",
            9:"60–64", 10:"65–69", 11:"70–74", 12:"75–79", 13:"80+"
        }[x]
    )
    Education = st.select_slider(
        "Education Level",
        options=[1, 2, 3, 4, 5, 6],
        value=4,
        format_func=lambda x: {
            1: "No School", 2: "Elementary", 3: "Some High School",
            4: "High School Graduate", 5: "Some College", 6: "College Graduate"
        }[x]
    )
    Income = st.select_slider(
        "Household Income",
        options=list(range(1, 9)),
        value=5,
        format_func=lambda x: {
            1: "< $10K", 2: "$10–15K", 3: "$15–20K", 4: "$20–25K",
            5: "$25–35K", 6: "$35–50K", 7: "$50–75K", 8: "$75K+"
        }[x]
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem; text-align:center; padding:1rem;
                background:#0f172a; border-radius:12px; border:1px solid #1e293b;">
        <div style="font-size:0.78rem; color:#475569; margin-bottom:0.3rem;">Developed by</div>
        <div style="font-size:0.92rem; font-weight:600; color:#a78bfa;">Saikat Sarkar</div>
        <div style="font-size:0.78rem; color:#64748b;">🎓 IIT Jodhpur</div>
        <div style="font-size:0.75rem; color:#475569;">📊 Data Science & AI</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────────────────────────
input_data = pd.DataFrame([[
    HighBP, HighChol, CholCheck, BMI, Smoker,
    Stroke, HeartDiseaseorAttack, PhysActivity, Fruits,
    Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
    GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
    Education, Income
]], columns=FEATURE_COLS)


# ─────────────────────────────────────────────────────────────
# TABS LAYOUT
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Feature Analysis", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────
# TAB 1: PREDICTION
# ─────────────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_pct = probability * 100

        # ── Result card
        is_high = prediction == 1
        card_class = "risk-card-high" if is_high else "risk-card-low"
        badge_class = "badge-high" if is_high else "badge-low"
        result_text = "⚠️ Diabetes Risk Detected" if is_high else "✅ Low Diabetes Risk"
        result_color = "#f43f5e" if is_high else "#10b981"
        emoji = "🔴" if is_high else "🟢"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                <span style="font-size:2.5rem;">{emoji}</span>
                <div>
                    <div style="font-size:1.5rem; font-weight:700; color:{result_color};">{result_text}</div>
                    <span class="{badge_class}">{'HIGH RISK' if is_high else 'LOW RISK'}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-box">
                <div class="value" style="color:{'#f43f5e' if is_high else '#10b981'};">{risk_pct:.1f}%</div>
                <div class="label">Diabetes Probability</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-box">
                <div class="value">{100 - risk_pct:.1f}%</div>
                <div class="label">No Diabetes Probability</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            bmi_cat = "Underweight" if BMI < 18.5 else "Normal" if BMI < 25 else "Overweight" if BMI < 30 else "Obese"
            st.markdown(f"""<div class="metric-box">
                <div class="value" style="font-size:1.3rem;">{bmi_cat}</div>
                <div class="label">BMI Category (BMI={BMI})</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            risk_factors = sum([HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack, DiffWalk])
            st.markdown(f"""<div class="metric-box">
                <div class="value">{risk_factors}/6</div>
                <div class="label">Major Risk Factors Active</div>
            </div>""", unsafe_allow_html=True)

        # ── Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>📊 Risk Probability Breakdown</div>", unsafe_allow_html=True)
        bar_col1, bar_col2 = st.columns([3, 1])
        with bar_col1:
            bar_color = "#f43f5e" if risk_pct > 50 else "#f59e0b" if risk_pct > 25 else "#10b981"
            st.markdown(f"""
            <div class="prob-bar-bg">
                <div style="width:{risk_pct}%; height:100%; background:linear-gradient(90deg, {bar_color}80, {bar_color}); border-radius:999px; transition:width 0.6s ease;"></div>
            </div>
            """, unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            col_a.markdown(f"<span style='color:#10b981; font-weight:600;'>No Diabetes: {100-risk_pct:.1f}%</span>", unsafe_allow_html=True)
            col_b.markdown(f"<span style='color:#f43f5e; font-weight:600; float:right;'>Diabetes: {risk_pct:.1f}%</span>", unsafe_allow_html=True)

        # ── Gauge chart
        with bar_col2:
            fig_g, ax_g = plt.subplots(figsize=(3, 1.5), subplot_kw=dict(projection='polar'))
            fig_g.patch.set_facecolor('#0f172a')
            theta = np.linspace(np.pi, 0, 200)
            for i, (t_start, t_end, color) in enumerate([(np.pi, np.pi*2/3, '#10b981'), (np.pi*2/3, np.pi/3, '#f59e0b'), (np.pi/3, 0, '#f43f5e')]):
                ax_g.barh(1, t_start - t_end, left=t_end, height=0.5, color=color, alpha=0.8)
            needle = np.pi - (probability * np.pi)
            ax_g.annotate('', xy=(needle, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
            ax_g.set_yticks([])
            ax_g.set_xticks([])
            ax_g.spines['polar'].set_visible(False)
            ax_g.set_facecolor('#0f172a')
            ax_g.set_ylim(0, 1.5)
            ax_g.set_xlim(0, np.pi)
            st.pyplot(fig_g, use_container_width=True)
            plt.close()

        # ── Key active risk factors
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🚨 Active Risk Factors</div>", unsafe_allow_html=True)
        risk_dict = {
            "High Blood Pressure": HighBP, "High Cholesterol": HighChol,
            "Smoker": Smoker, "Stroke History": Stroke,
            "Heart Disease/Attack": HeartDiseaseorAttack, "Difficulty Walking": DiffWalk,
            "No Healthcare Coverage": 1 - AnyHealthcare, "Couldn't Afford Doctor": NoDocbcCost,
            "Heavy Alcohol Use": HvyAlcoholConsump, "Poor General Health (Fair/Poor)": 1 if GenHlth >= 4 else 0,
            "High BMI (≥30)": 1 if BMI >= 30 else 0
        }
        active = [k for k, v in risk_dict.items() if v == 1]
        inactive = [k for k, v in risk_dict.items() if v == 0]
        rf_col1, rf_col2 = st.columns(2)
        with rf_col1:
            st.markdown("**⚠️ Present:**")
            for r in active:
                st.markdown(f"<span style='color:#f43f5e;'>● {r}</span>", unsafe_allow_html=True)
            if not active:
                st.markdown("<span style='color:#10b981;'>None detected</span>", unsafe_allow_html=True)
        with rf_col2:
            st.markdown("**✅ Not Present:**")
            for r in inactive[:6]:
                st.markdown(f"<span style='color:#10b981;'>● {r}</span>", unsafe_allow_html=True)

        # ── Disclaimer
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="disclaimer">
            ⚠️ <b>Medical Disclaimer:</b> This tool is for educational and research purposes only.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare provider for medical concerns.
        </div>""", unsafe_allow_html=True)

    else:
        # Pre-prediction state
        st.markdown("""
        <div class="result-card" style="text-align:center; padding:3rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">🩺</div>
            <div style="font-size:1.3rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem;">Ready to Predict</div>
            <div style="color:#64748b;">Fill in your health profile in the sidebar and click <b>Predict Diabetes Risk</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Show feature info cards
        st.markdown("<div class='section-title'>📋 Input Summary</div>", unsafe_allow_html=True)
        st.dataframe(
            input_data.T.rename(columns={0: "Your Value"}),
            use_container_width=True,
            height=400
        )


# ─────────────────────────────────────────────────────────────
# TAB 2: FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>🌲 Model Feature Importances (XGBoost)</div>", unsafe_allow_html=True)

    if xgb_model is not None:
        importances = xgb_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        fig_fi.patch.set_facecolor('#0f172a')
        ax_fi.set_facecolor('#0f172a')

        colors = ['#4f46e5' if f not in ['HighBP', 'BMI', 'GenHlth', 'Age', 'HighChol', 'DiffWalk']
                  else '#f43f5e' for f in fi_df['Feature']]
        bars = ax_fi.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='#0f172a', height=0.7)

        for bar, val in zip(bars, fi_df['Importance']):
            ax_fi.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', ha='left', color='#94a3b8', fontsize=8)

        ax_fi.set_xlabel('Feature Importance Score', color='#94a3b8', fontsize=11)
        ax_fi.set_title('Top Feature Importances — XGBoost Base Estimator', color='#f1f5f9', fontsize=13, fontweight='bold', pad=15)
        ax_fi.tick_params(colors='#94a3b8')
        ax_fi.spines['top'].set_visible(False)
        ax_fi.spines['right'].set_visible(False)
        for spine in ['bottom', 'left']:
            ax_fi.spines[spine].set_color('#334155')
        ax_fi.set_facecolor('#0f172a')

        red_patch = mpatches.Patch(color='#f43f5e', label='Top Clinical Predictors')
        blue_patch = mpatches.Patch(color='#4f46e5', label='Other Features')
        ax_fi.legend(handles=[red_patch, blue_patch], frameon=False,
                     labelcolor='#94a3b8', fontsize=9, loc='lower right')

        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📋 Full Feature Importance Table</div>", unsafe_allow_html=True)
        full_fi = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances})
        full_fi = full_fi.sort_values('Importance', ascending=False).reset_index(drop=True)
        full_fi['Importance'] = full_fi['Importance'].round(5)
        full_fi.index = full_fi.index + 1
        st.dataframe(full_fi, use_container_width=True)
    else:
        st.markdown("""<div class="info-box">
            ℹ️ XGBoost model file (<code>best_model_xgb.pkl</code>) not found in the current directory.
            Feature importance chart will appear once the model file is available.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Your Input vs. Population Risk Factors</div>", unsafe_allow_html=True)

    # Radar-style input visualization
    binary_features = {
        "HighBP": HighBP, "HighChol": HighChol, "Smoker": Smoker,
        "Stroke": Stroke, "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
        "HvyAlcohol": HvyAlcoholConsump, "DiffWalk": DiffWalk
    }
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    fig_bar.patch.set_facecolor('#0f172a')
    ax_bar.set_facecolor('#0f172a')
    risk_features = ["HighBP", "HighChol", "Smoker", "Stroke", "HeartDiseaseorAttack", "HvyAlcohol", "DiffWalk"]
    protect_features = ["PhysActivity", "Fruits", "Veggies"]
    bar_labels = list(binary_features.keys())
    bar_values = list(binary_features.values())
    bar_colors = []
    for label, val in zip(bar_labels, bar_values):
        if label in risk_features:
            bar_colors.append('#f43f5e' if val == 1 else '#334155')
        else:
            bar_colors.append('#10b981' if val == 1 else '#334155')
    ax_bar.bar(bar_labels, bar_values, color=bar_colors, edgecolor='#0f172a', width=0.6)
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels(['No', 'Yes'], color='#94a3b8')
    ax_bar.tick_params(axis='x', colors='#94a3b8', rotation=30)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    for spine in ['bottom', 'left']:
        ax_bar.spines[spine].set_color('#334155')
    ax_bar.set_title("Your Binary Health Indicators", color='#f1f5f9', fontweight='bold', pad=12)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────────────────────
# TAB 3: ABOUT
# ─────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="result-card">
            <div class="section-title">🤖 About the Model</div>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
            The prediction engine is a <b style="color:#e2e8f0;">Stacking Classifier</b> — an ensemble of 5 base models
            combined by a Logistic Regression meta-learner:
            </p>
            <ul style="color:#94a3b8; font-size:0.88rem; line-height:2;">
                <li>🌳 Decision Tree</li>
                <li>🌲 Random Forest</li>
                <li>⚡ XGBoost</li>
                <li>⚡ Linear SVM</li>
                <li>🚀 Gradient Boosting</li>
            </ul>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.7;">
            All models were tuned using <b style="color:#e2e8f0;">GridSearchCV with Stratified K-Fold</b>,
            optimizing for <b style="color:#e2e8f0;">Recall</b> on the Diabetes class to minimize missed diagnoses.
            Class imbalance was handled using <b style="color:#e2e8f0;">SMOTE</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="result-card">
            <div class="section-title">📊 Dataset Information</div>
            <table style="width:100%; color:#94a3b8; font-size:0.88rem; border-collapse:collapse;">
                <tr><td style="padding:6px 0; color:#64748b;">Source</td><td style="color:#e2e8f0;">CDC BRFSS 2015 Survey</td></tr>
                <tr><td style="padding:6px 0; color:#64748b;">Records</td><td style="color:#e2e8f0;">253,680 respondents</td></tr>
                <tr><td style="padding:6px 0; color:#64748b;">Features</td><td style="color:#e2e8f0;">21 health indicators</td></tr>
                <tr><td style="padding:6px 0; color:#64748b;">Task</td><td style="color:#e2e8f0;">Binary Classification</td></tr>
                <tr><td style="padding:6px 0; color:#64748b;">Imbalance Handling</td><td style="color:#e2e8f0;">SMOTE (train set only)</td></tr>
                <tr><td style="padding:6px 0; color:#64748b;">Primary Metric</td><td style="color:#e2e8f0;">Recall (Class 1)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card">
        <div class="section-title">⚠️ Limitations & Disclaimer</div>
        <ul style="color:#94a3b8; font-size:0.88rem; line-height:2;">
            <li><b style="color:#e2e8f0;">Self-Report Bias:</b> Dataset is based on telephone survey responses — may underreport sensitive behaviors</li>
            <li><b style="color:#e2e8f0;">US Population Only:</b> Model was trained on US adults; may not generalize internationally</li>
            <li><b style="color:#e2e8f0;">No Lab Data:</b> Model uses only behavioral/demographic indicators — not clinical measurements like blood glucose</li>
            <li><b style="color:#e2e8f0;">Screening Tool Only:</b> This is NOT a diagnostic tool. Always consult a licensed medical professional.</li>
        </ul>
    </div>

    <div class="disclaimer" style="margin-top:1rem;">
        ⚕️ This application is intended for <b>educational and research purposes only</b>.
        It does not constitute medical advice and should not be used as a substitute for professional medical consultation.
        Always seek the advice of your physician or qualified healthcare provider.
    </div>

    <div class="result-card" style="margin-top:1.5rem; text-align:center; background:linear-gradient(135deg,#1a0a2e,#0f172a); border:1px solid #4f46e5;">
        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">👨‍💻 Developed by</div>
        <div style="font-size:1.3rem; font-weight:700; background:linear-gradient(90deg,#a78bfa,#60a5fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Saikat Sarkar</div>
        <div style="color:#94a3b8; font-size:0.9rem; margin-top:0.3rem;">🎓 Indian Institute of Technology, Jodhpur</div>
        <div style="color:#64748b; font-size:0.85rem; margin-top:0.2rem;">📊 Data Science & Artificial Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
