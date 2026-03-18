import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from imblearn.over_sampling import SMOTE
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank · Personal Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #1a2f5e 0%, #2563eb 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
  }
  .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
  .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

  .metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .metric-card .label { font-size: 0.75rem; font-weight: 600; color: #6b7280;
                         text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #111827; }
  .metric-card .sub   { font-size: 0.8rem; color: #9ca3af; margin-top: 2px; }

  .insight-box {
    background: #f0f7ff;
    border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #1e3a5f;
    line-height: 1.55;
    margin-top: 0.5rem;
  }

  .section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }

  .badge-good { background:#dcfce7; color:#166534; padding:3px 10px;
                border-radius:999px; font-size:0.75rem; font-weight:600; }
  .badge-warn { background:#fef9c3; color:#854d0e; padding:3px 10px;
                border-radius:999px; font-size:0.75rem; font-weight:600; }
  .badge-bad  { background:#fee2e2; color:#991b1b; padding:3px 10px;
                border-radius:999px; font-size:0.75rem; font-weight:600; }

  .stTab [data-baseweb="tab"] { font-size:0.9rem; font-weight:500; }
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  .prescriptive-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border-radius: 12px; padding: 1.2rem 1.5rem;
  }
  .prescriptive-card h4 { margin: 0 0 0.5rem; font-size: 1rem; }
  .prescriptive-card p  { margin: 0; font-size: 0.82rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE    = "#2563eb"
TEAL    = "#0d9488"
AMBER   = "#f59e0b"
CORAL   = "#ef4444"
PURPLE  = "#7c3aed"
GREEN   = "#16a34a"
GREY    = "#6b7280"
PALETTE = [BLUE, TEAL, AMBER, CORAL, PURPLE, GREEN, GREY,
           "#f97316","#06b6d4","#8b5cf6"]

# ── Data loading & preprocessing ───────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("UniversalBank.csv")
    df.columns = df.columns.str.strip()

    # Fix negative experience
    df["Experience"] = df["Experience"].clip(lower=0)

    # Drop non-predictive columns
    df_model = df.drop(columns=["ID", "ZIP Code"], errors="ignore")

    # Feature matrix & target
    X = df_model.drop(columns=["Personal Loan"])
    y = df_model["Personal Loan"]

    return df, df_model, X, y

@st.cache_data
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE on training set only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_sc, y_train)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42),
    }

    results   = {}
    trained   = {}
    for name, clf in models.items():
        clf.fit(X_res, y_res)
        trained[name] = clf

        y_pred      = clf.predict(X_test_sc)
        y_prob      = clf.predict_proba(X_test_sc)[:, 1]
        y_pred_tr   = clf.predict(X_res)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)

        results[name] = {
            "clf":          clf,
            "y_test":       y_test,
            "y_pred":       y_pred,
            "y_prob":       y_prob,
            "y_pred_train": y_pred_tr,
            "y_train":      y_res,
            "fpr":          fpr,
            "tpr":          tpr,
            "roc_auc":      roc_auc,
            "cm":           confusion_matrix(y_test, y_pred),
        }

    return results, scaler, X_train, X_test, y_train, y_test, X_res, y_res, trained

df, df_model, X, y = load_and_preprocess()
results, scaler, X_train, X_test, y_train, y_test, X_res, y_res, trained_models = train_models(X, y)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("### 🏦 Universal Bank")
    st.markdown("**Personal Loan Intelligence**")
    st.markdown("---")
    st.markdown("#### Navigation")
    st.markdown("""
- 📋 Data Overview
- 📊 Descriptive Analytics
- 🔍 Exploratory Analysis
- 🤖 ML Models
- 🎯 Prescriptive Analytics
- 🔮 Predict New Data
    """)
    st.markdown("---")
    st.markdown("#### Dataset Info")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Features:** {X.shape[1]}")
    st.markdown(f"**Loan Acceptance:** 9.6%")
    st.markdown(f"**Class Imbalance:** 9.4 : 1")
    st.markdown("---")
    st.caption("Built for Universal Bank Marketing · 2024")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 Universal Bank — Personal Loan Intelligence Dashboard</h1>
  <p>From descriptive analytics to AI-powered prescriptive targeting · Empowering hyper-personalised campaign decisions</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Data Overview",
    "📊 Descriptive Analytics",
    "🔍 Exploratory Analysis",
    "🤖 ML Models & Performance",
    "🎯 Prescriptive Analytics",
    "🔮 Predict New Data",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Dataset at a Glance</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Total Records",    f"{len(df):,}",   "Clean, no missing values"),
        (c2, "Features Used",    f"{X.shape[1]}",  "After dropping ID & ZIP"),
        (c3, "Loan Accepted",    "480 (9.6%)",     "Positive class"),
        (c4, "Avg. Income",      "$73.8K",         "Range: $8K – $224K"),
        (c5, "Avg. Age",         "45.3 yrs",       "Range: 23 – 67"),
    ]
    for col, label, val, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{val}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Raw Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=380)

    with col_r:
        st.markdown('<div class="section-header">Feature Dictionary</div>', unsafe_allow_html=True)
        feat_dict = pd.DataFrame({
            "Feature":     ["Age","Experience","Income","Family","CCAvg",
                            "Education","Mortgage","Securities Acct","CD Account",
                            "Online","CreditCard","Personal Loan"],
            "Type":        ["Continuous","Continuous","Continuous","Categorical","Continuous",
                            "Categorical","Continuous","Binary","Binary",
                            "Binary","Binary","Target (Binary)"],
            "Description": [
                "Customer age (years)",
                "Years of professional experience",
                "Annual income ($000)",
                "Family size (1–4)",
                "Avg monthly CC spend ($000)",
                "1=Undergrad  2=Graduate  3=Advanced",
                "Mortgage value ($000)",
                "Has securities account?",
                "Has CD account?",
                "Uses internet banking?",
                "Uses bank-issued credit card?",
                "Accepted personal loan offer?",
            ],
        })
        st.dataframe(feat_dict, use_container_width=True, height=380, hide_index=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df_model.describe().round(2), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>Data Quality Note:</b> 52 records had negative Experience values (data entry errors) —
    these have been clipped to 0. ZIP Code and ID were dropped as they carry no predictive signal.
    Class imbalance (9.4:1) was handled using SMOTE oversampling on the training set only to
    prevent data leakage.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Target Variable Distribution</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=["Did Not Accept (0)", "Accepted Loan (1)"],
            values=[4520, 480],
            hole=0.55,
            marker_colors=[BLUE, CORAL],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig_pie.update_layout(
            title=dict(text="Personal Loan — Class Split", font_size=15, x=0.5),
            showlegend=True, height=320,
            margin=dict(t=50, b=10, l=10, r=10),
            annotations=[dict(text="5,000<br>Records", x=0.5, y=0.5,
                              font_size=14, showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("""<div class="insight-box">
        The dataset is heavily imbalanced — only 9.6% of customers accepted the personal loan.
        This is realistic for banking campaigns but requires SMOTE resampling and F1/AUC metrics
        rather than plain accuracy.</div>""", unsafe_allow_html=True)

    with c2:
        cont_cols = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
        fig_dist = make_subplots(rows=2, cols=3,
                                  subplot_titles=[f"Distribution of {c}" for c in cont_cols] + [""],
                                  vertical_spacing=0.18, horizontal_spacing=0.1)
        row_col = [(1,1),(1,2),(1,3),(2,1),(2,2)]
        colors_d = [BLUE, TEAL, AMBER, CORAL, PURPLE]
        for i, (col_name, (r, c), color) in enumerate(zip(cont_cols, row_col, colors_d)):
            vals = df[col_name]
            fig_dist.add_trace(go.Histogram(
                x=vals, name=col_name, marker_color=color, opacity=0.85,
                nbinsx=30, showlegend=False
            ), row=r, col=c)
        fig_dist.update_layout(height=420, title_text="Continuous Feature Distributions",
                                title_x=0.5, margin=dict(t=60, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("""<div class="insight-box">
        Income and CCAvg are right-skewed — a small segment of high earners dominates.
        Mortgage shows a large zero-spike (69% have no mortgage). These distributions
        guide feature scaling and outlier handling before model training.</div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Categorical Feature Distributions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cat_plots = [
        (c1, "Education", {1:"Undergrad", 2:"Graduate", 3:"Advanced"},
         "Education level is split fairly evenly. Graduate & Advanced customers have 3× higher loan uptake than Undergrads."),
        (c2, "Family",    {1:"1 member", 2:"2 members", 3:"3 members", 4:"4 members"},
         "Larger families (3–4) tend to have higher loan acceptance — likely driven by greater financial need."),
        (c3, "Online",    {0:"No Online", 1:"Uses Online Banking"},
         "60% use online banking. Digitally engaged customers are easier to reach through targeted digital campaigns."),
        (c4, "CreditCard",{0:"No CC", 1:"Has Bank CC"},
         "Only 29% use a bank-issued credit card — an untapped cross-sell base for personal loans."),
    ]
    for col, feat, label_map, insight in cat_plots:
        with col:
            vc = df[feat].map(label_map).value_counts().reset_index()
            vc.columns = [feat, "Count"]
            vc["Pct"] = (vc["Count"] / vc["Count"].sum() * 100).round(1)
            vc["Label"] = vc["Count"].astype(str) + " (" + vc["Pct"].astype(str) + "%)"
            fig = px.bar(vc, x=feat, y="Count", text="Label",
                         color=feat, color_discrete_sequence=PALETTE,
                         title=f"{feat} Distribution")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=300,
                               margin=dict(t=40, b=10, l=10, r=10),
                               xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Conversion Rate by Key Drivers</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Income quartile analysis
    with c1:
        df_tmp = df.copy()
        df_tmp["Income Quartile"] = pd.qcut(df_tmp["Income"], 4,
            labels=["Q1: <$39K", "Q2: $39–64K", "Q3: $64–98K", "Q4: >$98K"])
        conv_inc = df_tmp.groupby("Income Quartile", observed=True)["Personal Loan"]\
                         .agg(["mean","sum","count"]).reset_index()
        conv_inc.columns = ["Income Quartile","Rate","Accepted","Total"]
        conv_inc["Rate_pct"] = (conv_inc["Rate"] * 100).round(1)
        conv_inc["Label"] = conv_inc["Rate_pct"].astype(str) + "%"

        fig = px.bar(conv_inc, x="Income Quartile", y="Rate_pct",
                     text="Label", color="Rate_pct",
                     color_continuous_scale=[[0,BLUE],[0.5,AMBER],[1,CORAL]],
                     title="Loan Acceptance Rate by Income Quartile")
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=350,
                           yaxis_title="Acceptance Rate (%)",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="insight-box">
        Income is the #1 predictor. Top-quartile earners (>$98K) convert at <b>35.6%</b> vs
        near-zero for low-income customers. Marketing spend concentrated on Q4 income segment
        will yield the highest ROI.</div>""", unsafe_allow_html=True)

    with c2:
        edu_map = {1:"Undergrad", 2:"Graduate", 3:"Advanced"}
        df_tmp2 = df.copy()
        df_tmp2["Education Label"] = df_tmp2["Education"].map(edu_map)
        conv_edu = df_tmp2.groupby("Education Label")["Personal Loan"]\
                          .agg(["mean","sum","count"]).reset_index()
        conv_edu.columns = ["Education","Rate","Accepted","Total"]
        conv_edu["Rate_pct"] = (conv_edu["Rate"] * 100).round(1)
        conv_edu["Label"] = conv_edu["Rate_pct"].astype(str) + "%\n(" + conv_edu["Accepted"].astype(str) + " accepted)"

        fig = px.bar(conv_edu, x="Education", y="Rate_pct",
                     text="Rate_pct", color="Education",
                     color_discrete_sequence=[BLUE, TEAL, PURPLE],
                     title="Loan Acceptance Rate by Education Level")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=350,
                           yaxis_title="Acceptance Rate (%)",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="insight-box">
        Graduate and Advanced-degree customers convert at <b>3× the rate</b> of Undergrads (13% vs 4.4%).
        Campaign messaging should be segmented — highlight career-growth financing for graduates
        and premium loan products for advanced-degree professionals.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # CD Account cross-tab
        cd_map = {0:"No CD Account", 1:"Has CD Account"}
        df_tmp3 = df.copy()
        df_tmp3["CD Account Label"] = df_tmp3["CD Account"].map(cd_map)
        conv_cd = df_tmp3.groupby("CD Account Label")["Personal Loan"]\
                         .agg(["mean","sum","count"]).reset_index()
        conv_cd.columns = ["CD Account","Rate","Accepted","Total"]
        conv_cd["Rate_pct"] = (conv_cd["Rate"] * 100).round(1)

        fig = px.bar(conv_cd, x="CD Account", y="Rate_pct",
                     text="Rate_pct", color="CD Account",
                     color_discrete_sequence=[GREY, TEAL],
                     title="Loan Acceptance Rate by CD Account Status")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=330,
                           yaxis_title="Acceptance Rate (%)",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="insight-box">
        CD Account holders convert at a staggering <b>46.4%</b> — nearly 5× the average.
        They already trust the bank with savings products, making them the highest-value
        segment for a personal loan campaign.</div>""", unsafe_allow_html=True)

    with c2:
        # Family size analysis
        conv_fam = df.groupby("Family")["Personal Loan"]\
                     .agg(["mean","sum","count"]).reset_index()
        conv_fam.columns = ["Family Size","Rate","Accepted","Total"]
        conv_fam["Rate_pct"] = (conv_fam["Rate"] * 100).round(1)
        conv_fam["Family Label"] = conv_fam["Family Size"].apply(
            lambda x: f"{x} member{'s' if x>1 else ''}")

        fig = px.bar(conv_fam, x="Family Label", y="Rate_pct",
                     text="Rate_pct", color="Rate_pct",
                     color_continuous_scale=[[0, BLUE],[1, AMBER]],
                     title="Loan Acceptance Rate by Family Size")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=330,
                           yaxis_title="Acceptance Rate (%)",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="insight-box">
        Families of 3–4 show notably higher loan uptake. Larger households likely face
        greater financial demands (education, home renovation, medical expenses) — tailor
        campaign messaging around these life-stage needs.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Income vs CCAvg — Loan Acceptance Scatter</div>',
                unsafe_allow_html=True)

    df_scatter = df.copy()
    df_scatter["Loan Status"] = df_scatter["Personal Loan"].map({0:"No Loan", 1:"Accepted Loan"})
    fig_sc = px.scatter(
        df_scatter.sample(2000, random_state=1),
        x="Income", y="CCAvg",
        color="Loan Status",
        color_discrete_map={"No Loan": BLUE, "Accepted Loan": CORAL},
        opacity=0.55, size_max=6,
        title="Income vs Credit Card Avg Spend — Coloured by Loan Acceptance",
        labels={"Income":"Annual Income ($000)", "CCAvg":"Avg Monthly CC Spend ($000)"},
    )
    fig_sc.update_layout(height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("""<div class="insight-box">
    Loan acceptors (coral) cluster in the <b>high-income + high CC spend</b> quadrant.
    This confirms that customers who earn more and spend more on credit cards are prime targets.
    Customers in the lower-left quadrant (low income, low spend) are very unlikely to accept
    — avoid wasting marketing budget here.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df_model.corr().round(2)
    fig_heat = px.imshow(corr, color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto",
                          title="Feature Correlation Matrix",
                          text_auto=True)
    fig_heat.update_layout(height=500, margin=dict(t=60, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("""<div class="insight-box">
    Income has the strongest positive correlation with Personal Loan (0.50), followed by
    CCAvg (0.37) and CD Account (0.32). Age and Experience are nearly perfectly correlated
    (0.99) — suggesting we could drop one without losing information. Mortgage shows weak
    but positive correlation, while Online/CreditCard show very weak signals.</div>""",
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — ML MODELS & PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

    # Build metrics table
    rows = []
    for name, res in results.items():
        y_tr = res["y_train"]
        yp_tr = res["y_pred_train"]
        rows.append({
            "Model":             name,
            "Train Accuracy":    f"{accuracy_score(y_tr, yp_tr)*100:.2f}%",
            "Test Accuracy":     f"{accuracy_score(res['y_test'], res['y_pred'])*100:.2f}%",
            "Precision":         f"{precision_score(res['y_test'], res['y_pred'])*100:.2f}%",
            "Recall":            f"{recall_score(res['y_test'], res['y_pred'])*100:.2f}%",
            "F1 Score":          f"{f1_score(res['y_test'], res['y_pred'])*100:.2f}%",
            "ROC-AUC":           f"{res['roc_auc']:.4f}",
        })

    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.style.set_properties(**{"text-align":"center"})
                        .set_table_styles([{"selector":"th","props":[("text-align","center")]}]),
        use_container_width=True, hide_index=True
    )
    st.markdown("""<div class="insight-box">
    All three models are evaluated on a held-out 20% test set. SMOTE oversampling was applied
    only to the training data to handle class imbalance (9.4:1 ratio). F1 Score and ROC-AUC
    are the primary metrics — they account for imbalance unlike plain accuracy.</div>""",
    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Single combined ROC curve
    st.markdown('<div class="section-header">Combined ROC Curve — All Models</div>',
                unsafe_allow_html=True)

    model_colors = {
        "Decision Tree":       AMBER,
        "Random Forest":       TEAL,
        "Gradient Boosted Tree": CORAL,
    }

    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dash", color=GREY, width=1.5))

    for name, res in results.items():
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"],
            mode="lines",
            name=f"{name} (AUC = {res['roc_auc']:.4f})",
            line=dict(color=model_colors[name], width=2.5),
        ))

    fig_roc.update_layout(
        title=dict(text="ROC Curve Comparison — Decision Tree vs Random Forest vs Gradient Boosted Tree",
                   font_size=15, x=0.5),
        xaxis=dict(title="False Positive Rate", tickformat=".0%"),
        yaxis=dict(title="True Positive Rate", tickformat=".0%"),
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=GREY, borderwidth=1),
        height=480,
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        xaxis_showgrid=True, xaxis_gridcolor="#f0f0f0",
        yaxis_showgrid=True, yaxis_gridcolor="#f0f0f0",
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown("""<div class="insight-box">
    The ROC curve plots True Positive Rate vs False Positive Rate across all classification
    thresholds. A curve hugging the top-left corner indicates near-perfect discrimination.
    Higher AUC = better ability to distinguish loan-acceptors from non-acceptors. The model
    with the highest AUC is recommended for campaign targeting.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrices
    st.markdown('<div class="section-header">Confusion Matrices — All Models</div>',
                unsafe_allow_html=True)

    col_dt, col_rf, col_gb = st.columns(3)

    def plot_cm(name, res, col):
        cm = res["cm"]
        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()

        labels_text = [
            [f"TN\n{tn}\n({tn/total*100:.1f}%)", f"FP\n{fp}\n({fp/total*100:.1f}%)"],
            [f"FN\n{fn}\n({fn/total*100:.1f}%)", f"TP\n{tp}\n({tp/total*100:.1f}%)"],
        ]

        fig = px.imshow(
            cm,
            text_auto=False,
            color_continuous_scale=[[0,"#EFF6FF"],[0.5,"#BFDBFE"],[1,"#1D4ED8"]],
            labels=dict(x="Predicted Label", y="Actual Label"),
            x=["Predicted: No Loan (0)", "Predicted: Loan (1)"],
            y=["Actual: No Loan (0)", "Actual: Loan (1)"],
            title=f"{name}",
            aspect="equal",
        )

        for i in range(2):
            for j in range(2):
                fig.add_annotation(
                    x=j, y=i,
                    text=labels_text[i][j],
                    showarrow=False,
                    font=dict(size=12, color="white" if cm[i,j] > cm.max()*0.5 else "#1e3a5f",
                              family="Inter"),
                    align="center",
                )

        fig.update_layout(
            height=340,
            margin=dict(t=50, b=40, l=60, r=10),
            coloraxis_showscale=False,
            xaxis_tickangle=-15,
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)
            prec = precision_score(res["y_test"], res["y_pred"])
            rec  = recall_score(res["y_test"], res["y_pred"])
            f1   = f1_score(res["y_test"], res["y_pred"])
            st.markdown(f"""<div class="insight-box">
            <b>Precision:</b> {prec*100:.1f}% &nbsp;|&nbsp;
            <b>Recall:</b> {rec*100:.1f}% &nbsp;|&nbsp;
            <b>F1:</b> {f1*100:.1f}%<br>
            TP={tp} · FP={fp} · FN={fn} · TN={tn}
            </div>""", unsafe_allow_html=True)

    plot_cm("Decision Tree",        results["Decision Tree"],        col_dt)
    plot_cm("Random Forest",        results["Random Forest"],        col_rf)
    plot_cm("Gradient Boosted Tree",results["Gradient Boosted Tree"],col_gb)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance — Random Forest & Gradient Boosted Tree</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, mname in [(c1, "Random Forest"), (c2, "Gradient Boosted Tree")]:
        clf  = trained_models[mname]
        imp  = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
        imp  = imp.sort_values("Importance", ascending=True)
        imp["Importance_pct"] = (imp["Importance"] * 100).round(2)

        fig = px.bar(imp, x="Importance_pct", y="Feature",
                     orientation="h", text="Importance_pct",
                     color="Importance_pct",
                     color_continuous_scale=[[0, BLUE],[1, CORAL]],
                     title=f"{mname} — Feature Importances (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=380,
                           xaxis_title="Importance (%)",
                           margin=dict(t=50, b=10, r=60))
        with col:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    Feature importance scores reveal which variables drive predictions most.
    <b>Income</b> consistently ranks #1, followed by <b>CCAvg</b>, <b>CD Account</b>, and
    <b>Education</b>. These four features should anchor your campaign targeting criteria.
    Low-importance features (Online, CreditCard binary flags) add noise — consider
    excluding them in a simplified scoring model.</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — PRESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🎯 Campaign Targeting Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="insight-box" style="margin-bottom:1rem;">
    Prescriptive analytics goes beyond <i>what happened</i> and <i>why</i> — it recommends
    <b>what to do next</b>. The segments below are ranked by conversion probability and ROI
    potential to help you deploy your reduced marketing budget most effectively.</div>""",
    unsafe_allow_html=True)

    # Segment profiling
    df_seg = df.copy()
    df_seg["Segment"] = "General"
    df_seg.loc[(df_seg["Income"] > 98) & (df_seg["CD Account"] == 1),
               "Segment"] = "Platinum: High-Income CD Holders"
    df_seg.loc[(df_seg["Income"] > 98) & (df_seg["CD Account"] == 0) &
               (df_seg["Education"].isin([2,3])), "Segment"] = "Gold: High-Income Graduates"
    df_seg.loc[(df_seg["Income"].between(64,98)) & (df_seg["CCAvg"] > 2),
               "Segment"] = "Silver: Mid-Income High Spenders"
    df_seg.loc[df_seg["Segment"] == "General", "Segment"] = "Bronze: Nurture Base"

    seg_summary = df_seg.groupby("Segment").agg(
        Count=("Personal Loan","count"),
        Accepted=("Personal Loan","sum"),
    ).reset_index()
    seg_summary["Conversion Rate"] = (seg_summary["Accepted"] / seg_summary["Count"] * 100).round(1)
    seg_summary["% of Database"] = (seg_summary["Count"] / len(df_seg) * 100).round(1)
    seg_summary = seg_summary.sort_values("Conversion Rate", ascending=False)

    fig_seg = px.bar(seg_summary, x="Segment", y="Conversion Rate",
                     text="Conversion Rate", color="Conversion Rate",
                     color_continuous_scale=[[0, BLUE],[0.5, AMBER],[1, GREEN]],
                     title="Predicted Loan Conversion Rate by Campaign Segment")
    fig_seg.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_seg.update_layout(coloraxis_showscale=False, height=380,
                           xaxis_tickangle=-15,
                           yaxis_title="Conversion Rate (%)",
                           margin=dict(t=60, b=80))
    st.plotly_chart(fig_seg, use_container_width=True)

    st.dataframe(seg_summary.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Segment Action Playbook</div>', unsafe_allow_html=True)

    action_data = [
        ("🥇 Platinum", "High-Income + CD Account", "~46–60%",
         "Priority 1 — Personalised outreach via relationship manager. Offer competitive rate + loyalty reward.",
         "Top 3% of database. Highest ROI per contact."),
        ("🥈 Gold", "High-Income + Graduate/Advanced", "~25–40%",
         "Priority 2 — Digital + email campaign. Highlight career-growth financing and tax benefits.",
         "Educate on loan-to-income ratio advantages."),
        ("🥉 Silver", "Mid-Income + High CC Spend", "~10–20%",
         "Priority 3 — Targeted email with consolidation messaging. Position as credit card debt consolidation.",
         "CC spend signals creditworthiness and need."),
        ("⬜ Bronze", "All Other Customers", "<5%",
         "Nurture track only — low-cost touchpoints (app notification, newsletter). Do NOT invest heavy budget.",
         "Reserve budget for higher-yield segments."),
    ]

    for tier, profile, conv, action, note in action_data:
        c1, c2, c3 = st.columns([1.5, 3, 2])
        with c1:
            st.markdown(f"**{tier}**<br><span style='font-size:0.8rem;color:#6b7280;'>{profile}</span>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Action:** {action}<br><i style='font-size:0.8rem;color:#6b7280;'>{note}</i>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Est. Conversion Rate**<br><span style='font-size:1.5rem;font-weight:700;color:#2563eb;'>{conv}</span>",
                        unsafe_allow_html=True)
        st.divider()

    st.markdown('<div class="section-header">Budget Allocation Recommendation</div>',
                unsafe_allow_html=True)

    budget_data = {
        "Segment":        ["Platinum","Gold","Silver","Bronze"],
        "Recommended %":  [40, 35, 20, 5],
        "Est. Returns":   ["Highest","High","Moderate","Minimal"],
    }
    budget_df = pd.DataFrame(budget_data)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(budget_df, use_container_width=True, hide_index=True)
        st.markdown("""<div class="insight-box">
        With a halved budget, concentrate 95% on the top 3 segments. Bronze segment
        should only receive no-cost digital nudges.</div>""", unsafe_allow_html=True)
    with c2:
        fig_budget = px.pie(
            budget_df, names="Segment", values="Recommended %",
            color_discrete_sequence=[GREEN, TEAL, AMBER, GREY],
            title="Recommended Budget Allocation by Segment",
            hole=0.4,
        )
        fig_budget.update_traces(textinfo="label+percent")
        fig_budget.update_layout(height=320, margin=dict(t=50, b=10))
        st.plotly_chart(fig_budget, use_container_width=True)

    # Best combination heatmap
    st.markdown('<div class="section-header">Conversion Heatmap — Income × Education</div>',
                unsafe_allow_html=True)

    df_heat = df.copy()
    df_heat["Income Band"] = pd.cut(df_heat["Income"],
        bins=[0,39,64,98,300],
        labels=["<$39K","$39–64K","$64–98K",">$98K"])
    df_heat["Education Label"] = df_heat["Education"].map(
        {1:"Undergrad",2:"Graduate",3:"Advanced"})

    pivot = df_heat.groupby(["Income Band","Education Label"], observed=True)["Personal Loan"]\
                   .mean().round(3).mul(100).unstack()

    fig_hm = px.imshow(pivot,
                        color_continuous_scale=[[0,"#EFF6FF"],[0.5,"#60A5FA"],[1,"#1E3A8A"]],
                        text_auto=".1f",
                        labels=dict(color="Conversion %"),
                        title="Conversion Rate (%) — Income Band × Education Level",
                        aspect="auto")
    fig_hm.update_coloraxes(colorbar_title="Conv %")
    fig_hm.update_layout(height=350, margin=dict(t=60, b=40))
    st.plotly_chart(fig_hm, use_container_width=True)
    st.markdown("""<div class="insight-box">
    The darkest cells represent the highest-yield combinations. <b>High-income + Graduate/Advanced</b>
    customers are the sweet spot. Even within the same income band, education level amplifies
    conversion significantly — use both dimensions when building your campaign audience.</div>""",
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 6 — PREDICT NEW DATA
# ═══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🔮 Upload Customer Data for Loan Prediction</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="margin-bottom:1.25rem;">
    Upload a CSV file with customer data (same format as the training dataset — without the
    <b>Personal Loan</b> column). The selected model will predict each customer's probability
    of accepting a personal loan. Download the results with predictions appended.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_file = st.file_uploader(
            "Upload Customer CSV File",
            type=["csv"],
            help="File must include: Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard"
        )
    with c2:
        model_choice = st.selectbox(
            "Select Prediction Model",
            ["Random Forest", "Gradient Boosted Tree", "Decision Tree"],
            help="Random Forest or Gradient Boosted Tree recommended for highest accuracy"
        )
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Lower threshold = more customers flagged as likely-to-accept (higher recall)"
        )

    # Sample file download
    with open("/home/claude/universal_bank_app/sample_test_data.csv", "rb") as f:
        st.download_button(
            "📥 Download Sample Test File",
            data=f.read(),
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download a sample file with 100 customers to test the prediction feature"
        )

    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Uploaded:** {len(test_df)} records · {test_df.shape[1]} columns")

            # Preprocess test data
            test_clean = test_df.copy()
            test_clean["Experience"] = test_clean["Experience"].clip(lower=0)

            # Drop same cols as training
            drop_cols = [c for c in ["ID", "ZIP Code", "Personal Loan"] if c in test_clean.columns]
            test_feat = test_clean.drop(columns=drop_cols, errors="ignore")

            # Align columns to training feature order
            missing_cols = [c for c in X.columns if c not in test_feat.columns]
            extra_cols   = [c for c in test_feat.columns if c not in X.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                if extra_cols:
                    test_feat = test_feat.drop(columns=extra_cols)
                test_feat = test_feat[X.columns]

                # Scale & predict
                test_scaled  = scaler.transform(test_feat)
                clf          = trained_models[model_choice]
                proba        = clf.predict_proba(test_scaled)[:, 1]
                pred_label   = (proba >= threshold).astype(int)

                # Build output
                output_df = test_df.copy()
                output_df["Loan_Probability"] = proba.round(4)
                output_df["Personal_Loan_Prediction"] = pred_label
                output_df["Prediction_Label"] = pred_label.map(
                    {0:"Will Not Accept", 1:"Likely to Accept"})
                output_df["Confidence"] = proba.apply(
                    lambda p: "High" if p > 0.75 or p < 0.25
                              else "Medium" if p > 0.55 or p < 0.45
                              else "Low")

                # Summary stats
                n_accept = pred_label.sum()
                n_total  = len(pred_label)
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                for col, label, val, sub in [
                    (m1, "Total Customers",    f"{n_total}",          "Uploaded"),
                    (m2, "Predicted Acceptors",f"{n_accept}",         f"{n_accept/n_total*100:.1f}% of batch"),
                    (m3, "Model Used",         model_choice.split()[0],"Classification"),
                    (m4, "Threshold",          f"{threshold:.2f}",    "Probability cutoff"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="label">{label}</div>
                          <div class="value">{val}</div>
                          <div class="sub">{sub}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Probability distribution chart
                fig_prob = px.histogram(
                    output_df, x="Loan_Probability",
                    color="Prediction_Label",
                    color_discrete_map={"Likely to Accept": CORAL, "Will Not Accept": BLUE},
                    nbins=30, title="Prediction Probability Distribution",
                    labels={"Loan_Probability":"Loan Acceptance Probability"},
                    barmode="overlay", opacity=0.75,
                )
                fig_prob.add_vline(x=threshold, line_dash="dash",
                                    line_color=GREY, line_width=2,
                                    annotation_text=f"Threshold = {threshold}")
                fig_prob.update_layout(height=320, margin=dict(t=50, b=20))
                st.plotly_chart(fig_prob, use_container_width=True)

                st.markdown('<div class="section-header">Prediction Results Preview</div>',
                            unsafe_allow_html=True)

                display_cols = (["ID"] if "ID" in output_df.columns else []) + [
                    "Age", "Income", "Education", "CD Account",
                    "Loan_Probability", "Personal_Loan_Prediction",
                    "Prediction_Label", "Confidence"
                ]
                display_cols = [c for c in display_cols if c in output_df.columns]
                st.dataframe(output_df[display_cols].head(25),
                             use_container_width=True, hide_index=True)

                # Download button
                csv_out = output_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=csv_out,
                    file_name=f"loan_predictions_{model_choice.replace(' ','_')}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV matches the expected format. Download the sample file above for reference.")

    else:
        st.info("👆 Upload a customer CSV file above to generate loan predictions. Download the sample test file to see the expected format.")

        st.markdown('<div class="section-header">Expected Column Format</div>',
                    unsafe_allow_html=True)
        sample_schema = pd.DataFrame({
            "Column":   ["ID","Age","Experience","Income","ZIP Code","Family",
                         "CCAvg","Education","Mortgage","Securities Account",
                         "CD Account","Online","CreditCard"],
            "Required": ["Yes"]*13,
            "Type":     ["Integer","Integer","Integer","Integer","Integer","Integer",
                         "Float","Integer (1–3)","Integer","Binary (0/1)","Binary (0/1)",
                         "Binary (0/1)","Binary (0/1)"],
            "Example":  [1502, 29, 4, 95, 9307, 2,
                         3.2, 2, 0, 0, 1, 1, 0],
        })
        st.dataframe(sample_schema, use_container_width=True, hide_index=True)
