import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2535;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] p {
        color: #a0aec0 !important;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #f0f4ff;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.15rem;
    }
    .hero-subtitle { font-size: 0.95rem; color: #6b7a99; font-weight: 400; margin-bottom: 0; }
    .hero-accent { color: #6366f1; }

    .kpi-card {
        background: #161b27;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    .kpi-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #6b7a99; margin-bottom: 0.35rem; }
    .kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #f0f4ff; line-height: 1; }
    .kpi-sub { font-size: 0.75rem; color: #4ade80; margin-top: 0.3rem; }
    .kpi-sub-warn { font-size: 0.75rem; color: #f87171; margin-top: 0.3rem; }

    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem; font-weight: 600; color: #c9d1e0;
        letter-spacing: -0.01em; margin-bottom: 0.75rem;
        padding-bottom: 0.5rem; border-bottom: 1px solid #1e2535;
    }
    .section-eyebrow { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #6366f1; margin-bottom: 0.2rem; }

    hr { border: none; border-top: 1px solid #1e2535; margin: 1.5rem 0; }

    .info-box {
        background: #161b27;
        border: 1px solid #1e2535;
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.83rem; color: #8892a4; line-height: 1.6;
    }
    .info-box strong { color: #c9d1e0; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1", "secondary": "#8b5cf6",
    "green": "#4ade80",   "red": "#f87171",
    "amber": "#fbbf24",   "cyan": "#22d3ee",
    "bg": "#0f1117",      "card": "#161b27",
    "border": "#1e2535",  "text": "#c9d1e0",  "muted": "#6b7a99",
}
MODEL_COLORS = ["#6366f1", "#4ade80", "#fbbf24", "#f87171"]
MODEL_NAMES_SHORT = ["Logistic Reg.", "Random Forest", "Grad. Boosting", "SVM"]
MODEL_NAMES_FULL = [
    "Linear Classification",
    "Random Forest Classification",
    "Gradient Boosting Classification",
    "Support Vector Machines (SVMs)",
]
COLUMNS = [
    "Age", "Employment_type", "Weighting_factor", "Education_level",
    "Schooling_years", "Marital_status", "Employment_area", "Partnership",
    "Ethnicity", "Gender", "Gains", "Losses", "Weekly_working_time",
    "Country_of_birth", "Income",
]
NUM_FEATURES = ["Age", "Weighting_factor", "Schooling_years", "Gains", "Losses", "Weekly_working_time"]
CAT_FEATURES = ["Employment_type", "Education_level", "Marital_status", "Employment_area",
                "Partnership", "Ethnicity", "Gender", "Country_of_birth"]

RESULTS = {
    "KNN": {
        "Linear Classification":            {"accuracy": 0.806, "f1_score": 0.503, "recall": 0.415, "precision": 0.636, "roc_auc": 0.671, "cm": [[354,28],[69,49]]},
        "Random Forest Classification":     {"accuracy": 0.844, "f1_score": 0.639, "recall": 0.585, "precision": 0.704, "roc_auc": 0.754, "cm": [[353,29],[49,69]]},
        "Gradient Boosting Classification": {"accuracy": 0.870, "f1_score": 0.680, "recall": 0.585, "precision": 0.812, "roc_auc": 0.771, "cm": [[366,16],[49,69]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.772, "f1_score": 0.066, "recall": 0.034, "precision": 1.000, "roc_auc": 0.517, "cm": [[382,0],[114,4]]},
    },
    "Iterative": {
        "Linear Classification":            {"accuracy": 0.806, "f1_score": 0.481, "recall": 0.381, "precision": 0.652, "roc_auc": 0.659, "cm": [[358,24],[73,45]]},
        "Random Forest Classification":     {"accuracy": 0.842, "f1_score": 0.636, "recall": 0.585, "precision": 0.697, "roc_auc": 0.753, "cm": [[352,30],[49,69]]},
        "Gradient Boosting Classification": {"accuracy": 0.868, "f1_score": 0.680, "recall": 0.593, "precision": 0.795, "roc_auc": 0.773, "cm": [[364,18],[48,70]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.772, "f1_score": 0.066, "recall": 0.034, "precision": 1.000, "roc_auc": 0.517, "cm": [[382,0],[114,4]]},
    },
    "Dropped": {
        "Linear Classification":            {"accuracy": 0.814, "f1_score": 0.465, "recall": 0.343, "precision": 0.725, "roc_auc": 0.651, "cm": [[336,14],[71,37]]},
        "Random Forest Classification":     {"accuracy": 0.849, "f1_score": 0.657, "recall": 0.611, "precision": 0.710, "roc_auc": 0.767, "cm": [[323,27],[42,66]]},
        "Gradient Boosting Classification": {"accuracy": 0.871, "f1_score": 0.694, "recall": 0.620, "precision": 0.788, "roc_auc": 0.784, "cm": [[332,18],[41,67]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.769, "f1_score": 0.036, "recall": 0.019, "precision": 1.000, "roc_auc": 0.509, "cm": [[350,0],[106,2]]},
    },
}
PREDICTED_COUNTS = {
    "KNN":       {"<=50K": 20324, ">50K": 4676},
    "Iterative": {"<=50K": 20955, ">50K": 4045},
    "Dropped":   {"<=50K": 18812, ">50K": 4412},
}

# ─── Data loading ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "data/einkommen.train") -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS)
    df.replace(" ?", pd.NA, inplace=True)
    df["Income"] = df["Income"].str.strip()
    # Strip leading/trailing whitespace from all string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

try:
    df = load_data()
    DATA_LOADED = True
except FileNotFoundError:
    df = None
    DATA_LOADED = False

# ─── Helper functions ────────────────────────────────────────────────────────────
def base_layout(title="", height=400) -> dict:
    """Base Plotly layout — excludes xaxis, yaxis, legend so callers never get duplicate-key errors."""
    return dict(
        title=dict(text=title, font=dict(family="Space Grotesk", size=14, color=COLORS["text"]), x=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"], size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )

def axis_style(**kwargs) -> dict:
    """Default axis styling merged with any caller overrides."""
    base = dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], tickfont=dict(size=11))
    base.update(kwargs)
    return base

DEFAULT_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"], font=dict(size=11))

def apply_layout(fig, title="", height=400, xaxis_kw=None, yaxis_kw=None, legend_kw=None, **extra):
    """Single entry-point for update_layout — builds the full dict so no key ever appears twice."""
    layout = base_layout(title, height)
    layout["xaxis"]  = axis_style(**(xaxis_kw or {}))
    layout["yaxis"]  = axis_style(**(yaxis_kw or {}))
    layout["legend"] = {**DEFAULT_LEGEND, **(legend_kw or {})}
    layout.update(extra)          # barmode, bargap, showlegend, etc.
    fig.update_layout(**layout)

def make_kpi(label, value, sub="", warn=False):
    sub_class = "kpi-sub-warn" if warn else "kpi-sub"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="{sub_class}">{sub}</div>' if sub else ''}
    </div>"""

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem 0;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#f0f4ff;letter-spacing:-0.01em;">
            ⚡ Income Predict
        </div>
        <div style="font-size:0.72rem;color:#6b7a99;margin-top:0.2rem;">ML Classification Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p>Navigation</p>", unsafe_allow_html=True)
    page = st.radio(
        "", ["📊 Overview", "🔍 Data Exploration", "🤖 Model Results", "📈 Model Comparison", "🎯 Predictions"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2535;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p>Imputation Method</p>", unsafe_allow_html=True)
    imputation = st.selectbox("", ["KNN", "Iterative", "Dropped"], label_visibility="collapsed")

    st.markdown("<p>Primary Model</p>", unsafe_allow_html=True)
    model_choice = st.selectbox("", MODEL_NAMES_FULL, index=2, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2535;margin:1rem 0;'>", unsafe_allow_html=True)
    if DATA_LOADED:
        labeled = df[df["Income"].isin(["<=50K", ">50K"])]
        n_total = len(df)
        n_labeled = len(labeled)
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#6b7a99;line-height:1.8;">
            <strong style="color:#c9d1e0;">Dataset loaded ✓</strong><br>
            {n_total:,} total rows<br>
            {n_labeled:,} labeled rows<br><br>
            <strong style="color:#c9d1e0;">Best model</strong><br>
            Gradient Boosting · Dropped<br>
            Accuracy 87.1% · F1 0.694
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.72rem;color:#f87171;line-height:1.8;">
            ⚠ data/einkommen.train not found.<br>
            Place the file at <code>data/einkommen.train</code> to enable live charts.
        </div>""", unsafe_allow_html=True)

# ─── Page: Overview ──────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div class="hero-title">Income <span class="hero-accent">Classification</span><br>Dashboard</div>
        <div class="hero-subtitle">Binary prediction — ≤50K vs >50K annual income · Gradient Boosting best performer</div>
    </div>""", unsafe_allow_html=True)

    # Derive stats from real data if available, else fallback constants
    if DATA_LOADED:
        labeled = df[df["Income"].isin(["<=50K", ">50K"])]
        n_total    = len(df)
        n_labeled  = len(labeled)
        cls_counts = labeled["Income"].value_counts()
        n_low  = int(cls_counts.get("<=50K", 3779))
        n_high = int(cls_counts.get(">50K",  1221))
    else:
        n_total, n_labeled, n_low, n_high = 30000, 5000, 3779, 1221

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(make_kpi("Total Instances", f"{n_total:,}", "Full dataset size"), unsafe_allow_html=True)
    with k2: st.markdown(make_kpi("Labeled Rows", f"{n_labeled:,}", "Used for training"), unsafe_allow_html=True)
    with k3: st.markdown(make_kpi("Best Accuracy", "87.1%", "Gradient Boosting · Dropped"), unsafe_allow_html=True)
    with k4: st.markdown(make_kpi("Best ROC AUC", "0.784", "Gradient Boosting · Dropped"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-eyebrow">Model Performance</div><div class="section-header">Accuracy across all configurations</div>', unsafe_allow_html=True)
        methods = ["KNN", "Iterative", "Dropped"]
        fig = go.Figure()
        for i, m_full in enumerate(MODEL_NAMES_FULL):
            accs = [RESULTS[mth][m_full]["accuracy"] for mth in methods]
            fig.add_trace(go.Bar(
                name=MODEL_NAMES_SHORT[i], x=methods, y=accs,
                marker_color=MODEL_COLORS[i], marker_line_width=0,
                text=[f"{v:.1%}" for v in accs], textposition="outside", textfont=dict(size=10),
                offsetgroup=i, width=0.18,
            ))
        apply_layout(fig, height=360,
                     yaxis_kw=dict(range=[0.6, 0.97], tickformat=".0%", gridcolor=COLORS["border"]),
                     barmode="group", bargap=0.2, bargroupgap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-eyebrow">Dataset</div><div class="section-header">Class distribution (labeled)</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["≤50K", ">50K"], values=[n_low, n_high],
            hole=0.62,
            marker=dict(colors=[COLORS["primary"], COLORS["red"]], line=dict(color=COLORS["bg"], width=3)),
            textinfo="percent+label", textfont=dict(size=12, color=COLORS["text"]),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig2.add_annotation(text=f"{n_labeled:,}<br><span style='font-size:10px'>labeled</span>",
                            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=COLORS["text"]), align="center")
        apply_layout(fig2, height=360,
                     legend_kw=dict(orientation="h", y=-0.05),
                     showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="info-box"><strong>Problem Setting</strong><br>
        Binary classification to predict whether a person earns ≤50K or >50K per year
        using 14 demographic and employment features.</div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="info-box"><strong>Missing Data Strategy</strong><br>
        Three approaches compared: <strong>KNN Imputation</strong>, <strong>Iterative Imputation</strong>,
        and <strong>Dropping</strong> rows with missing categorical values.</div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="info-box"><strong>Validation Strategy</strong><br>
        10-fold cross-validation on labeled training data. The test set (25K rows) has no income
        labels, so all reported metrics come from the validation folds.</div>""", unsafe_allow_html=True)

# ─── Page: Data Exploration ───────────────────────────────────────────────────────
elif page == "🔍 Data Exploration":
    st.markdown('<div class="hero-title">Data <span class="hero-accent">Exploration</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Dataset structure, missing values, and feature distributions</div><br>', unsafe_allow_html=True)

    if not DATA_LOADED:
        st.warning("Place `einkommen.train` inside a `data/` folder next to `app.py` to enable live EDA.")
        st.stop()

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-eyebrow">Schema</div><div class="section-header">Feature overview</div>', unsafe_allow_html=True)
        miss = df.isna().sum()
        schema = pd.DataFrame({
            "Feature": COLUMNS,
            "Type":    [str(df[c].dtype) for c in COLUMNS],
            "Missing": [f"{miss[c]:,}" if miss[c] > 0 else "—" for c in COLUMNS],
            "Unique":  [str(df[c].nunique()) for c in COLUMNS],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="section-eyebrow">Missing Values</div><div class="section-header">Columns with nulls</div>', unsafe_allow_html=True)
        miss_df = miss[miss > 0].sort_values(ascending=True)
        miss_pcts = (miss_df / len(df)).tolist()
        fig_miss = go.Figure(go.Bar(
            x=miss_df.values, y=miss_df.index.tolist(),
            orientation="h",
            marker=dict(color=[COLORS["primary"], COLORS["secondary"], COLORS["amber"], COLORS["red"]][:len(miss_df)]),
            text=[f"{v:,}  ({p:.1%})" for v, p in zip(miss_df.values, miss_pcts)],
            textposition="outside", textfont=dict(size=10),
        ))
        apply_layout(fig_miss, height=320,
                     xaxis_kw=dict(title="Missing count", gridcolor=COLORS["border"]),
                     yaxis_kw=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_miss, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Numerical distributions from real data ──
    st.markdown('<div class="section-eyebrow">EDA</div><div class="section-header">Numerical feature distributions</div>', unsafe_allow_html=True)
    fig_num = make_subplots(rows=2, cols=3, subplot_titles=NUM_FEATURES,
                             vertical_spacing=0.16, horizontal_spacing=0.08)
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for idx, feat in enumerate(NUM_FEATURES):
        r, c = positions[idx]
        series = df[feat].dropna().astype(float)
        mu = series.mean()
        counts, bin_edges = np.histogram(series, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        norm = counts / counts.max()
        fig_num.add_trace(go.Scatter(
            x=bin_centers, y=norm, fill="tozeroy",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color=COLORS["primary"], width=1.5),
            showlegend=False,
        ), row=r, col=c)
        fig_num.add_vline(x=mu, line_dash="dot", line_color=COLORS["red"], line_width=1.5, row=r, col=c)

    fig_num.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=COLORS["muted"], size=10),
                           margin=dict(l=10, r=10, t=40, b=10))
    fig_num.update_xaxes(gridcolor=COLORS["border"], linecolor=COLORS["border"])
    fig_num.update_yaxes(gridcolor=COLORS["border"], linecolor=COLORS["border"], showticklabels=False)
    st.plotly_chart(fig_num, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Categorical distributions from real data ──
    st.markdown('<div class="section-eyebrow">Categorical</div><div class="section-header">Key categorical feature distributions</div>', unsafe_allow_html=True)
    cat_choice = st.selectbox("Select feature", CAT_FEATURES, index=CAT_FEATURES.index("Gender"))

    cat_counts = df[cat_choice].value_counts().head(15)
    fig_cat = go.Figure(go.Bar(
        x=cat_counts.index.tolist(), y=cat_counts.values,
        marker_color=COLORS["primary"], marker_line_width=0,
        text=cat_counts.values, textposition="outside",
    ))
    apply_layout(fig_cat, title=f"{cat_choice} distribution", height=340,
                 yaxis_kw=dict(gridcolor=COLORS["border"]))
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Income vs numerical feature ──
    st.markdown('<div class="section-eyebrow">Bivariate</div><div class="section-header">Income class vs numerical feature</div>', unsafe_allow_html=True)
    labeled = df[df["Income"].isin(["<=50K", ">50K"])]
    num_choice = st.selectbox("Select numerical feature", NUM_FEATURES, index=0)

    fig_biv = go.Figure()
    for cls, col in [("<=50K", COLORS["primary"]), (">50K", COLORS["red"])]:
        vals = labeled[labeled["Income"] == cls][num_choice].dropna()
        counts, edges = np.histogram(vals, bins=40)
        centers = (edges[:-1] + edges[1:]) / 2
        fig_biv.add_trace(go.Scatter(
            x=centers, y=counts / counts.max(),
            fill="tozeroy", name=cls,
            fillcolor=col.replace("#", "rgba(") + ",0.15)" if "#" not in col else f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:],16)},0.15)",
            line=dict(color=col, width=2),
        ))
    apply_layout(fig_biv, title=f"{num_choice} by income class", height=320,
                 yaxis_kw=dict(showticklabels=False, gridcolor=COLORS["border"]))
    st.plotly_chart(fig_biv, use_container_width=True)

# ─── Page: Model Results ──────────────────────────────────────────────────────────
elif page == "🤖 Model Results":
    st.markdown('<div class="hero-title">Model <span class="hero-accent">Results</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">Imputation: <strong style="color:#c9d1e0;">{imputation}</strong> · Model: <strong style="color:#c9d1e0;">{model_choice}</strong></div><br>', unsafe_allow_html=True)

    metrics = RESULTS[imputation][model_choice]

    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        ("Accuracy",  f"{metrics['accuracy']:.1%}",  ""),
        ("F1-Score",  f"{metrics['f1_score']:.3f}",  "For >50K class"),
        ("Recall",    f"{metrics['recall']:.3f}",    "Sensitivity"),
        ("Precision", f"{metrics['precision']:.3f}", "Pos. predictive value"),
        ("ROC AUC",   f"{metrics['roc_auc']:.3f}",   "Discrimination ability"),
    ]
    for col, (label, val, sub) in zip([k1, k2, k3, k4, k5], cards):
        with col:
            st.markdown(make_kpi(label, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_bar = st.columns(2)

    # ── Confusion Matrix ──────────────────────────────────────────────────────────
    with col_cm:
        st.markdown('<div class="section-eyebrow">Evaluation</div><div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm  = metrics["cm"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        z = [[tn, fp], [fn, tp]]
        text_vals = [
            [f"TN<br>{tn}<br>({tn/total:.1%})", f"FP<br>{fp}<br>({fp/total:.1%})"],
            [f"FN<br>{fn}<br>({fn/total:.1%})", f"TP<br>{tp}<br>({tp/total:.1%})"],
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=z, text=text_vals, texttemplate="%{text}",
            colorscale=[[0, COLORS["card"]], [0.5, "#312e81"], [1, COLORS["primary"]]],
            showscale=False, xgap=3, ygap=3,
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_traces(textfont=dict(size=12, color="white"))
        apply_layout(fig_cm, height=320,
                     xaxis_kw=dict(tickvals=[0,1], ticktext=["≤50K",">50K"],
                                   title="Predicted", gridcolor="rgba(0,0,0,0)"),
                     yaxis_kw=dict(tickvals=[0,1], ticktext=["≤50K",">50K"],
                                   title="Actual", gridcolor="rgba(0,0,0,0)",
                                   autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Metric bar ───────────────────────────────────────────────────────────────
    with col_bar:
        st.markdown('<div class="section-eyebrow">Evaluation</div><div class="section-header">All metrics at a glance</div>', unsafe_allow_html=True)
        metric_names = ["Accuracy", "F1-Score", "Recall", "Precision", "ROC AUC"]
        metric_vals  = [metrics["accuracy"], metrics["f1_score"], metrics["recall"], metrics["precision"], metrics["roc_auc"]]
        bar_colors   = [COLORS["primary"] if v >= 0.7 else COLORS["amber"] if v >= 0.5 else COLORS["red"] for v in metric_vals]

        fig_bar = go.Figure(go.Bar(
            x=metric_names, y=metric_vals,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in metric_vals], textposition="outside",
        ))
        apply_layout(fig_bar, height=320,
                     yaxis_kw=dict(range=[0, 1.18], gridcolor=COLORS["border"], tickformat=".0%"))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Full table for this imputation ────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-eyebrow">All Models</div><div class="section-header">Full comparison — {imputation} imputation</div>', unsafe_allow_html=True)
    rows = []
    for mname in MODEL_NAMES_FULL:
        m = RESULTS[imputation][mname]
        is_best = mname == "Gradient Boosting Classification"
        rows.append({
            "Model":     mname.replace(" Classification","").replace(" (SVMs)","") + (" ★" if is_best else ""),
            "Accuracy":  f"{m['accuracy']:.1%}",
            "F1-Score":  f"{m['f1_score']:.3f}",
            "Recall":    f"{m['recall']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "ROC AUC":   f"{m['roc_auc']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── Page: Model Comparison ───────────────────────────────────────────────────────
elif page == "📈 Model Comparison":
    st.markdown('<div class="hero-title">Model <span class="hero-accent">Comparison</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Cross-imputation performance · 10-fold cross-validation</div><br>', unsafe_allow_html=True)

    metric_opt = st.selectbox(
        "Compare metric", ["accuracy", "f1_score", "recall", "precision", "roc_auc"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    methods = ["KNN", "Iterative", "Dropped"]
    fig_cmp = go.Figure()
    for i, m_full in enumerate(MODEL_NAMES_FULL):
        vals = [RESULTS[mth][m_full][metric_opt] for mth in methods]
        fig_cmp.add_trace(go.Bar(
            name=MODEL_NAMES_SHORT[i], x=methods, y=vals,
            marker_color=MODEL_COLORS[i], marker_line_width=0,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            offsetgroup=i, width=0.18,
        ))
    apply_layout(fig_cmp, title=f"{metric_opt.replace('_',' ').title()} by imputation method", height=400,
                 yaxis_kw=dict(gridcolor=COLORS["border"],
                               range=[0, max(RESULTS[m][n][metric_opt] for m in methods for n in MODEL_NAMES_FULL) * 1.2]),
                 barmode="group", bargap=0.2, bargroupgap=0.05)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Radar: best model per imputation ─────────────────────────────────────────
    st.markdown('<div class="section-header">Multi-metric radar — Gradient Boosting across imputation methods</div>', unsafe_allow_html=True)
    radar_metrics = ["accuracy", "f1_score", "recall", "precision", "roc_auc"]
    radar_labels  = ["Accuracy", "F1-Score", "Recall", "Precision", "ROC AUC"]
    best_model = "Gradient Boosting Classification"

    fig_radar = go.Figure()
    for i, mth in enumerate(methods):
        vals = [RESULTS[mth][best_model][m] for m in radar_metrics] + [RESULTS[mth][best_model][radar_metrics[0]]]
        r, g, b = int(MODEL_COLORS[i][1:3],16), int(MODEL_COLORS[i][3:5],16), int(MODEL_COLORS[i][5:],16)
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            name=f"{mth}", fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(color=MODEL_COLORS[i], width=2),
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor=COLORS["border"], tickfont=dict(size=9, color=COLORS["muted"])),
            angularaxis=dict(gridcolor=COLORS["border"], tickfont=dict(size=11, color=COLORS["text"])),
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=420,
        font=dict(family="Inter", color=COLORS["muted"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=50, t=40, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Performance heatmap — all models × all methods</div>', unsafe_allow_html=True)
    heat_metric = st.selectbox("Heatmap metric", ["accuracy", "f1_score", "roc_auc"],
                                format_func=lambda x: x.replace("_"," ").title(), key="heat")

    z_vals = [[RESULTS[mth][m][heat_metric] for mth in methods] for m in MODEL_NAMES_FULL]
    y_labels = [n.replace(" Classification","").replace(" (SVMs)","") for n in MODEL_NAMES_FULL]
    fig_heat = go.Figure(go.Heatmap(
        z=z_vals, x=methods, y=y_labels,
        colorscale=[[0,"#1a2035"],[0.5,"#312e81"],[1,COLORS["primary"]]],
        text=[[f"{v:.3f}" for v in row] for row in z_vals],
        texttemplate="%{text}", textfont=dict(size=13),
        xgap=4, ygap=4,
    ))
    apply_layout(fig_heat, height=300,
                 xaxis_kw=dict(gridcolor="rgba(0,0,0,0)"),
                 yaxis_kw=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_heat, use_container_width=True)

# ─── Page: Predictions ────────────────────────────────────────────────────────────
elif page == "🎯 Predictions":
    st.markdown('<div class="hero-title">Test Set <span class="hero-accent">Predictions</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Gradient Boosting applied to 25,000 unlabeled test instances</div><br>', unsafe_allow_html=True)

    methods = ["KNN", "Iterative", "Dropped"]
    p1, p2, p3 = st.columns(3)
    for col, mth in zip([p1, p2, p3], methods):
        data = PREDICTED_COUNTS[mth]
        total = data["<=50K"] + data[">50K"]
        pct_high = data[">50K"] / total
        with col:
            st.markdown(make_kpi(f"{mth} Imputation", f"{data['>50K']:,}", f"{pct_high:.1%} predicted >50K"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grouped bar ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Predicted class counts — Gradient Boosting Classifier</div>', unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(
        name="≤50K", x=methods,
        y=[PREDICTED_COUNTS[m]["<=50K"] for m in methods],
        marker_color=COLORS["green"], marker_line_width=0,
        text=[PREDICTED_COUNTS[m]["<=50K"] for m in methods], textposition="outside",
    ))
    fig_pred.add_trace(go.Bar(
        name=">50K", x=methods,
        y=[PREDICTED_COUNTS[m][">50K"] for m in methods],
        marker_color=COLORS["red"], marker_line_width=0,
        text=[PREDICTED_COUNTS[m][">50K"] for m in methods], textposition="outside",
    ))
    apply_layout(fig_pred, height=380, yaxis_kw=dict(gridcolor=COLORS["border"]),
                 barmode="group", bargap=0.3)
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Stacked % bar ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Class share by imputation method</div>', unsafe_allow_html=True)
    fig_stk = go.Figure()
    totals = {m: PREDICTED_COUNTS[m]["<=50K"] + PREDICTED_COUNTS[m][">50K"] for m in methods}
    fig_stk.add_trace(go.Bar(
        name="≤50K", x=methods,
        y=[PREDICTED_COUNTS[m]["<=50K"] / totals[m] for m in methods],
        marker_color=COLORS["primary"], marker_line_width=0,
        text=[f"{PREDICTED_COUNTS[m]['<=50K']/totals[m]:.1%}" for m in methods], textposition="inside",
    ))
    fig_stk.add_trace(go.Bar(
        name=">50K", x=methods,
        y=[PREDICTED_COUNTS[m][">50K"] / totals[m] for m in methods],
        marker_color=COLORS["red"], marker_line_width=0,
        text=[f"{PREDICTED_COUNTS[m]['>50K']/totals[m]:.1%}" for m in methods], textposition="inside",
    ))
    apply_layout(fig_stk, height=300, yaxis_kw=dict(tickformat=".0%", gridcolor=COLORS["border"]),
                 barmode="stack")
    st.plotly_chart(fig_stk, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <strong>Interpretation</strong><br>
    Gradient Boosting consistently predicts ~18–23% of test instances as earning >50K across all imputation strategies.
    The <strong>Iterative</strong> method predicts the most ≤50K cases (20,955), while <strong>KNN</strong> and
    <strong>Dropped</strong> show similar distributions. Imputation method does not strongly shift prediction
    proportions, suggesting robust model behaviour.
    </div>""", unsafe_allow_html=True)