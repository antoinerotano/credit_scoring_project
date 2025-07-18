# streamlit_app.py
# ══════════════════════════════════════════════════════════════
# Dashboard Streamlit – Crédit Scoring (v2)
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
import json
import urllib.parse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
import joblib

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣  Configuration générale
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Credit‑Scoring Dashboard",
    page_icon="📊",
    layout="wide",
)

HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE / "data"
FEAT_PATH   = DATA_DIR / "features_sample.parquet"
MODEL_PATH  = HERE / "models_artifacts" / "model.joblib"   # ← modèle prod
GLOBAL_SHAP = DATA_DIR / "shap_global_values.parquet"      # cache global SHAP
API_URL     = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD   = 0.206                                        # seuil décision
PALETTE     = px.colors.sequential.Viridis                # palette WCAG‑friendly

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Chargements (features, modèle, SHAP)
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement des features…")
def load_features(file: Path):
    df = pd.read_parquet(file)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    num_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), num_cols

@st.cache_resource(show_spinner="🧠 Chargement du modèle…")
def load_model(model_path: Path):
    pipe: LGBMClassifier | None = joblib.load(model_path)
    imputer: SimpleImputer      = pipe.named_steps["imputer"]
    clf: LGBMClassifier         = pipe.named_steps["clf"]
    return pipe, imputer, clf

@st.cache_data(show_spinner="🔍 Calcul SHAP global (1 ère fois uniquement)…")
def compute_global_shap(sample: pd.DataFrame, clf) -> pd.Series:
    """Retourne l'importance SHAP moyenne absolue pour chaque feature."""
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(sample)[1]  # classe 1
    importance = np.abs(shap_vals).mean(axis=0)
    return pd.Series(importance, index=sample.columns).sort_values(ascending=False)

# ────────────────────────────────────────────────────────────────
try:
    df, NUM_COLS = load_features(FEAT_PATH)
except FileNotFoundError:
    st.error(f"❌ Fichier introuvable : {FEAT_PATH}")
    st.stop()

PIPE, IMPUTER, CLF = load_model(MODEL_PATH)

# échantillon pour SHAP global
if GLOBAL_SHAP.exists():
    GLOBAL_S = pd.read_parquet(GLOBAL_SHAP)["importance"]
else:
    GLOBAL_S = compute_global_shap(df[NUM_COLS].sample(1_000, random_state=0), CLF)
    GLOBAL_S.to_frame("importance").to_parquet(GLOBAL_SHAP)

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Sidebar – paramètres utilisateur
# ╰──────────────────────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")

CID = st.sidebar.selectbox("ID client :", df.index.sort_values(), format_func=str)

# axes par défaut
IDX_X = NUM_COLS.index("AMT_CREDIT")       if "AMT_CREDIT"       in NUM_COLS else 0
IDX_Y = NUM_COLS.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in NUM_COLS else 1

X_AXIS = st.sidebar.selectbox("Axe X (scatter)", NUM_COLS, index=IDX_X)
Y_AXIS = st.sidebar.selectbox("Axe Y (scatter)", NUM_COLS, index=IDX_Y)

UNI_FEATURE = st.sidebar.selectbox("Distribution univariée", NUM_COLS, index=IDX_Y)

st.sidebar.markdown(
    f"<small><b>Seuil décision&nbsp;:</b> <span style='background:#E6F4EA;"
    f"padding:2px 6px;border-radius:4px;color:#16A34A;font-weight:600'>{THRESHOLD:.3f}</span></small>",
    unsafe_allow_html=True,
)

# ╭──────────────────────────────────────────────────────────────╮
# 4️⃣  Appel de l’API
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(ttl=300)
def call_api(sk_id: int) -> dict:
    r = requests.get(API_URL, params={"id_client": sk_id}, timeout=10)
    r.raise_for_status()
    return r.json()

try:
    with st.spinner("⏳ Requête API…"):
        payload = call_api(int(CID))
except requests.RequestException as err:
    st.error(f"Erreur API : {err}")
    st.stop()

PROBA    = payload["proba"]
DECISION = payload["decision"]        # 0 = accord, 1 = refus
DEFAULT_ = payload["default_used"]

# ╭──────────────────────────────────────────────────────────────╮
# 5️⃣  En‑tête – jauge & décision
# ╰──────────────────────────────────────────────────────────────╯
gauge = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=PROBA * 100,
        number={"suffix": "%", "font": {"size": 32}},
        delta={"reference": THRESHOLD * 100, "increasing": {"color": "#B91C1C"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "#A7F3D0"},
                {"range": [THRESHOLD * 100, 100], "color": "#FECACA"},
            ],
            "threshold": {
                "line": {"color": "#B91C1C", "width": 4},
                "thickness": 0.75,
                "value": THRESHOLD * 100,
            },
        },
    )
).update_layout(height=200, margin_t=10)

col_g, col_d = st.columns([1, 2])
col_g.plotly_chart(gauge, use_container_width=True)
col_d.success("✅ Crédit accordé" if DECISION == 0 else "❌ Crédit refusé")

if DEFAULT_:
    st.warning("ℹ️ ID client absent ‑ valeurs par défaut utilisées.")

# ╭──────────────────────────────────────────────────────────────╮
# 6️⃣  Profil synthétique
# ╰──────────────────────────────────────────────────────────────╯
row = df.loc[CID]

def euro(x): return f"{x:,.0f} €".replace(",", " ")

c1, c2, c3, c4, c5 = st.columns(5)
age = int(round(-row["DAYS_BIRTH"] / 365.25)) if "DAYS_BIRTH" in row else None
c1.metric("Âge", f"{age} ans" if age else "—")
c2.metric("Revenu annuel", euro(row["AMT_INCOME_TOTAL"]) if "AMT_INCOME_TOTAL" in row else "—")
c3.metric("Montant crédit", euro(row["AMT_CREDIT"]) if "AMT_CREDIT" in row else "—")
c4.metric("Annuité", euro(row["AMT_ANNUITY"]) if "AMT_ANNUITY" in row else "—")
c5.metric("Membres foyer", int(row["CNT_FAM_MEMBERS"]) if "CNT_FAM_MEMBERS" in row else "—")

# ╭──────────────────────────────────────────────────────────────╮
# 7️⃣  SHAP – Importance globale vs locale
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("🧩 Explicabilité du score")

# — locale —
with st.spinner("Calcul SHAP local…"):
    x_proc = pd.DataFrame(
        IMPUTER.transform(row[NUM_COLS].to_frame().T),
        columns=NUM_COLS,
        index=[CID],
    )
    explainer = shap.TreeExplainer(CLF)
    sv = explainer.shap_values(x_proc)[1][0]        # (p,)
local_imp = pd.Series(sv, index=NUM_COLS).sort_values(key=np.abs, ascending=False)[:20]

# — global (déjà trié) —
global_imp = GLOBAL_S.head(20)

tab1, tab2 = st.tabs(["🌍 Importance globale", "👤 Importance locale"])
with tab1:
    fig_g = px.bar(
        global_imp.iloc[::-1],
        orientation="h",
        color_discrete_sequence=[PALETTE[-2]],
        height=500,
        labels={"value": "Impact moyen absolu", "index": "Feature"},
        title="Top 20 – importance SHAP globale",
    ).update_layout(yaxis_title="")
    st.plotly_chart(fig_g, use_container_width=True)

with tab2:
    fig_l = px.bar(
        local_imp.iloc[::-1],
        orientation="h",
        color=local_imp.iloc[::-1].apply(lambda x: "#DC2626" if x > 0 else "#16A34A"),
        height=500,
        labels={"value": "Contribution (client)", "index": "Feature"},
        title="Top 20 – contributions SHAP client",
    ).update_layout(yaxis_title="")
    st.plotly_chart(fig_l, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 8️⃣  Analyse univariée
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("📈 Position du client VS population")

hist_df = df[UNI_FEATURE].dropna()
fig_uni = px.histogram(
    hist_df,
    x=UNI_FEATURE,
    nbins=40,
    opacity=0.7,
    color_discrete_sequence=[PALETTE[2]],
    template="simple_white",
    labels={UNI_FEATURE: UNI_FEATURE},
)
fig_uni.add_vline(row[UNI_FEATURE], line_dash="dash", line_color="crimson", line_width=3)
st.plotly_chart(fig_uni, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 9️⃣  Analyse bi‑variée (scatter)
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("🧮 Analyse bi‑variée")

x_val, y_val = row[X_AXIS], row[Y_AXIS]
others = df.drop(CID, errors="ignore")

fig_scatter = px.scatter(
    others,
    x=X_AXIS,
    y=Y_AXIS,
    opacity=0.23,
    height=550,
    template="simple_white",
    color_discrete_sequence=[PALETTE[4]],
)
fig_scatter.add_scatter(
    x=[x_val],
    y=[y_val],
    mode="markers+text",
    marker=dict(size=16, color="crimson", line=dict(width=2, color="black")),
    text=[str(CID)],
    textposition="top center",
    showlegend=False,
)
fig_scatter.update_layout(title=f"{X_AXIS} vs {Y_AXIS}")
st.plotly_chart(fig_scatter, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 🔟  Détails complets
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗒️ Voir toutes les features du client"):
    st.dataframe(row.to_frame("Valeur"), use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 🔚  Footer
# ╰──────────────────────────────────────────────────────────────╯
st.caption(
    "© 2025 Prêt à dépenser — Dashboard conçu pour l’explicabilité du modèle "
    "LightGBM. Palettes Plotly respectant le contraste WCAG AA."
)
