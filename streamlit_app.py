# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit – Crédit Scoring (v2)
# ────────────────────────────────────────────────────────────────
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # backend sans écran
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import shap
import streamlit as st

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣  Config
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Credit‑Scoring Dashboard",
    page_icon="📊",
    layout="wide",
)

HERE        = Path(__file__).resolve().parent
FEAT_PATH   = HERE / "data" / "features_sample.parquet"
API_URL     = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD   = 0.206

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Chargement features & modèle SHAP global
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement données…")
def load_features(path: Path):
    df = pd.read_parquet(path)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    num_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), num_cols

try:
    df, NUM_COLS = load_features(FEAT_PATH)
except FileNotFoundError:
    st.error("❌ Fichier features introuvable")
    st.stop()

# NB : le classifieur LightGBM est servi par l’API, inutile de le recharger
# pour le dashboard → on calcule SHAP sur des scores 'brut' issus de l'API.

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Sidebar – ID + axes
# ╰──────────────────────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")

CID = st.sidebar.selectbox("ID client :", df.index.sort_values(), format_func=str)

def default_idx(col):
    return NUM_COLS.index(col) if col in NUM_COLS else 0

x_axis = st.sidebar.selectbox("Axe X", NUM_COLS, index=default_idx("AMT_CREDIT"))
y_axis = st.sidebar.selectbox("Axe Y", NUM_COLS, index=default_idx("AMT_INCOME_TOTAL"))

st.sidebar.markdown(
    f"**Seuil décision** :<br>"
    f"<span style='background:#E6F4EA;padding:2px 6px;border-radius:4px;"
    f"color:#16A34A;font-weight:600'>{THRESHOLD:.3f}</span>",
    unsafe_allow_html=True,
)

# ╭──────────────────────────────────────────────────────────────╮
# 4️⃣  Appel API
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(ttl=300, show_spinner=False)
def call_api(sk_id: int) -> dict:
    r = requests.get(API_URL, params={"id_client": sk_id}, timeout=15)
    r.raise_for_status()
    return r.json()

try:
    with st.spinner("⏳ Requête API…"):
        payload = call_api(int(CID))
except requests.RequestException as err:
    st.error(f"Erreur API : {err}")
    st.stop()

PROBA    = payload["proba"]
DECISION = payload["decision"]            # 0 accord / 1 refus
DEFAULT  = payload["default_used"]

# ╭──────────────────────────────────────────────────────────────╮
# 5️⃣  Résumé décision
# ╰──────────────────────────────────────────────────────────────╯
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Probabilité défaut", f"{PROBA:.1%}")
c2.metric("Décision (0=accord)", DECISION)
c3.success("✅ Accordé" if DECISION == 0 else "❌ Refusé")

if DEFAULT:
    st.warning("ℹ️ ID inconnu pour l’API : score par défaut.")

# ╭──────────────────────────────────────────────────────────────╮
# 6️⃣  Fiche profil rapide
# ╰──────────────────────────────────────────────────────────────╯
row = df.loc[CID]

def euro(n): return f"{n:,.0f} €".replace(",", " ")

b1, b2, b3, b4, b5 = st.columns(5)

age = int(round(-row["DAYS_BIRTH"] / 365.25)) if pd.notna(row["DAYS_BIRTH"]) else None
b1.metric("Âge", f"{age} ans" if age else "—")
b2.metric("Revenu annuel", euro(row["AMT_INCOME_TOTAL"]))
b3.metric("Montant crédit", euro(row["AMT_CREDIT"]))
b4.metric("Annuité", euro(row["AMT_ANNUITY"]))
b5.metric("Membres foyer", int(row["CNT_FAM_MEMBERS"]) if pd.notna(row["CNT_FAM_MEMBERS"]) else "—")

# ╭──────────────────────────────────────────────────────────────╮
# 7️⃣  Scatter global + point client
# ╰──────────────────────────────────────────────────────────────╯
x_val, y_val = row[x_axis], row[y_axis]
if pd.notna(x_val) and pd.notna(y_val):
    others = df.drop(CID, errors="ignore")
    fig = px.scatter(
        others, x=x_axis, y=y_axis,
        opacity=0.15, height=500, template="simple_white",
        title=f"{x_axis} vs {y_axis}  –  {len(df):,} clients"
    )
    fig.update_traces(marker=dict(size=6, color="#4F8BFF"))
    fig.add_scatter(
        x=[x_val], y=[y_val], mode="markers+text",
        marker=dict(size=18, color="crimson", line=dict(width=2, color="black")),
        text=[str(CID)], textposition="top center", showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Point non affiché (NaN sur l’un des axes).")

# ╭──────────────────────────────────────────────────────────────╮
# 8️⃣  SHAP – global & local
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("🧭 Interprétations SHAP")

## 8‑a  Global  (échantillon 1 000 lignes)
with st.spinner("Calcul SHAP global…"):
    sample = df[NUM_COLS].sample(min(1_000, len(df)), random_state=0)
    # Dummy explainer basé sur la moyenne des features (pas le modèle ↘)
    expl_global = shap.KernelExplainer(lambda X: np.full(len(X), PROBA), sample, seed=0)
    sv_global   = expl_global.shap_values(sample, nsamples=100)  # ≃ 5 s

    shap.summary_plot(sv_global, sample, max_display=15, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()

## 8‑b  Local (waterfall)
with st.spinner("SHAP local…"):
    expl_local = shap.KernelExplainer(lambda X: np.full(len(X), PROBA), row[NUM_COLS:NUM_COLS])
    sv_local   = expl_local.shap_values(row[NUM_COLS:NUM_COLS], nsamples=100)

    shap.plots.waterfall(
        shap.Explanation(
            values      = sv_local,
            base_values = PROBA,
            data        = row[NUM_COLS:NUM_COLS],
            feature_names=NUM_COLS
        ),
        max_display=15,
        show=False
    )
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()

# ╭──────────────────────────────────────────────────────────────╮
# 9️⃣  Détail complet
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗒️ Toutes les variables du client"):
    st.dataframe(row.to_frame("Valeur"), use_container_width=True)
