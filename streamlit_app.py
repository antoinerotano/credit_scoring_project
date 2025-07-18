# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit – Crédit Scoring
# ────────────────────────────────────────────────────────────────
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import shap                       # <‑‑ installé dans requirements.txt
import streamlit as st

# ╭────────────────────────────────────────────╮
# 1.  CONFIG
# ╰────────────────────────────────────────────╯
st.set_page_config(page_title="Credit‑Scoring",
                   page_icon="📊", layout="wide")
HERE        = Path(__file__).resolve().parent
FEAT_PATH   = HERE / "data" / "features_sample.parquet"
API_URL     = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD   = 0.206                   # seuil métier

# ╭────────────────────────────────────────────╮
# 2.  DONNÉES
# ╰────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement des features…")
def load_features(path: Path):
    df = pd.read_parquet(path).copy()
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    num_cols = [c for c in df
                if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), num_cols


try:
    df, NUM_COLS = load_features(FEAT_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {FEAT_PATH.relative_to(HERE)}")
    st.stop()

# ╭────────────────────────────────────────────╮
# 3.  SIDEBAR
# ╰────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")
CID = st.sidebar.selectbox("ID client :", df.index.sort_values(),
                           format_func=str)

idx_x = NUM_COLS.index("AMT_CREDIT")       if "AMT_CREDIT" in NUM_COLS else 0
idx_y = NUM_COLS.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in NUM_COLS else 1
X_AXIS = st.sidebar.selectbox("Axe X", NUM_COLS, index=idx_x)
Y_AXIS = st.sidebar.selectbox("Axe Y", NUM_COLS, index=idx_y)

st.sidebar.markdown(
    f"**Seuil** : <span style='background:#E6F4EA;"
    f"padding:2px 6px;border-radius:4px;color:#16A34A;font-weight:600'>"
    f"{THRESHOLD:.3f}</span>", unsafe_allow_html=True)

# ╭────────────────────────────────────────────╮
# 4.  API SCORE
# ╰────────────────────────────────────────────╯
@st.cache_data(ttl=300)
def call_api(sk_id: int) -> dict:
    resp = requests.get(API_URL, params={"id_client": sk_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()

try:
    with st.spinner("⏳ Requête API…"):
        payload = call_api(int(CID))
except requests.RequestException as e:
    st.error(f"Erreur API : {e}")
    st.stop()

PROBA, DECISION, DEFAULT_USED = (payload["proba"],
                                 payload["decision"],
                                 payload["default_used"])

# ╭────────────────────────────────────────────╮
# 5.  RÉSUMÉ TOP
# ╰────────────────────────────────────────────╯
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Probabilité de défaut", f"{PROBA:.1%}")
c2.metric("Décision (0 = accord)", DECISION)
c3.success("✅ Accordé" if DECISION == 0 else "❌ Refusé")
if DEFAULT_USED:
    st.warning("ℹ️ ID inconnu : score par défaut.")

row = df.loc[CID]

# ╭────────────────────────────────────────────╮
# 6.  PROFIL SYNTHÉTIQUE
# ╰────────────────────────────────────────────╯
def euro(n): return f"{n:,.0f} €".replace(",", " ")

b1, b2, b3, b4, b5 = st.columns(5)
age = int(round(-row["DAYS_BIRTH"] / 365.25)) if pd.notna(row["DAYS_BIRTH"]) else "—"
b1.metric("Âge",          f"{age} ans")
b2.metric("Revenu annuel", euro(row["AMT_INCOME_TOTAL"]))
b3.metric("Montant crédit", euro(row["AMT_CREDIT"]))
b4.metric("Annuité",       euro(row["AMT_ANNUITY"]))
b5.metric("Membres foyer", int(row["CNT_FAM_MEMBERS"]))

# ╭────────────────────────────────────────────╮
# 7.  SCATTER GLOBAL
# ╰────────────────────────────────────────────╯
if pd.notna(row[X_AXIS]) and pd.notna(row[Y_AXIS]):
    others = df.drop(CID, errors="ignore")
    fig = px.scatter(others, x=X_AXIS, y=Y_AXIS,
                     opacity=0.25, height=500,
                     template="simple_white",
                     title=f"{X_AXIS} vs {Y_AXIS} – {len(df):,} clients")
    fig.update_traces(marker_size=6, marker_color="#4F80FF")
    fig.add_scatter(x=[row[X_AXIS]], y=[row[Y_AXIS]],
                    mode="markers+text",
                    marker=dict(size=18, color="crimson",
                                line=dict(width=2, color="black")),
                    text=[str(CID)], textposition="top center",
                    showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Valeurs NaN sur l’un des axes ; scatter non affiché.")

# ╭────────────────────────────────────────────╮
# 8.  DÉTAILS FEATURES
# ╰────────────────────────────────────────────╯
with st.expander("🗒️ Voir toutes les features du client"):
    st.dataframe(row.to_frame("Valeur"), use_container_width=True)
