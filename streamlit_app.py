# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit – Crédit Scoring
# ────────────────────────────────────────────────────────────────
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣  Config générale
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Credit-Scoring Dashboard",
    layout="wide",
    page_icon="📊",
)

ROOT = Path(__file__).resolve().parent
FEAT_PATH = ROOT / "data" / "features_sample.parquet"
API_URL = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD = 0.206  # même seuil que l’API

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Chargement des features
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement des features…")
def load_features(pq_path: Path):
    df = pd.read_parquet(pq_path)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    numeric_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), numeric_cols

try:
    df, num_cols = load_features(FEAT_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {FEAT_PATH}")
    st.stop()

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Sidebar – ID et axes
# ╰──────────────────────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")

cid = st.sidebar.selectbox(
    "ID client :",
    options=df.index.sort_values(),
    format_func=str,
)

# Axes par défaut “crédit vs revenu”
def_axis_x = num_cols.index("AMT_CREDIT")       if "AMT_CREDIT"       in num_cols else 0
def_axis_y = num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1

x_axis = st.sidebar.selectbox("Axe X", num_cols, index=def_axis_x)
y_axis = st.sidebar.selectbox("Axe Y", num_cols, index=def_axis_y)

st.sidebar.markdown(
    f"**Seuil décision** : "
    f"<span style='background:#E6F4EA;padding:2px 6px;border-radius:4px;"
    f"color:#16A34A;font-weight:600'>{THRESHOLD:.3f}</span>",
    unsafe_allow_html=True,
)

# ╭──────────────────────────────────────────────────────────────╮
# 4️⃣  Appel API
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(ttl=300)
def call_api(sk_id: int):
    r = requests.get(API_URL, params={"id_client": sk_id}, timeout=10)
    r.raise_for_status()
    return r.json()

try:
    with st.spinner("⏳ Requête API…"):
        payload = call_api(int(cid))
except requests.RequestException as e:
    st.error(f"Erreur API : {e}")
    st.stop()

proba    = payload["proba"]
decision = payload["decision"]          # 0 = accord
default_ = payload["default_used"]

# ╭──────────────────────────────────────────────────────────────╮
# 5️⃣  Résumé
# ╰──────────────────────────────────────────────────────────────╯
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Probabilité de défaut", f"{proba:.2%}")
c2.metric("Décision (0 = accord)", decision)
c3.success("✅ Accordé" if decision == 0 else "❌ Refusé")

if default_:
    st.warning("L’API a utilisé son ID par défaut (ID inconnu).")

# ╭──────────────────────────────────────────────────────────────╮
# 6️⃣  Scatter – point rouge au-dessus
# ╰──────────────────────────────────────────────────────────────╯
row = df.loc[cid]
x_val, y_val = row[x_axis], row[y_axis]

if pd.isna(x_val) or pd.isna(y_val):
    st.error(
        f"Impossible d’afficher le point rouge : "
        f"`{x_axis}` ou `{y_axis}` est manquant pour le client {cid}. "
        "Changez d’ID ou d’axes."
    )
else:
    others = df.drop(cid, errors="ignore")
    fig = px.scatter(
        others,
        x=x_axis,
        y=y_axis,
        opacity=0.25,
        height=550,
        template="simple_white",
        title=f"{x_axis} vs {y_axis} – {len(df):,} clients",
    )
    fig.update_traces(marker=dict(size=6, color="#4F80FF"))

    # Ajouté après → passe DEVANT les autres points
    fig.add_scatter(
        x=[x_val],
        y=[y_val],
        mode="markers+text",
        marker=dict(size=16, color="crimson", line=dict(width=2, color="black")),
        text=[str(cid)],
        textposition="top center",
        name="Client sélectionné",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 7️⃣  Détails
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗒️ Voir toutes les features du client"):
    st.dataframe(row.to_frame("Valeur"), use_container_width=True)
