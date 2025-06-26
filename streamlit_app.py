# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit pour le projet Credit Scoring
# ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# 1️⃣  Configuration & chargement des données locales
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit-Scoring Dashboard", layout="wide")

API_URL  = "https://credit-scoring-project-5d5k.onrender.com/predict"
FEAT_PATH = Path(__file__).parent / "features_sample.parquet"

@st.cache_data(show_spinner="📦 Chargement des features…")
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # On garde uniquement les colonnes numériques (hors ID)
    numeric_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), numeric_cols

df, num_cols = load_features(FEAT_PATH)

# ────────────────────────────────────────────────────────────────
# 2️⃣  Barre latérale – sélection de l’ID et des axes
# ────────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Paramètres")

default_id = int(df.index[0])
client_id = st.sidebar.selectbox(
    "Choisissez un ID client",
    options=df.index.sort_values(),
    index=0,
    format_func=lambda x: f"{x:,}"
)

# Axes pour le scatter
x_axis = st.sidebar.selectbox("Axe X (scatter)", options=num_cols, index=0)
y_axis = st.sidebar.selectbox("Axe Y (scatter)", options=num_cols, index=1)

threshold = 0.206  # même seuil que dans l’API
st.sidebar.markdown(f"**Seuil de décision :** {threshold:.3f}")

# ────────────────────────────────────────────────────────────────
# 3️⃣  Appel API & affichage du résultat
# ────────────────────────────────────────────────────────────────
params = {"sk_id": client_id}
try:
    with st.spinner("⏳ Appel à l’API…"):
        resp = requests.get(API_URL, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    proba    = payload["proba"]
    decision = payload["decision"]
    color    = "✅ Accordé" if decision == 0 else "❌ Refusé"   # 0 = OK / 1 = défaut

    st.metric("Probabilité de défaut", f"{proba:.2%}")
    st.metric("Décision (0 = accord)", decision, label_visibility="visible")
    st.info(f"**Résultat :** {color}", icon="📊")
except Exception as e:
    st.error(f"Erreur lors de l’appel API : {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────
# 4️⃣  Scatter-plot interactif
# ────────────────────────────────────────────────────────────────
fig = px.scatter(
    df, x=x_axis, y=y_axis,
    opacity=0.35, template="simple_white",
    title=f"Distribution des clients ({x_axis} vs {y_axis})"
)

# Met en évidence l’observation sélectionnée
fig.add_scatter(
    x=[df.loc[client_id, x_axis]],
    y=[df.loc[client_id, y_axis]],
    mode="markers+text",
    marker=dict(size=12, color="red", line=dict(width=2, color="black")),
    text=[str(client_id)],
    textposition="top center",
    name="Client sélectionné"
)

st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# 5️⃣  Tableau des features (facultatif)
# ────────────────────────────────────────────────────────────────
with st.expander("🗒️ Voir les features du client"):
    st.dataframe(df.loc[[client_id]].T, use_container_width=True)
