# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit – Crédit Scoring
# ────────────────────────────────────────────────────────────────
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣  Configuration globale
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Credit-Scoring Dashboard",
    page_icon="📊",
    layout="wide",
)

API_URL             = "https://credit-scoring-project-5d5k.onrender.com/predict"
DECISION_THRESHOLD  = 0.206                      # seuil model / API
ROOT                = Path(__file__).resolve().parent
FEAT_PATH           = ROOT / "data" / "features_sample.parquet"

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Chargement des features (cache)
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Lecture des features…")
def load_features(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Impossible de trouver {path}.\n"
            "Vérifiez que le fichier existe bien ou changez FEAT_PATH."
        )
    df = pd.read_parquet(path)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)    # nettoie l’ID

    num_cols = [
        c for c in df.columns
        if df[c].dtype != "object" and c != "SK_ID_CURR"
    ]
    return df.set_index("SK_ID_CURR"), num_cols

try:
    df, num_cols = load_features(FEAT_PATH)
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Barre latérale – paramètres utilisateur
# ╰──────────────────────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")

client_id = st.sidebar.selectbox(
    "ID client :",
    options=df.index.sort_values(),
    index=0,
    format_func=str,
)

# axes par défaut (montant du crédit vs revenu)
default_x = num_cols.index("AMT_CREDIT")       if "AMT_CREDIT"       in num_cols else 0
default_y = num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1

x_axis = st.sidebar.selectbox("Axe X", options=num_cols, index=default_x)
y_axis = st.sidebar.selectbox("Axe Y", options=num_cols, index=default_y)

st.sidebar.markdown(
    f"**Seuil décision** : "
    f"<span style='background:#E6F4EA;padding:2px 6px;border-radius:4px;"
    f"color:#16A34A;font-weight:600'>{DECISION_THRESHOLD:.3f}</span>",
    unsafe_allow_html=True,
)

# ╭──────────────────────────────────────────────────────────────╮
# 4️⃣  Appel à l’API (cache 5 min)
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(ttl=300)
def call_api(cid: int):
    r = requests.get(API_URL, params={"id_client": cid}, timeout=10)
    r.raise_for_status()
    return r.json()

try:
    with st.spinner("⏳ Requête API…"):
        payload = call_api(int(client_id))
except requests.RequestException as e:
    st.error(f"Erreur API : {e}")
    st.stop()

proba        = payload["proba"]
decision     = payload["decision"]          # 0 = accordé, 1 = refusé
default_used = payload["default_used"]

# ╭──────────────────────────────────────────────────────────────╮
# 5️⃣  Résultats globaux
# ╰──────────────────────────────────────────────────────────────╯
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Probabilité de défaut", f"{proba:.2%}")
c2.metric("Décision (0 = accord)", decision)
c3.info(f"**Résultat** : {'✅ Accordé' if decision == 0 else '❌ Refusé'}")

if default_used:
    st.warning("⚠️ L’API a utilisé son ID par défaut (ID inconnu).")

# ╭──────────────────────────────────────────────────────────────╮
# 6️⃣  Scatter-plot (points bleus derrière, point rouge devant)
# ╰──────────────────────────────────────────────────────────────╯
fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    opacity=0.20,                     # transparence pour mieux voir
    template="simple_white",
    title=f"Distribution ({x_axis} vs {y_axis}) – {len(df):,} clients",
    height=550,
)

# on applique style aux points de fond
fig.update_traces(
    marker=dict(size=6, color="#636EFA"),
    selector=dict(mode="markers")
)

# trace du client sélectionné – ajoutée en DERNIER → passe devant
fig.add_scatter(
    x=[df.loc[client_id, x_axis]],
    y=[df.loc[client_id, y_axis]],
    mode="markers+text",
    marker=dict(size=14, color="red", line=dict(width=2, color="black")),
    text=[str(client_id)],
    textposition="top center",
    name="Client sélectionné",
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 7️⃣  Détails des features
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗒️ Voir toutes les features du client"):
    st.dataframe(df.loc[[client_id]].T, use_container_width=True)
