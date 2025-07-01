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
    page_icon="📊",
    layout="wide",
)

HERE        = Path(__file__).resolve().parent          # dossier du script
FEAT_PATH   = HERE / "data" / "features_sample.parquet"
API_URL     = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD   = 0.206                                    # seuil décision API

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Chargement des features
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement des features…")
def load_features(file: Path):
    df = pd.read_parquet(file)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)          # IDs propres
    num_cols = [
        c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"
    ]
    return df.set_index("SK_ID_CURR"), num_cols


try:
    df, num_cols = load_features(FEAT_PATH)
except FileNotFoundError:
    st.error(f"❌ Fichier introuvable : {FEAT_PATH}")
    st.stop()

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Sidebar – choix ID & axes
# ╰──────────────────────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")

cid = st.sidebar.selectbox(
    "ID client :", df.index.sort_values(), format_func=str
)

# axes par défaut (crédit vs revenu)
idx_x = num_cols.index("AMT_CREDIT")       if "AMT_CREDIT"       in num_cols else 0
idx_y = num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1

x_axis = st.sidebar.selectbox("Axe X", num_cols, index=idx_x)
y_axis = st.sidebar.selectbox("Axe Y", num_cols, index=idx_y)

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
def call_api(sk_id: int) -> dict:
    r = requests.get(API_URL, params={"id_client": sk_id}, timeout=10)
    r.raise_for_status()
    return r.json()


try:
    with st.spinner("⏳ Requête API…"):
        payload = call_api(int(cid))
except requests.RequestException as err:
    st.error(f"Erreur API : {err}")
    st.stop()

proba    = payload["proba"]
decision = payload["decision"]         # 0 = accord, 1 = refus
default_ = payload["default_used"]

# ╭──────────────────────────────────────────────────────────────╮
# 5️⃣  En-tête & résumé décision
# ╰──────────────────────────────────────────────────────────────╯
col_a, col_b, col_c = st.columns([1, 1, 2])
col_a.metric("Probabilité de défaut", f"{proba:.2%}")
col_b.metric("Décision (0 = accord)", decision)
col_c.success("✅ Accordé" if decision == 0 else "❌ Refusé")

if default_:
    st.warning("ℹ️ L’API a utilisé son ID par défaut (ID inconnu).")

# ╭──────────────────────────────────────────────────────────────╮
# 6️⃣  Fiche profil synthétique
# ╰──────────────────────────────────────────────────────────────╯
row = df.loc[cid]

def euro(n): return f"{n:,.0f} €".replace(",", " ")

box1, box2, box3, box4, box5 = st.columns(5)

# Âge
if "DAYS_BIRTH" in row and pd.notna(row["DAYS_BIRTH"]):
    age = int(round(-row["DAYS_BIRTH"] / 365.25))
    box1.metric("Âge", f"{age} ans")
else:
    box1.metric("Âge", "—")

# Revenu annuel
box2.metric("Revenu annuel", euro(row["AMT_INCOME_TOTAL"]) if "AMT_INCOME_TOTAL" in row else "—")

# Montant crédit
box3.metric("Montant crédit", euro(row["AMT_CREDIT"]) if "AMT_CREDIT" in row else "—")

# Annuité
box4.metric("Annuité", euro(row["AMT_ANNUITY"]) if "AMT_ANNUITY" in row else "—")

# Taille foyer
fam = int(row["CNT_FAM_MEMBERS"]) if "CNT_FAM_MEMBERS" in row and pd.notna(row["CNT_FAM_MEMBERS"]) else "—"
box5.metric("Membres foyer", fam)

# ╭──────────────────────────────────────────────────────────────╮
# 7️⃣  Scatter : point rouge au-dessus
# ╰──────────────────────────────────────────────────────────────╯
x_val, y_val = row[x_axis], row[y_axis]

if pd.isna(x_val) or pd.isna(y_val):
    st.error("Impossible d’afficher le point (valeur NaN sur l’un des axes).")
else:
    others = df.drop(cid, errors="ignore")
    fig = px.scatter(
        others,
        x=x_axis,
        y=y_axis,
        opacity=0.22,
        height=550,
        template="simple_white",
        title=f"{x_axis} vs {y_axis} – {len(df):,} clients",
    )
    fig.update_traces(marker=dict(size=6, color="#4F80FF"))

    # on place APRES -> couche supérieure
    fig.add_scatter(
        x=[x_val],
        y=[y_val],
        mode="markers+text",
        marker=dict(size=18, color="crimson", line=dict(width=2, color="black")),
        text=[str(cid)],
        textposition="top center",
        name="Client sélectionné",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 8️⃣  Détails complets
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗒️ Voir toutes les features du client"):
    st.dataframe(row.to_frame("Valeur"), use_container_width=True)
