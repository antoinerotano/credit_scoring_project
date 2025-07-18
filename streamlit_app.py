# streamlit_app.py
# ────────────────────────────────────────────────────────────────
from pathlib import Path
import warnings, joblib, json, numpy as np, pandas as pd
import requests, plotly.express as px, plotly.graph_objects as go
import shap, streamlit as st

warnings.filterwarnings("ignore")            # SHAP noisy

# ╭────────────────────────────────────────────╮
# CONFIG
# ╰────────────────────────────────────────────╯
st.set_page_config(page_title="Credit‑Scoring",
                   page_icon="📊", layout="wide")
HERE         = Path(__file__).resolve().parent
FEAT_PATH    = HERE / "data" / "features_sample.parquet"
MODEL_PATH   = HERE / "models_artifacts" / "model.joblib"
API_URL      = "https://credit-scoring-project-5d5k.onrender.com/predict"
THR          = 0.206           # seuil métier (LightGBM)
PALETTE_ACC  = ["#1455E7", "#D92727"]      # bleu / rouge contrastés

# ╭────────────────────────────────────────────╮
# DATA & MODEL  (cachés)
# ╰────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement des données…")
def load_data(path: Path):
    df = pd.read_parquet(path).copy()
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    num = [c for c in df if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), num

@st.cache_resource(show_spinner="🔧 Chargement du modèle…")
def load_model(path: Path):
    pipe = joblib.load(path)
    lgbm = pipe.named_steps["clf"]      # LightGBM natif
    imp  = pipe.named_steps["imputer"]  # SimpleImputer
    return pipe, lgbm, imp

df, NUM_COLS      = load_data(FEAT_PATH)
PIPE, CLF, IMP    = load_model(MODEL_PATH)

# ╭────────────────────────────────────────────╮
# GLOBAL SHAP (pré‑calc une fois)
# ╰────────────────────────────────────────────╯
@st.cache_resource
def global_shap(df_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(CLF)
    sv = explainer.shap_values(df_sample)[1] \
         if isinstance(explainer.shap_values(df_sample), list) \
         else explainer.shap_values(df_sample)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=df_sample.columns)
    return mean_abs.sort_values(ascending=False)[:15]     # TOP‑15

SAMPLE = IMP.transform(df[NUM_COLS].sample(min(1_000, len(df)), random_state=0))
GLOBAL_S = global_shap(pd.DataFrame(SAMPLE, columns=NUM_COLS))

# ╭────────────────────────────────────────────╮
# SIDEBAR (sélection)
# ╰────────────────────────────────────────────╯
st.sidebar.header("🔎 Paramètres")
CID = st.sidebar.selectbox("ID client :", df.index.sort_values(), format_func=str)

x_def = NUM_COLS.index("AMT_CREDIT") if "AMT_CREDIT" in NUM_COLS else 0
y_def = NUM_COLS.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in NUM_COLS else 1
X_VAR = st.sidebar.selectbox("Axe X", NUM_COLS, index=x_def)
Y_VAR = st.sidebar.selectbox("Axe Y", NUM_COLS, index=y_def)

st.sidebar.markdown(f"**Seuil** : <span style='background:#E6F4EA;"
                    f"padding:2px 6px;border-radius:4px;color:#117A37'>{THR:.3f}</span>",
                    unsafe_allow_html=True)

# ╭────────────────────────────────────────────╮
# API SCORE
# ╰────────────────────────────────────────────╯
@st.cache_data(ttl=300)
def call_api(sk_id: int) -> dict:
    r = requests.get(API_URL, params={"id_client": sk_id}, timeout=10)
    r.raise_for_status()
    return r.json()

try:
    with st.spinner("⏳ Appel API…"):
        payload = call_api(int(CID))
except Exception as e:
    st.error(f"Erreur API : {e}")
    st.stop()

PROBA = payload["proba"]
DEC   = "✅ Accordé" if payload["decision"] == 0 else "❌ Refusé"
row   = df.loc[CID]

# ╭────────────────────────────────────────────╮
# HEADER (score + jauge + décision)
# ╰────────────────────────────────────────────╯
col1, col2 = st.columns([1, 2])
fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=PROBA,
        number={'valueformat': ".0%"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': PALETTE_ACC[1] if PROBA > THR else PALETTE_ACC[0]},
            'threshold': {'line': {'color': "black", 'width': 3},
                          'thickness': 0.75, 'value': THR}}))
fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
col1.plotly_chart(fig_gauge, use_container_width=True)
col2.markdown(f"### {DEC}")

# ╭────────────────────────────────────────────╮
# 5 INDICATEURS CLEFS
# ╰────────────────────────────────────────────╯
def euro(x): return f"{x:,.0f} €".replace(",", " ")
a,b,c,d,e = st.columns(5)
a.metric("Âge", f"{int(-row.DAYS_BIRTH/365.25)} ans")
b.metric("Revenu", euro(row.AMT_INCOME_TOTAL))
c.metric("Crédit", euro(row.AMT_CREDIT))
d.metric("Annuité", euro(row.AMT_ANNUITY))
e.metric("Foyer", int(row.CNT_FAM_MEMBERS))

# ╭────────────────────────────────────────────╮
# COMPARAISON CLIENT vs POPULATION (scatter)
# ╰────────────────────────────────────────────╯
if pd.notna(row[X_VAR]) and pd.notna(row[Y_VAR]):
    others = df.drop(CID)
    fig = px.scatter(others, x=X_VAR, y=Y_VAR, opacity=0.25,
                     template="simple_white", height=500,
                     title=f"{X_VAR} vs {Y_VAR}")
    fig.update_traces(marker_size=5, marker_color=PALETTE_ACC[0])
    fig.add_scatter(x=[row[X_VAR]], y=[row[Y_VAR]],
                    mode="markers+text",
                    marker=dict(size=18, color=PALETTE_ACC[1],
                                line=dict(width=2, color="black")),
                    text=[str(CID)], textposition="top center",
                    showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ╭────────────────────────────────────────────╮
# GLOBAL SHAP (barre horizontale TOP‑15)
# ╰────────────────────────────────────────────╯
st.subheader("🌍 Importance globale (TOP 15)")
st.bar_chart(GLOBAL_S[::-1])              # Streamlit bar (accessible)

# ╭────────────────────────────────────────────╮
# LOCAL SHAP (waterfall)
# ╰────────────────────────────────────────────╯
st.subheader("🔎 Explication locale")
x_raw = row[NUM_COLS].to_frame().T
x_proc = pd.DataFrame(IMP.transform(x_raw), columns=NUM_COLS, index=[CID])

explainer = shap.TreeExplainer(CLF)
sv = explainer.shap_values(x_proc)[1] if isinstance(
        explainer.shap_values(x_proc), list) else explainer.shap_values(x_proc)

fig_local = shap.plots.waterfall(shap.Explanation(
        base_values=explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        values=sv[0], data=x_proc.iloc[0], feature_names=NUM_COLS),
        max_display=15, show=False)
st.pyplot(fig_local, use_container_width=True)

# ╭────────────────────────────────────────────╮
# TABLE COMPLÈTE
# ╰────────────────────────────────────────────╯
with st.expander("📄 Détail complet du dossier client"):
    st.dataframe(row.to_frame("valeur"), use_container_width=True)

st.caption("Prêt à dépenser – Dashboard v1.0")
