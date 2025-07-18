# streamlit_app.py
# Dashboard Credit Scoring - Version Simplifiée
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from pathlib import Path

# ╭──────────────────────────────────────────────────────────────╮
# Configuration
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Dashboard Credit Scoring",
    page_icon="🏦",
    layout="wide"
)

# Paramètres
HERE = Path(__file__).resolve().parent
FEAT_PATH = HERE / "data" / "features_sample.parquet"
API_URL = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD = 0.206

# CSS simplifié
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.decision-ok { background: #10b981; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
.decision-ko { background: #ef4444; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ╭──────────────────────────────────────────────────────────────╮
# Fonctions utilitaires
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data
def load_data():
    df = pd.read_parquet(FEAT_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    num_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), num_cols

@st.cache_data(ttl=300)
def get_prediction(client_id):
    try:
        response = requests.get(API_URL, params={"id_client": client_id}, timeout=10)
        return response.json()
    except:
        return {"proba": 0.5, "decision": 1, "error": True}

def format_euro(value):
    return f"{value:,.0f} €".replace(",", " ") if pd.notna(value) else "—"

# ╭──────────────────────────────────────────────────────────────╮
# Chargement des données
# ╰──────────────────────────────────────────────────────────────╯
df, num_cols = load_data()

# ╭──────────────────────────────────────────────────────────────╮
# Interface
# ╰──────────────────────────────────────────────────────────────╯
st.title("🏦 Dashboard Credit Scoring")
st.markdown("**Prêt à Dépenser** - Analyse des décisions de crédit")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    client_id = st.selectbox("Client", sorted(df.index), format_func=lambda x: f"Client {x}")
    
    st.subheader("📊 Graphiques")
    x_axis = st.selectbox("Axe X", num_cols, index=num_cols.index("AMT_CREDIT") if "AMT_CREDIT" in num_cols else 0)
    y_axis = st.selectbox("Axe Y", num_cols, index=num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1)
    
    st.info(f"Seuil: {THRESHOLD:.3f}")

# Récupération des données
client_data = df.loc[client_id]
prediction = get_prediction(client_id)
proba = prediction["proba"]
decision = prediction["decision"]

# ╭──────────────────────────────────────────────────────────────╮
# Score et décision
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("🎯 Score de Crédit")

col1, col2, col3 = st.columns(3)

# Jauge
with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        title={'text': "Probabilité de Défaut"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "red" if proba > THRESHOLD else "green"},
            'steps': [
                {'range': [0, THRESHOLD], 'color': 'lightgreen'},
                {'range': [THRESHOLD, 1], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': THRESHOLD
            }
        }
    ))
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

# Métriques
with col2:
    st.metric("Score", f"{proba:.1%}")
    st.metric("Distance seuil", f"{abs(proba - THRESHOLD):.1%}")

# Décision
with col3:
    if decision == 0:
        st.markdown('<div class="decision-ok">✅ CRÉDIT ACCORDÉ</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="decision-ko">❌ CRÉDIT REFUSÉ</div>', unsafe_allow_html=True)

# ╭──────────────────────────────────────────────────────────────╮
# Profil client
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("👤 Profil Client")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    age = int(-client_data["DAYS_BIRTH"] / 365.25) if "DAYS_BIRTH" in client_data else None
    st.metric("Âge", f"{age} ans" if age else "—")

with col2:
    st.metric("Revenu", format_euro(client_data.get("AMT_INCOME_TOTAL")))

with col3:
    st.metric("Crédit", format_euro(client_data.get("AMT_CREDIT")))

with col4:
    st.metric("Annuité", format_euro(client_data.get("AMT_ANNUITY")))

with col5:
    family = int(client_data["CNT_FAM_MEMBERS"]) if "CNT_FAM_MEMBERS" in client_data else None
    st.metric("Foyer", f"{family} pers." if family else "—")

# ╭──────────────────────────────────────────────────────────────╮
# Graphiques de comparaison
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("📊 Positionnement du Client")

col_scatter, col_hist = st.columns([2, 1])

with col_scatter:
    # Scatter plot - Construction avec go.Figure pour contrôler l'ordre des couches
    others = df.drop(client_id)
    
    fig_scatter = go.Figure()
    
    # D'abord les autres clients (en arrière-plan)
    fig_scatter.add_trace(go.Scatter(
        x=others[x_axis], 
        y=others[y_axis],
        mode="markers",
        marker=dict(size=5, color="lightblue", opacity=0.6),
        name="Autres clients",
        showlegend=True
    ))
    
    # Ensuite le client sélectionné (au premier plan)
    fig_scatter.add_trace(go.Scatter(
        x=[client_data[x_axis]], 
        y=[client_data[y_axis]],
        mode="markers+text",
        marker=dict(size=15, color="red", symbol="circle", line=dict(width=2, color="darkred")),
        text=[f"Client {client_id}"],
        textposition="top center",
        textfont=dict(size=12, color="red"),
        name="Client sélectionné",
        showlegend=True
    ))
    
    fig_scatter.update_layout(
        title=f"{x_axis} vs {y_axis}",
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        height=400
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_hist:
    # Histogramme
    fig_hist = px.histogram(df, x=x_axis, nbins=30, title=f"Distribution {x_axis}")
    fig_hist.add_vline(x=client_data[x_axis], line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# Détails
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗂️ Détails du client"):
    st.dataframe(client_data.to_frame("Valeur"))

# Footer
st.markdown("---")
st.markdown("Dashboard développé pour **Prêt à Dépenser** - Transparence des décisions de crédit")
