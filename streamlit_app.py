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


# ╭──────────────────────────────────────────────────────────────╮
# Fonctions utilitaires
# ╰──────────────────────────────────────────────────────────────╮
@st.cache_data
def load_data():
    """Charge et prépare les données des clients."""
    df = pd.read_parquet(FEAT_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    num_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
    return df.set_index("SK_ID_CURR"), num_cols

@st.cache_data(ttl=300)
def get_prediction(client_id):
    """Récupère la prédiction de l'API pour un client donné."""
    try:
        response = requests.get(API_URL, params={"id_client": client_id}, timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {"proba": 0.5, "decision": 1, "error": True}

def format_euro(value):
    """Formate une valeur numérique en chaîne de caractères Euro."""
    return f"{value:,.0f} €".replace(",", " ") if pd.notna(value) else "—"

# ╭──────────────────────────────────────────────────────────────╮
# Chargement des données
# ╰──────────────────────────────────────────────────────────────╮
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

    st.subheader("🎨 Affichage")
    # Ajout du Toggle pour le mode WCAG
    wcag_mode = st.toggle("Mode Contraste Élevé (WCAG)", help="Active une palette de couleurs optimisée pour l'accessibilité.")

    st.subheader("📊 Graphiques")
    x_axis = st.selectbox("Axe X", num_cols, index=num_cols.index("AMT_CREDIT") if "AMT_CREDIT" in num_cols else 0)
    y_axis = st.selectbox("Axe Y", num_cols, index=num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1)

    st.info(f"Seuil de décision: {THRESHOLD:.3f}")

# --- Définition des palettes de couleurs ---
if wcag_mode:
    # Palette WCAG (Contraste Élevé avec plus de distinction)
    colors = {
        "gauge_bar_ok": "#008000",       # Vert forêt profond
        "gauge_bar_ko": "#800000",       # Rouge bordeaux foncé
        "gauge_step_ok": "#32CD32",      # Vert lime vif
        "gauge_step_ko": "#FF4500",      # Orange brûlé (très distinct du rouge)
        "gauge_threshold": "#000000",    # Noir
        "scatter_client": "#FFFF00",     # Jaune fluo éclatant
        "scatter_others": "#333333",     # Gris très foncé
        "decision_ok_bg": "#008000",     # Vert forêt profond
        "decision_ko_bg": "#800000",     # Rouge bordeaux foncé
        "decision_text": "#FFFFFF",      # Texte blanc
        "hist_vline": "#000000"          # Ligne noire
    }
else:
    # Palette par défaut (esthétique avec des couleurs plus douces et des accents vifs)
    colors = {
        "gauge_bar_ok": "#2ECC71",       # Vert frais
        "gauge_bar_ko": "#E74C3C",       # Rouge passion
        "gauge_step_ok": "#A8E6CF",      # Vert pomme clair
        "gauge_step_ko": "#FFD3B6",      # Rose saumon clair
        "gauge_threshold": "#C0392B",    # Rouge plus profond
        "scatter_client": "#FFC300",     # Jaune d'or vibrant
        "scatter_others": "#ADD8E6",     # Bleu ciel doux
        "decision_ok_bg": "#2ECC71",     # Vert frais
        "decision_ko_bg": "#E74C3C",     # Rouge passion
        "decision_text": "#FFFFFF",      # Blanc
        "hist_vline": "#36454F"          # Gris anthracite
    }

# --- Injection du CSS dynamique ---
st.markdown(f"""
<style>
.decision-ok {{ background: {colors['decision_ok_bg']}; color: {colors['decision_text']}; padding: 1rem; border-radius: 8px; text-align: center; }}
.decision-ko {{ background: {colors['decision_ko_bg']}; color: {colors['decision_text']}; padding: 1rem; border-radius: 8px; text-align: center; }}
</style>
""", unsafe_allow_html=True)


# Récupération des données client et prédiction
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
            'bar': {'color': colors['gauge_bar_ko'] if proba > THRESHOLD else colors['gauge_bar_ok']},
            'steps': [
                {'range': [0, THRESHOLD], 'color': colors['gauge_step_ok']},
                {'range': [THRESHOLD, 1], 'color': colors['gauge_step_ko']}
            ],
            'threshold': {
                'line': {'color': colors['gauge_threshold'], 'width': 4},
                'thickness': 0.75,
                'value': THRESHOLD
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=40, b=40))
    st.plotly_chart(fig_gauge, use_container_width=True)

# Métriques
with col2:
    st.metric("Score", f"{proba:.1%}")
    st.metric("Distance / Seuil", f"{abs(proba - THRESHOLD)::.1%}")

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
    age = int(-client_data.get("DAYS_BIRTH", 0) / 365.25)
    st.metric("Âge", f"{age} ans" if age else "—")
with col2:
    st.metric("Revenu", format_euro(client_data.get("AMT_INCOME_TOTAL")))
with col3:
    st.metric("Crédit", format_euro(client_data.get("AMT_CREDIT")))
with col4:
    st.metric("Annuité", format_euro(client_data.get("AMT_ANNUITY")))
with col5:
    family = client_data.get("CNT_FAM_MEMBERS")
    st.metric("Foyer", f"{int(family)} pers." if pd.notna(family) else "—")


# ╭──────────────────────────────────────────────────────────────╮
# Graphiques de comparaison
# ╰──────────────────────────────────────────────────────────────╮
st.subheader("📊 Positionnement du Client")

col_scatter, col_hist = st.columns([2, 1])

with col_scatter:
    others = df.drop(client_id)
    fig_scatter = go.Figure()

    # Autres clients (arrière-plan)
    fig_scatter.add_trace(go.Scatter(
        x=others[x_axis], y=others[y_axis], mode="markers",
        marker=dict(size=5, color=colors['scatter_others'], opacity=0.6),
        name="Autres clients"
    ))
    # Client sélectionné (premier plan)
    fig_scatter.add_trace(go.Scatter(
        x=[client_data[x_axis]], y=[client_data[y_axis]], mode="markers",
        marker=dict(size=15, color=colors['scatter_client'], symbol="star", line=dict(width=2, color="black")),
        name="Client sélectionné"
    ))
    fig_scatter.update_layout(
        title=f"{x_axis} vs {y_axis}", xaxis_title=x_axis, yaxis_title=y_axis, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_hist:
    # Histogramme
    fig_hist = px.histogram(df, x=x_axis, nbins=30, title=f"Distribution de {x_axis}")
    fig_hist.add_vline(x=client_data[x_axis], line_dash="dash", line_color=colors['hist_vline'], line_width=3)
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# Détails
# ╰──────────────────────────────────────────────────────────────╮
with st.expander("🗂️ Afficher les détails complets du client"):
    st.dataframe(client_data.to_frame("Valeur").astype(str))

# Footer
st.markdown("---")
st.markdown("Dashboard développé pour **Prêt à Dépenser** - Transparence des décisions de crédit")
