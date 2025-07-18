# streamlit_app.py
# Dashboard Credit Scoring - Version avec Thème Adaptatif
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from pathlib import Path

# ╭──────────────────────────────────────────────────────────────╮
# Configuration de la page (doit être la première commande Streamlit)
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Dashboard Credit Scoring",
    page_icon="🏦",
    layout="wide"
)

# ╭──────────────────────────────────────────────────────────────╮
# Paramètres et Fonctions
# ╰──────────────────────────────────────────────────────────────╯
# Paramètres
HERE = Path(__file__).resolve().parent
FEAT_PATH = HERE / "data" / "features_sample.parquet"
API_URL = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD = 0.206

# Fonctions utilitaires
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
# Chargement des données et Interface
# ╰──────────────────────────────────────────────────────────────╯
df, num_cols = load_data()

st.title("🏦 Dashboard Credit Scoring")
st.markdown("**Prêt à Dépenser** - Analyse des décisions de crédit")

# --- Barre Latérale (Sidebar) ---
with st.sidebar:
    st.header("👤 Sélection du Client")
    client_id = st.selectbox("Client", sorted(df.index), format_func=lambda x: f"Client {x}")

    st.header("🎨 Thème d'Affichage")
    theme_choice = st.radio(
        "Choisissez un thème",
        ["Clair (Défaut)", "Sombre (Contraste élevé)"],
        label_visibility="collapsed"
    )
    dark_mode = (theme_choice == "Sombre (Contraste élevé)")

    st.header("📊 Configuration des Axes")
    x_axis = st.selectbox("Axe X", num_cols, index=num_cols.index("AMT_CREDIT") if "AMT_CREDIT" in num_cols else 0)
    y_axis = st.selectbox("Axe Y", num_cols, index=num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1)

    st.info(f"Seuil de décision: {THRESHOLD:.3f}")

# --- Définition des palettes de couleurs et du thème ---
if dark_mode:
    # --- THÈME SOMBRE (WCAG) ---
    plot_template = "plotly_dark"
    colors = {
        "app_bg": "#0E1117",
        "text": "#FAFAFA",
        "gauge_bar_ok": "#3DD56D",      # Vert vif
        "gauge_bar_ko": "#FF6B6B",      # Rouge/rose vif
        "gauge_step_ok": "#1E5832",
        "gauge_step_ko": "#6B1717",
        "gauge_threshold": "#FF6B6B",
        "scatter_client": "#FFD700",     # Or/Jaune vif
        "scatter_others": "#808080",
        "decision_ok_bg": "#006400",     # Vert foncé
        "decision_ko_bg": "#8B0000",     # Rouge foncé
        "decision_text": "#FFFFFF",
        "hist_vline": "#FFD700"
    }
    # Injection du CSS pour le thème sombre global
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="st-"] {{
        font-family: 'Roboto', sans-serif;
    }}
    .stApp {{
        background-color: {colors['app_bg']};
        color: {colors['text']};
    }}
    .st-emotion-cache-16txtl3 {{
        color: {colors['text']};
    }}
    h1, h2, h3 {{
        color: {colors['text']};
    }}
    .decision-ok {{ background: {colors['decision_ok_bg']}; color: {colors['decision_text']}; padding: 1.1rem; border-radius: 8px; text-align: center; font-weight: bold; border: 1px solid {colors['gauge_bar_ok']}; }}
    .decision-ko {{ background: {colors['decision_ko_bg']}; color: {colors['decision_text']}; padding: 1.1rem; border-radius: 8px; text-align: center; font-weight: bold; border: 1px solid {colors['gauge_bar_ko']};}}
    </style>
    """, unsafe_allow_html=True)
else:
    # --- THÈME CLAIR (Défaut) ---
    plot_template = "plotly_white"
    colors = {
        "gauge_bar_ok": "#28a745",
        "gauge_bar_ko": "#dc3545",
        "gauge_step_ok": "#d4edda",
        "gauge_step_ko": "#f8d7da",
        "gauge_threshold": "#dc3545",
        "scatter_client": "#ff4b4b",
        "scatter_others": "#09abeb",
        "decision_ok_bg": "#28a745",
        "decision_ko_bg": "#dc3545",
        "decision_text": "white",
        "hist_vline": "#ff4b4b"
    }
    st.markdown(f"""
    <style>
    .decision-ok {{ background: {colors['decision_ok_bg']}; color: {colors['decision_text']}; padding: 1.1rem; border-radius: 8px; text-align: center; font-weight: bold; }}
    .decision-ko {{ background: {colors['decision_ko_bg']}; color: {colors['decision_text']}; padding: 1.1rem; border-radius: 8px; text-align: center; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# --- Récupération des données et prédiction ---
client_data = df.loc[client_id]
prediction = get_prediction(client_id)
proba = prediction["proba"]
decision = prediction["decision"]

# ╭──────────────────────────────────────────────────────────────╮
# Score et décision
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("🎯 Score de Crédit")
col1, col2, col3 = st.columns([2, 1, 1])

# Jauge
with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=proba,
        title={'text': "Probabilité de Défaut"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': colors['gauge_bar_ko'] if proba > THRESHOLD else colors['gauge_bar_ok']},
            'steps': [
                {'range': [0, THRESHOLD], 'color': colors['gauge_step_ok']},
                {'range': [THRESHOLD, 1], 'color': colors['gauge_step_ko']}
            ],
            'threshold': {'line': {'color': colors['gauge_threshold'], 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD}
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=40, b=40), template=plot_template, paper_bgcolor='rgba(0,0,0,0)', font_color=colors.get("text"))
    st.plotly_chart(fig_gauge, use_container_width=True)

# Métriques et Décision
with col2:
    st.metric("Score Client", f"{proba:.1%}")
    st.metric("Distance / Seuil", f"{abs(proba - THRESHOLD):.1%}")

with col3:
    st.write("Décision Finale")
    if decision == 0:
        st.markdown('<div class="decision-ok">✅ CRÉDIT ACCORDÉ</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="decision-ko">❌ CRÉDIT REFUSÉ</div>', unsafe_allow_html=True)

# ╭──────────────────────────────────────────────────────────────╮
# Profil client et Graphiques de comparaison
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("📊 Positionnement du Client")
col_scatter, col_hist = st.columns([2, 1])

with col_scatter:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df[x_axis], y=df[y_axis], mode="markers",
        marker=dict(size=5, color=colors['scatter_others'], opacity=0.4), name="Autres clients"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[client_data[x_axis]], y=[client_data[y_axis]], mode="markers",
        marker=dict(size=15, color=colors['scatter_client'], symbol="star", line=dict(width=2, color="black")), name="Client sélectionné"
    ))
    fig_scatter.update_layout(
        title=f"{x_axis} vs {y_axis}", xaxis_title=x_axis, yaxis_title=y_axis, height=450, template=plot_template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor='rgba(0,0,0,0)', font_color=colors.get("text")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_hist:
    fig_hist = px.histogram(df, x=x_axis, nbins=30, title=f"Distribution de {x_axis}", template=plot_template)
    fig_hist.add_vline(x=client_data[x_axis], line_dash="dash", line_color=colors['hist_vline'], line_width=3, annotation_text="Client", annotation_position="top left")
    fig_hist.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font_color=colors.get("text"))
    st.plotly_chart(fig_hist, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# Informations détaillées
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗂️ Afficher les informations détaillées du client"):
    st.dataframe(client_data.to_frame("Valeur").astype(str))

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Dashboard développé pour Prêt à Dépenser</p>", unsafe_allow_html=True)
