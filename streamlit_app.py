# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit – Crédit Scoring Amélioré
# ────────────────────────────────────────────────────────────────
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import plotly.figure_factory as ff
from datetime import datetime
import json

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣  Config générale & accessibilité
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Dashboard Credit Scoring - Prêt à Dépenser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour l'accessibilité (WCAG conformité)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 1rem;
    text-align: center;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.5rem;
}

.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.decision-approved {
    background: linear-gradient(135deg, #4ade80 0%, #16a34a 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    font-size: 1.2rem;
    margin: 1rem 0;
}

.decision-rejected {
    background: linear-gradient(135deg, #f87171 0%, #dc2626 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    font-size: 1.2rem;
    margin: 1rem 0;
}

.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #374151;
    margin: 2rem 0 1rem 0;
    border-left: 4px solid #3b82f6;
    padding-left: 1rem;
}

.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #92400e;
}

/* Accessibilité - contraste élevé */
.high-contrast {
    background: #000000;
    color: #ffffff;
}

/* Focus visible pour navigation clavier */
.stSelectbox > div > div > div:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
}
</style>
""", unsafe_allow_html=True)

# Configuration
HERE = Path(__file__).resolve().parent
FEAT_PATH = HERE / "data" / "features_sample.parquet"
API_URL = "https://credit-scoring-project-5d5k.onrender.com/predict"
THRESHOLD = 0.206

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Fonctions utilitaires
# ╰──────────────────────────────────────────────────────────────╯
def format_currency(value):
    """Format en euros avec espaces"""
    if pd.isna(value):
        return "—"
    return f"{value:,.0f} €".replace(",", " ")

def format_percentage(value):
    """Format en pourcentage"""
    if pd.isna(value):
        return "—"
    return f"{value:.1%}"

def get_risk_level(proba, threshold=THRESHOLD):
    """Détermine le niveau de risque"""
    if proba < threshold * 0.5:
        return "Très Faible", "#10b981"
    elif proba < threshold * 0.8:
        return "Faible", "#3b82f6"
    elif proba < threshold:
        return "Modéré", "#f59e0b"
    elif proba < threshold * 1.5:
        return "Élevé", "#ef4444"
    else:
        return "Très Élevé", "#dc2626"

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Chargement des données
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner="📦 Chargement des données...")
def load_features(file: Path):
    try:
        df = pd.read_parquet(file)
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
        
        # Identification des colonnes numériques
        num_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
        
        return df.set_index("SK_ID_CURR"), num_cols
    except Exception as e:
        st.error(f"Erreur lors du chargement : {str(e)}")
        return None, []

# ╭──────────────────────────────────────────────────────────────╮
# 4️⃣  Appel API avec gestion d'erreur
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data(ttl=300)
def call_api(sk_id: int) -> dict:
    """Appel API avec gestion d'erreur robuste"""
    try:
        response = requests.get(
            API_URL, 
            params={"id_client": sk_id}, 
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"❌ Erreur API : {str(e)}")
        # Retourner des valeurs par défaut en cas d'erreur
        return {
            "proba": 0.5,
            "decision": 1,
            "default_used": True,
            "error": str(e)
        }

# ╭──────────────────────────────────────────────────────────────╮
# 5️⃣  Interface principale
# ╰──────────────────────────────────────────────────────────────╯

# En-tête principal
st.markdown('<h1 class="main-header">🏦 Dashboard Credit Scoring</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem;">Prêt à Dépenser - Analyse transparente des décisions de crédit</p>', unsafe_allow_html=True)

# Chargement des données
try:
    df, num_cols = load_features(FEAT_PATH)
    if df is None:
        st.error("❌ Impossible de charger les données")
        st.stop()
except Exception as e:
    st.error(f"❌ Erreur fatale : {str(e)}")
    st.stop()

# ╭──────────────────────────────────────────────────────────────╮
# 6️⃣  Sidebar - Configuration
# ╰──────────────────────────────────────────────────────────────╯
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Sélection du client
    client_ids = sorted(df.index.tolist())
    selected_client = st.selectbox(
        "📋 Sélectionner un client",
        client_ids,
        format_func=lambda x: f"Client {x}",
        help="Choisissez l'ID du client à analyser"
    )
    
    st.divider()
    
    # Paramètres de comparaison
    st.subheader("📊 Paramètres de visualisation")
    
    # Sélection des axes pour le graphique principal
    default_x = "AMT_CREDIT" if "AMT_CREDIT" in num_cols else num_cols[0]
    default_y = "AMT_INCOME_TOTAL" if "AMT_INCOME_TOTAL" in num_cols else num_cols[1]
    
    x_axis = st.selectbox("Axe X", num_cols, 
                         index=num_cols.index(default_x) if default_x in num_cols else 0)
    y_axis = st.selectbox("Axe Y", num_cols, 
                         index=num_cols.index(default_y) if default_y in num_cols else 1)
    
    # Filtre pour comparaison
    st.subheader("🔍 Filtres de comparaison")
    comparison_feature = st.selectbox(
        "Variable pour grouper",
        ["Tous les clients"] + num_cols,
        help="Choisir une variable pour comparer le client à un sous-groupe"
    )
    
    if comparison_feature != "Tous les clients":
        quartiles = df[comparison_feature].quantile([0.25, 0.5, 0.75]).values
        comparison_range = st.select_slider(
            "Plage de comparaison",
            options=["Q1 (25%)", "Q2 (50%)", "Q3 (75%)", "Q4 (100%)"],
            value="Q2 (50%)"
        )
    
    st.divider()
    
    # Informations système
    st.markdown(f"""
    <div class="info-box">
        <h4>ℹ️ Informations</h4>
        <p><strong>Seuil de décision:</strong> {THRESHOLD:.3f}</p>
        <p><strong>Clients total:</strong> {len(df):,}</p>
        <p><strong>Variables:</strong> {len(num_cols)}</p>
    </div>
    """, unsafe_allow_html=True)

# ╭──────────────────────────────────────────────────────────────╮
# 7️⃣  Appel API et récupération des données
# ╰──────────────────────────────────────────────────────────────╯
with st.spinner("⏳ Analyse en cours..."):
    api_result = call_api(int(selected_client))
    
proba = api_result["proba"]
decision = api_result["decision"]
default_used = api_result.get("default_used", False)

# Récupération des données du client
if selected_client in df.index:
    client_data = df.loc[selected_client]
else:
    st.error(f"❌ Client {selected_client} non trouvé dans les données")
    st.stop()

# ╭──────────────────────────────────────────────────────────────╮
# 8️⃣  Section principale - Score et décision
# ╰──────────────────────────────────────────────────────────────╯
st.markdown('<div class="section-header">🎯 Score de Crédit et Décision</div>', unsafe_allow_html=True)

# Colonnes principales
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    # Jauge de score
    risk_level, risk_color = get_risk_level(proba)
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilité de Défaut", 'font': {'size': 20}},
        delta = {'reference': THRESHOLD, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
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
    
    fig_gauge.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # Métriques clés
    st.metric(
        label="Score de Risque",
        value=f"{proba:.1%}",
        delta=f"{(proba - THRESHOLD)*100:+.1f}pp"
    )
    
    st.metric(
        label="Niveau de Risque",
        value=risk_level
    )
    
    # Distance au seuil
    distance_seuil = abs(proba - THRESHOLD)
    st.metric(
        label="Distance au Seuil",
        value=f"{distance_seuil:.1%}"
    )

with col3:
    # Décision finale
    if decision == 0:
        st.markdown(f"""
        <div class="decision-approved">
            ✅ CRÉDIT ACCORDÉ
            <br><small>Le client présente un risque acceptable</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="decision-rejected">
            ❌ CRÉDIT REFUSÉ
            <br><small>Le client présente un risque trop élevé</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Explication de la décision
    if proba < THRESHOLD * 0.7:
        explanation = "Score très favorable - Risque minimal"
    elif proba < THRESHOLD:
        explanation = "Score favorable - Risque contrôlé"
    elif proba < THRESHOLD * 1.3:
        explanation = "Score limite - Risque élevé mais gérable"
    else:
        explanation = "Score défavorable - Risque trop important"
    
    st.info(f"💡 **Explication:** {explanation}")

# Avertissement si données par défaut
if default_used:
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ <strong>Attention:</strong> L'API a utilisé des données par défaut car le client n'a pas été trouvé.
    </div>
    """, unsafe_allow_html=True)

# ╭──────────────────────────────────────────────────────────────╮
# 9️⃣  Profil du client
# ╰──────────────────────────────────────────────────────────────╯
st.markdown('<div class="section-header">👤 Profil du Client</div>', unsafe_allow_html=True)

# Informations principales
info_cols = st.columns(6)

with info_cols[0]:
    age = int(round(-client_data["DAYS_BIRTH"] / 365.25)) if "DAYS_BIRTH" in client_data and pd.notna(client_data["DAYS_BIRTH"]) else None
    st.metric("👤 Âge", f"{age} ans" if age else "—")

with info_cols[1]:
    income = client_data.get("AMT_INCOME_TOTAL", 0)
    st.metric("💰 Revenu Annuel", format_currency(income))

with info_cols[2]:
    credit = client_data.get("AMT_CREDIT", 0)
    st.metric("🏠 Montant Crédit", format_currency(credit))

with info_cols[3]:
    annuity = client_data.get("AMT_ANNUITY", 0)
    st.metric("💳 Annuité", format_currency(annuity))

with info_cols[4]:
    family = int(client_data["CNT_FAM_MEMBERS"]) if "CNT_FAM_MEMBERS" in client_data and pd.notna(client_data["CNT_FAM_MEMBERS"]) else None
    st.metric("👨‍👩‍👧‍👦 Foyer", f"{family} pers." if family else "—")

with info_cols[5]:
    # Ratio crédit/revenu
    if income > 0 and credit > 0:
        ratio = credit / income
        st.metric("📊 Ratio C/R", f"{ratio:.1f}x")
    else:
        st.metric("📊 Ratio C/R", "—")

# ╭──────────────────────────────────────────────────────────────╮
# 🔟  Analyse comparative
# ╰──────────────────────────────────────────────────────────────╯
st.markdown('<div class="section-header">📈 Analyse Comparative</div>', unsafe_allow_html=True)

# Graphique de positionnement
col_graph, col_hist = st.columns([2, 1])

with col_graph:
    # Sélection des données de comparaison
    if comparison_feature == "Tous les clients":
        comparison_data = df
        title_suffix = "tous les clients"
    else:
        # Filtrage par quartile
        quartile_ranges = {
            "Q1 (25%)": (0, 0.25),
            "Q2 (50%)": (0.25, 0.5),
            "Q3 (75%)": (0.5, 0.75),
            "Q4 (100%)": (0.75, 1.0)
        }
        
        q_low, q_high = quartile_ranges[comparison_range]
        feature_quantiles = df[comparison_feature].quantile([q_low, q_high])
        
        comparison_data = df[
            (df[comparison_feature] >= feature_quantiles.iloc[0]) & 
            (df[comparison_feature] <= feature_quantiles.iloc[1])
        ]
        title_suffix = f"{comparison_range} de {comparison_feature}"
    
    # Graphique scatter
    if selected_client in comparison_data.index:
        others = comparison_data.drop(selected_client, errors="ignore")
    else:
        others = comparison_data
    
    fig_scatter = px.scatter(
        others,
        x=x_axis,
        y=y_axis,
        opacity=0.3,
        height=500,
        template="plotly_white",
        title=f"Position du client vs {title_suffix}",
        color_discrete_sequence=["#3b82f6"]
    )
    
    # Point du client sélectionné
    if selected_client in df.index:
        x_val = client_data[x_axis]
        y_val = client_data[y_axis]
        
        if not (pd.isna(x_val) or pd.isna(y_val)):
            fig_scatter.add_scatter(
                x=[x_val],
                y=[y_val],
                mode="markers+text",
                marker=dict(
                    size=15,
                    color="red",
                    line=dict(width=2, color="darkred"),
                    symbol="star"
                ),
                text=[f"Client {selected_client}"],
                textposition="top center",
                name="Client sélectionné",
                showlegend=False
            )
    
    fig_scatter.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        showlegend=False
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_hist:
    # Histogramme de distribution
    feature_for_hist = x_axis
    
    fig_hist = px.histogram(
        comparison_data,
        x=feature_for_hist,
        nbins=30,
        title=f"Distribution de {feature_for_hist}",
        opacity=0.7,
        color_discrete_sequence=["#3b82f6"]
    )
    
    # Ligne verticale pour le client
    if selected_client in df.index and not pd.isna(client_data[feature_for_hist]):
        fig_hist.add_vline(
            x=client_data[feature_for_hist],
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Client {selected_client}"
        )
    
    fig_hist.update_layout(
        xaxis_title=feature_for_hist,
        yaxis_title="Fréquence",
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣1️⃣  Analyse bivariée
# ╰──────────────────────────────────────────────────────────────╯
st.markdown('<div class="section-header">🔍 Analyse Bivariée</div>', unsafe_allow_html=True)

col_bivar1, col_bivar2 = st.columns(2)

with col_bivar1:
    # Sélection des variables pour l'analyse bivariée
    bivar_x = st.selectbox("Variable X pour analyse bivariée", num_cols, 
                          index=num_cols.index(default_x) if default_x in num_cols else 0)

with col_bivar2:
    bivar_y = st.selectbox("Variable Y pour analyse bivariée", num_cols, 
                          index=num_cols.index(default_y) if default_y in num_cols else 1)

# Création du graphique bivarié avec densité
if bivar_x != bivar_y:
    fig_bivar = px.scatter(
        df,
        x=bivar_x,
        y=bivar_y,
        opacity=0.4,
        title=f"Analyse bivariée: {bivar_x} vs {bivar_y}",
        template="plotly_white",
        height=400
    )
    
    # Ajout du point client
    if selected_client in df.index:
        x_val = client_data[bivar_x]
        y_val = client_data[bivar_y]
        
        if not (pd.isna(x_val) or pd.isna(y_val)):
            fig_bivar.add_scatter(
                x=[x_val],
                y=[y_val],
                mode="markers",
                marker=dict(size=12, color="red", symbol="star"),
                name=f"Client {selected_client}",
                showlegend=True
            )
    
    st.plotly_chart(fig_bivar, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣2️⃣  Informations détaillées
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗂️ Informations détaillées du client", expanded=False):
    st.markdown("### Toutes les variables disponibles")
    
    # Création d'un DataFrame pour l'affichage
    client_df = client_data.to_frame("Valeur")
    client_df["Variable"] = client_df.index
    client_df = client_df.reset_index(drop=True)
    client_df = client_df[["Variable", "Valeur"]]
    
    # Formatage des valeurs
    def format_value(val):
        if pd.isna(val):
            return "—"
        elif isinstance(val, (int, float)):
            if abs(val) > 1000:
                return f"{val:,.0f}".replace(",", " ")
            else:
                return f"{val:.2f}"
        else:
            return str(val)
    
    client_df["Valeur"] = client_df["Valeur"].apply(format_value)
    
    st.dataframe(client_df, use_container_width=True, height=400)

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣3️⃣  Footer et informations
# ╰──────────────────────────────────────────────────────────────╯
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p><strong>Dashboard Credit Scoring</strong> - Prêt à Dépenser</p>
    <p>Développé pour la transparence des décisions de crédit</p>
    <p><small>Conforme aux standards d'accessibilité WCAG 2.1</small></p>
</div>
""", unsafe_allow_html=True)
