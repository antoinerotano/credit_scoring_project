# streamlit_app.py
# Dashboard Credit Scoring - Version Complète
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
API_BASE_URL = "https://credit-scoring-project-5d5k.onrender.com"
THRESHOLD = 0.206


# ╭──────────────────────────────────────────────────────────────╮
# Fonctions utilitaires
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_data
def load_data():
    """Charge et prépare les données des clients."""
    try:
        df = pd.read_parquet(FEAT_PATH)
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
        num_cols = [c for c in df.columns if df[c].dtype != "object" and c != "SK_ID_CURR"]
        return df.set_index("SK_ID_CURR"), num_cols
    except FileNotFoundError:
        st.error(f"Le fichier de données '{FEAT_PATH}' est introuvable. Assurez-vous qu'il est au bon endroit.")
        return pd.DataFrame(), []


@st.cache_data(ttl=300)
def get_prediction(client_id):
    """Récupère la prédiction de l'API pour un client donné."""
    api_url = f"{API_BASE_URL}/predict"
    try:
        response = requests.get(api_url, params={"id_client": client_id}, timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        st.error(f"Erreur de communication avec l'API de prédiction : {e}")
        return {"proba": 0.5, "decision": 1, "error": True}

def update_client_data(client_id, data_payload):
    """(SIMULATION) Envoie les données client modifiées à l'API via une requête PUT."""
    api_url = f"{API_BASE_URL}/client/{client_id}"
    
    # NOTE : Dans un cas réel, vous utiliseriez le code suivant :
    # try:
    #     response = requests.put(api_url, json=data_payload, timeout=10)
    #     response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
    #     return {"success": True, "message": "Informations client mises à jour avec succès !", "data": response.json()}
    # except requests.RequestException as e:
    #     return {"success": False, "message": f"Échec de la mise à jour : {e}"}

    # Pour la démonstration, nous simulons une réponse réussie
    print(f"--- SIMULATION API CALL ---\nEndpoint: PUT {api_url}\nPayload: {data_payload}\n--- END SIMULATION ---")
    return {"success": True, "message": "Informations client mises à jour avec succès ! (Simulation)"}


def format_euro(value):
    """Formate une valeur numérique en chaîne de caractères Euro."""
    return f"{value:,.0f} €".replace(",", " ") if pd.notna(value) else "—"

# ╭──────────────────────────────────────────────────────────────╮
# Chargement des données
# ╰──────────────────────────────────────────────────────────────╯
df, num_cols = load_data()
if df.empty:
    st.stop()

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
    wcag_mode = st.toggle("Mode Contraste Élevé (WCAG)", help="Active une palette de couleurs optimisée pour l'accessibilité.")

    st.subheader("📊 Graphiques")
    x_axis = st.selectbox("Axe X", num_cols, index=num_cols.index("AMT_CREDIT") if "AMT_CREDIT" in num_cols else 0)
    y_axis = st.selectbox("Axe Y", num_cols, index=num_cols.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in num_cols else 1)

    st.info(f"Seuil de décision: {THRESHOLD:.3f}")

# --- Définition des palettes de couleurs ---
if wcag_mode:
    colors = {"gauge_bar_ok": "#018E42", "gauge_bar_ko": "#D91E18", "gauge_step_ok": "#ABE0C2", "gauge_step_ko": "#F5B8B5", "gauge_threshold": "#D91E18", "scatter_client": "#FFBF17", "scatter_others": "#707070", "decision_ok_bg": "#018E42", "decision_ko_bg": "#D91E18", "decision_text": "#FFFFFF", "hist_vline": "#000000"}
else:
    colors = {"gauge_bar_ok": "green", "gauge_bar_ko": "red", "gauge_step_ok": "lightgreen", "gauge_step_ko": "lightcoral", "gauge_threshold": "red", "scatter_client": "red", "scatter_others": "lightblue", "decision_ok_bg": "#10b981", "decision_ko_bg": "#ef4444", "decision_text": "white", "hist_vline": "red"}

# --- Injection du CSS dynamique ---
st.markdown(f"""<style>.decision-ok{{background:{colors['decision_ok_bg']};color:{colors['decision_text']};padding:1rem;border-radius:8px;text-align:center;}}.decision-ko{{background:{colors['decision_ko_bg']};color:{colors['decision_text']};padding:1rem;border-radius:8px;text-align:center;}}</style>""", unsafe_allow_html=True)


# Récupération des données client et prédiction
client_data = df.loc[client_id]
prediction = get_prediction(client_id)
proba = prediction["proba"]
decision = prediction["decision"]

# ... (Le reste du code de l'affichage principal est identique) ...
# ╭──────────────────────────────────────────────────────────────╮
# Score et décision
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("🎯 Score de Crédit")
col1, col2, col3 = st.columns(3)
with col1:
    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=proba, title={'text': "Probabilité de Défaut"}, gauge={'axis': {'range': [0, 1]}, 'bar': {'color': colors['gauge_bar_ko'] if proba > THRESHOLD else colors['gauge_bar_ok']}, 'steps': [{'range': [0, THRESHOLD], 'color': colors['gauge_step_ok']}, {'range': [THRESHOLD, 1], 'color': colors['gauge_step_ko']}], 'threshold': {'line': {'color': colors['gauge_threshold'], 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD}}))
    fig_gauge.update_layout(height=250, margin=dict(t=40, b=40))
    st.plotly_chart(fig_gauge, use_container_width=True)
with col2:
    st.metric("Score", f"{proba:.1%}")
    st.metric("Distance / Seuil", f"{abs(proba - THRESHOLD):.1%}")
with col3:
    st.markdown(f'<div class="{"decision-ko" if decision == 1 else "decision-ok"}">{ "❌ CRÉDIT REFUSÉ" if decision == 1 else "✅ CRÉDIT ACCORDÉ"}</div>', unsafe_allow_html=True)

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
# ╰──────────────────────────────────────────────────────────────╯
st.subheader("📊 Positionnement du Client")
col_scatter, col_hist = st.columns([2, 1])
with col_scatter:
    others = df.drop(client_id)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=others[x_axis], y=others[y_axis], mode="markers", marker=dict(size=5, color=colors['scatter_others'], opacity=0.6), name="Autres clients"))
    fig_scatter.add_trace(go.Scatter(x=[client_data[x_axis]], y=[client_data[y_axis]], mode="markers", marker=dict(size=15, color=colors['scatter_client'], symbol="star", line=dict(width=2, color="black")), name="Client sélectionné"))
    fig_scatter.update_layout(title=f"{x_axis} vs {y_axis}", xaxis_title=x_axis, yaxis_title=y_axis, height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_scatter, use_container_width=True)
with col_hist:
    fig_hist = px.histogram(df, x=x_axis, nbins=30, title=f"Distribution de {x_axis}")
    fig_hist.add_vline(x=client_data[x_axis], line_dash="dash", line_color=colors['hist_vline'], line_width=3)
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# ╭──────────────────────────────────────────────────────────────╮
# Modification des Informations Client
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("📝 Modifier les informations du client"):
    with st.form(key="update_client_form"):
        st.write("Modifiez les informations ci-dessous et cliquez sur 'Mettre à jour' pour les enregistrer.")
        
        # Création des colonnes pour un affichage plus compact
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            new_income = st.number_input(
                "Revenu Annuel (€)",
                min_value=0,
                value=int(client_data.get("AMT_INCOME_TOTAL", 0)),
                step=1000,
                help="Revenu annuel total du client."
            )
            new_annuity = st.number_input(
                "Annuité du prêt (€)",
                min_value=0,
                value=int(client_data.get("AMT_ANNUITY", 0)),
                step=500,
                help="Montant de l'annuité du crédit demandé."
            )
        
        with form_col2:
            new_credit = st.number_input(
                "Montant du Crédit (€)",
                min_value=0,
                value=int(client_data.get("AMT_CREDIT", 0)),
                step=1000,
                help="Montant total du crédit demandé."
            )
            new_fam_members = st.number_input(
                "Taille du Foyer",
                min_value=1,
                value=int(client_data.get("CNT_FAM_MEMBERS", 1)),
                step=1,
                help="Nombre de personnes dans le foyer du client."
            )

        # Bouton de soumission du formulaire
        submitted = st.form_submit_button("Mettre à jour les informations")

        if submitted:
            # Création du payload avec les nouvelles données
            update_payload = {
                "AMT_INCOME_TOTAL": new_income,
                "AMT_CREDIT": new_credit,
                "AMT_ANNUITY": new_annuity,
                "CNT_FAM_MEMBERS": new_fam_members
            }
            
            # Appel de la fonction de mise à jour
            result = update_client_data(client_id, update_payload)
            
            if result["success"]:
                st.success(result["message"])
                # On vide le cache et on recharge la page pour voir les changements
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result["message"])


# ╭──────────────────────────────────────────────────────────────╮
# Détails
# ╰──────────────────────────────────────────────────────────────╯
with st.expander("🗂️ Afficher les détails complets du client (données brutes)"):
    st.dataframe(client_data.to_frame("Valeur").astype(str))

# Footer
st.markdown("---")
st.markdown("Dashboard développé pour **Prêt à Dépenser** - Transparence des décisions de crédit")
