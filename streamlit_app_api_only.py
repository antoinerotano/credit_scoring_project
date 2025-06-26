# streamlit_app.py
# ────────────────────────────────────────────────────────────────
# Dashboard Streamlit – Crédit Scoring (full-API version)
# ────────────────────────────────────────────────────────────────
import streamlit as st
import requests

# ╭──────────────────────────────────────────────────────────────╮
# 1️⃣  Config générale
# ╰──────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Credit-Scoring Dashboard",
    page_icon="📊",
    layout="centered",
)

API_URL           = "https://credit-scoring-project-5d5k.onrender.com/predict"
DECISION_THRESHOLD = 0.206        # rappel (même que sur le back)

st.title("📊 Credit-Scoring – Résultat instantané")

# ╭──────────────────────────────────────────────────────────────╮
# 2️⃣  Saisie de l’ID client
# ╰──────────────────────────────────────────────────────────────╯
cid = st.text_input(
    label="Entrez l’identifiant client",
    placeholder="Ex. 285117",
    value="285117",
)

# petit rappel du format
st.caption(
    "Vous trouverez des ID valides dans le set de test Home-Credit "
    "(ex. 100038, 171430, 285117, etc.)."
)

btn = st.button("🔎 Obtenir la prédiction")

# ╭──────────────────────────────────────────────────────────────╮
# 3️⃣  Appel API
# ╰──────────────────────────────────────────────────────────────╯
if btn:
    # vérification rapide
    if not cid.strip().isdigit():
        st.error("Veuillez entrer un identifiant numérique.")
        st.stop()

    with st.spinner("⏳ Appel à l’API…"):
        try:
            resp = requests.get(API_URL, params={"id_client": int(cid)}, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as e:
            st.error(f"Erreur côté API : {e}")
            st.stop()

    # ╭──────────────────────────────────────────────────────────╮
    # 4️⃣  Affichage du résultat
    # ╰──────────────────────────────────────────────────────────╯
    proba        = payload["proba"]
    decision     = payload["decision"]         # 0 = accord, 1 = refus
    default_used = payload["default_used"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilité de défaut", f"{proba:.2%}")
    c2.metric("Décision", "Accord" if decision == 0 else "Refus")
    c3.metric("Seuil modèle", f"{DECISION_THRESHOLD:.3f}")

    st.info(
        f"**Résultat global :** "
        f"{'✅ Crédit accordé' if decision == 0 else '❌ Crédit refusé'}",
        icon="ℹ️",
    )

    if default_used:
        st.warning(
            "⚠️ L’API a utilisé son ID par défaut "
            "(l’identifiant demandé n’existe pas dans les features)."
        )

    # ╭──────────────────────────────────────────────────────────╮
    # 5️⃣  JSON complet (debug / développeurs)
    # ╰──────────────────────────────────────────────────────────╯
    with st.expander("Voir la réponse brute JSON"):
        st.json(payload)
