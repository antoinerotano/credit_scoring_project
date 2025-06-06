# src/api/app.py
from pathlib import Path
import joblib, pandas as pd
from flask import Flask, request, jsonify

# ────────────────────────────────────────────────────────────────
#  Chemins vers le modèle et les features
# ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models_artifacts/model.joblib"
FEAT_PATH  = ROOT / "data/features.parquet"

FEAT_URL = os.getenv("FEATURES_URL")      # variable Render
if not FEAT_PATH.exists():
    FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Téléchargement des features depuis {FEAT_URL} …")
    urllib.request.urlretrieve(FEAT_URL, FEAT_PATH)
    print("✔️  Features téléchargées")

# ────────────────────────────────────────────────────────────────
#  Chargements au démarrage
# ────────────────────────────────────────────────────────────────
print("🔄 Chargement du modèle…")
model = joblib.load(MODEL_PATH)
print("✔️  Modèle chargé")

print("🔄 Chargement des features…")
df = pd.read_parquet(FEAT_PATH)

# 1) Nettoyage des colonnes indésirables
df = df.drop(columns=["TARGET", "index"], errors="ignore")

# 2) Met en index SK_ID_CURR, puis conserve uniquement les colonnes attendues
if "SK_ID_CURR" not in df.columns:
    raise ValueError("La colonne SK_ID_CURR est manquante dans features.parquet")

df = df.set_index("SK_ID_CURR")

expected_cols = list(model.feature_names_in_)      # colonnes vues au fit
X_full = df[expected_cols]                         # sélection

print("✔️  Features nettoyées :", X_full.shape)

# ID par défaut pour test rapide
DEFAULT_SK_ID = int(X_full.index.dropna().astype(int)[0])
print(f"ℹ️  ID client par défaut = {DEFAULT_SK_ID}")

# ────────────────────────────────────────────────────────────────
#  Application Flask
# ────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    """
    GET /predict?sk_id=<id_client>
    - Si sk_id omis → DEFAULT_SK_ID
    - Renvoie JSON {sk_id, proba, decision, default_used}
    """
    param = request.args.get("sk_id")
    if param is None:
        sk_id = DEFAULT_SK_ID
        default_used = True
    else:
        try:
            sk_id = int(param)
            default_used = False
        except ValueError:
            return jsonify(error="Paramètre sk_id invalide"), 400

    if sk_id not in X_full.index:
        return jsonify(error=f"SK_ID_CURR {sk_id} introuvable"), 404

    proba = float(model.predict_proba(X_full.loc[[sk_id]])[:, 1][0])
    decision = int(proba >= 0.206)

    return jsonify(
        sk_id=sk_id,
        proba=proba,
        decision=decision,
        default_used=default_used
    )

if __name__ == "__main__":
    # Lancement local : python -m src.api.app
    app.run(host="0.0.0.0", port=5000, debug=True)
