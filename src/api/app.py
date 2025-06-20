# ─────────────────────────────  src/api/app.py  ─────────────────────────────
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, jsonify, abort

# ────────────────────────────────
#  Chemins
# ────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models_artifacts" / "model.joblib"
FEAT_PATH  = ROOT / "data" / "features_sample.parquet"      # ← on charge le set complet

# ────────────────────────────────
#  Chargements au démarrage
# ────────────────────────────────
print("🔄 Chargement du modèle…")
model = joblib.load(MODEL_PATH)
print("✔️  Modèle chargé")

print("🔄 Chargement des features…")
df = pd.read_parquet(FEAT_PATH)

# 1) nettoyage (on enlève éventuellement TARGET / index)
df = df.drop(columns=["TARGET", "index"], errors="ignore")

# 2) mise en index
if "SK_ID_CURR" not in df.columns:
    raise RuntimeError("La colonne 'SK_ID_CURR' est absente de features_sample.parquet")

df = df.set_index("SK_ID_CURR")

# 3) on conserve uniquement les colonnes vues par le modèle
expected_cols = list(model.feature_names_in_)
missing = set(expected_cols) - set(df.columns)
if missing:
    raise RuntimeError(f"Colonnes manquantes dans features_sample.parquet : {missing}")

X_full = df[expected_cols]
print("✔️  Features nettoyées :", X_full.shape)

# petit ID par défaut pour un test rapide (ex : dans un navigateur)
DEFAULT_SK_ID = int(X_full.index[0])
print(f"ℹ️  ID client par défaut = {DEFAULT_SK_ID}")

# seuil retenu lors de l’analyse métier
THRESHOLD = 0.206

# ────────────────────────────────
#  Application Flask
# ────────────────────────────────
app = Flask(__name__)


@app.get("/predict")
def predict():
    """
    GET /predict?id_client=<ID>
    - id_client obligatoire (sinon on renvoie le DEFAULT_SK_ID ; flag default_used=True)
    - 404 si l’ID n’existe pas dans X_full
    - JSON : {sk_id, proba, decision, default_used}
    """
    param = request.args.get("id_client")
    if param is None:
        sk_id = DEFAULT_SK_ID
        default_used = True
    else:
        try:
            sk_id = int(param)
            default_used = False
        except (TypeError, ValueError):
            abort(400, "'id_client' doit être un entier")

    if sk_id not in X_full.index:
        abort(404, f"id_client {sk_id} introuvable dans les features")

    proba = float(model.predict_proba(X_full.loc[[sk_id]])[0, 1])
    decision = int(proba >= THRESHOLD)

    return jsonify(
        sk_id=sk_id,
        proba=proba,
        decision=decision,
        default_used=default_used,
    )


if __name__ == "__main__":
    # Lancement local :  python -m src.api.app
    app.run(host="0.0.0.0", port=5000, debug=True)
