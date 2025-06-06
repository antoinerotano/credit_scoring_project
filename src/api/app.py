from pathlib import Path
import json, joblib, pandas as pd
from flask import Flask, request, jsonify

# ────────────────────────────────
#  Fichiers à charger
# ────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
MODEL_PATH  = ROOT / "models_artifacts/model.joblib"
THR_PATH    = ROOT / "models_artifacts/threshold.json"   # ← nouveau
FEAT_PATH   = ROOT / "data/features.parquet"

# ────────────────────────────────
#  Chargements au démarrage
# ────────────────────────────────
print("🔄  Chargement du modèle…")
model = joblib.load(MODEL_PATH)
print("✔️  Modèle OK")

print("🔄  Seuil optimal…")
best_thr = json.loads(THR_PATH.read_text())["threshold"]
print(f"✔️  Seuil chargé : {best_thr:.3f}")

print("🔄  Chargement des features…")
df = pd.read_parquet(FEAT_PATH)

# 1) nettoyage
df = df.drop(columns=["TARGET", "index"], errors="ignore")
if "SK_ID_CURR" not in df.columns:
    raise ValueError("🛑  SK_ID_CURR manquant dans features.parquet")

df = df.set_index("SK_ID_CURR")

# 2) alignement colonnes vs modèle
try:
    expected_cols = list(model.feature_names_in_)
except AttributeError:            # vieux sklearn
    expected_cols = df.columns.tolist()

X_full = df[expected_cols]
print("✔️  Features alignées :", X_full.shape)

DEFAULT_SK_ID = int(X_full.index[0])
print(f"ℹ️  ID client par défaut = {DEFAULT_SK_ID}")

# ────────────────────────────────
#  Flask
# ────────────────────────────────
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    """
    GET /predict?sk_id=<id_client>
      • sk_id omis → client par défaut
      • JSON : {sk_id, proba, decision, default_used}
    """
    param = request.args.get("sk_id")
    if param is None:
        sk_id, default_used = DEFAULT_SK_ID, True
    else:
        try:
            sk_id, default_used = int(param), False
        except ValueError:
            return jsonify(error="Paramètre sk_id invalide"), 400

    if sk_id not in X_full.index:
        return jsonify(error=f"SK_ID_CURR {sk_id} introuvable"), 404

    proba = float(model.predict_proba(X_full.loc[[sk_id]])[:, 1][0])
    decision = int(proba >= best_thr)

    return jsonify(
        sk_id=sk_id,
        proba=proba,
        threshold=best_thr,
        decision=decision,
        default_used=default_used
    )

if __name__ == "__main__":
    # Lancement local :
    #   python -m src.api.app
    app.run(host="0.0.0.0", port=5000, debug=True)
