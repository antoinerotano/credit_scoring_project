import os, requests, shutil, tempfile
from pathlib import Path
import joblib, pandas as pd
import pyarrow.dataset as ds          # <- NEW
from flask import Flask, request, jsonify

ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models_artifacts" / "model.joblib"
DATA_DIR   = ROOT / "data"
PARQUET    = DATA_DIR / "features.parquet"

FEAT_URL   = os.getenv("FEATURES_URL")  # lien Dropbox ?dl=1

# ───────────── Téléchargement si besoin ─────────────
if not PARQUET.exists():
    print(f"⬇️  Téléchargement des features…")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    with requests.get(FEAT_URL, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    shutil.move(tmp, PARQUET)
    print("✔️  Features téléchargées")

# ───────────── Ouverture paresseuse ─────────────
print("🔄 Ouverture du dataset Parquet (lazy)…")
dataset = ds.dataset(PARQUET)          # ne charge rien en RAM ici
FEATURE_COLS = joblib.load(MODEL_PATH).feature_names_in_

print("🔄 Chargement du modèle…")
model = joblib.load(MODEL_PATH)
print("✔️  Modèle chargé")

# ───────────── Flask API ─────────────
app = Flask(__name__)

def get_features(sk_id: int) -> pd.DataFrame:
    """
    Lit UNE SEULE ligne dans le Parquet, retourne DataFrame (1, n_features)
    """
    table = dataset.to_table(
        filter=ds.field("SK_ID_CURR") == sk_id,
        columns=["SK_ID_CURR", *FEATURE_COLS]     # on ne lit que ce qu'il faut
    )
    if table.num_rows == 0:
        return None
    df = table.to_pandas()
    df = df.set_index("SK_ID_CURR")
    return df

@app.route("/predict", methods=["GET"])
def predict():
    """
    GET /predict?sk_id=<id_client>
    """
    try:
        sk_id = int(request.args.get("sk_id", 0))
    except (ValueError, TypeError):
        return jsonify(error="Paramètre sk_id invalide"), 400

    X = get_features(sk_id)
    if X is None:
        return jsonify(error=f"SK_ID_CURR {sk_id} introuvable"), 404

    proba    = float(model.predict_proba(X)[:, 1][0])
    decision = int(proba >= 0.206)

    return jsonify(sk_id=sk_id, proba=proba, decision=decision)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
