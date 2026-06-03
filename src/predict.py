"""
Inférence — utilise le modèle entraîné pour prédire le churn
=============================================================
Charge le modèle sérialisé (pipeline complet : feature eng + preprocessing + classifier)
et le seuil optimisé, puis prédit le churn sur un échantillon.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

# Permet à pickle de retrouver la classe TelecomFeatureEngineer
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import TelecomFeatureEngineer  # noqa: F401, E402

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"


def load_model() -> dict:
    """Charge le pipeline entraîné + son seuil optimisé."""
    model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("Aucun modèle trouvé. Lance d'abord src/pipeline.py.")
    with open(model_files[0], "rb") as f:
        return pickle.load(f)


def predict_churn(customer_df: pd.DataFrame, model_dict: dict) -> pd.DataFrame:
    """
    Prédit le churn pour un DataFrame de clients.
    Retourne probabilité, prédiction binaire, et segment de risque.
    """
    pipeline = model_dict["pipeline"]
    threshold = model_dict["threshold"]

    proba = pipeline.predict_proba(customer_df)[:, 1]
    pred = (proba >= threshold).astype(int)

    risk_segment = pd.cut(
        proba,
        bins=[-0.001, 0.3, 0.6, 0.85, 1.001],
        labels=["Low", "Medium", "High", "Critical"],
    )

    return pd.DataFrame(
        {
            "churn_probability": proba.round(4),
            "predicted_churn": pred,
            "risk_segment": risk_segment,
        }
    )


def main():
    print("=" * 70)
    print("INFÉRENCE — IBM TELCO CUSTOMER CHURN")
    print("=" * 70)

    print("\n[1/3] Chargement du modèle...")
    model_dict = load_model()
    print(f"  ✓ Modèle : {model_dict['model_name']}")
    print(f"  ✓ Seuil de décision optimisé : {model_dict['threshold']:.3f}")

    print("\n[2/3] Chargement d'un échantillon de clients...")
    df = pd.read_csv(DATA_PATH).sample(n=20, random_state=123)
    feature_cols = [c for c in df.columns if c not in ("customerID", "Churn")]
    X_sample = df[feature_cols].reset_index(drop=True)
    print(f"  ✓ {len(X_sample)} clients à scorer")

    print("\n[3/3] Prédictions...")
    results = predict_churn(X_sample, model_dict)
    results["customerID"] = df["customerID"].values
    results["actual_churn"] = df["Churn"].values
    results = results[
        ["customerID", "churn_probability", "predicted_churn",
         "actual_churn", "risk_segment"]
    ]

    print("\n  RÉSULTATS :")
    print(results.to_string(index=False))

    print("\n  RÉPARTITION PAR SEGMENT DE RISQUE :")
    print(results["risk_segment"].value_counts().sort_index().to_string())

    accuracy = (results["predicted_churn"] == results["actual_churn"]).mean()
    print(f"\n  Accuracy sur cet échantillon : {accuracy:.2%}")


if __name__ == "__main__":
    main()
