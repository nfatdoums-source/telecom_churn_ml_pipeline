"""
Chargement du dataset IBM Telco Customer Churn (réel, public)
==============================================================
Source : IBM, distribué publiquement sur GitHub.
        https://github.com/IBM/telco-customer-churn-on-icp4d

7,043 clients, 21 variables, taux de churn 26.5%.

Description des variables :
- Démographie : gender, SeniorCitizen, Partner, Dependents
- Compte : tenure (mois), Contract, PaperlessBilling, PaymentMethod
- Services : PhoneService, MultipleLines, InternetService (DSL/Fiber/None),
            OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies
- Facturation : MonthlyCharges, TotalCharges
- Cible : Churn (Yes/No)

Ce script :
1. Télécharge le dataset depuis GitHub (si pas déjà présent)
2. Nettoie : convertit TotalCharges en numérique, gère les valeurs manquantes
3. Sauvegarde un CSV propre prêt à l'emploi
"""

from pathlib import Path
import urllib.request

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_PATH = DATA_DIR / "telco_churn_raw.csv"
CLEAN_PATH = DATA_DIR / "telco_churn.csv"


def download_dataset() -> None:
    """Télécharge le dataset IBM Telco Customer Churn si absent."""
    if RAW_PATH.exists():
        print(f"  Dataset déjà présent : {RAW_PATH}")
        return
    print(f"  Téléchargement depuis : {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    print(f"  ✓ Sauvegardé : {RAW_PATH}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset brut."""
    df = df.copy()

    # 1. TotalCharges est en string et contient des espaces (clients à tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2. Convertir Churn (Yes/No) en binaire
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # 3. Convertir les "No internet service" / "No phone service" en "No"
    # (sinon on a 3 catégories au lieu de 2 pour des variables conceptuellement binaires)
    cols_with_no_internet = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in cols_with_no_internet:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # 4. Convertir les Yes/No en 0/1 pour les vraies variables binaires
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    ]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    return df


def main():
    print("=" * 70)
    print("CHARGEMENT DU DATASET IBM TELCO CUSTOMER CHURN (public)")
    print("=" * 70)

    print("\n[1/3] Téléchargement...")
    download_dataset()

    print("\n[2/3] Chargement et nettoyage...")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"  Shape brute : {df_raw.shape}")

    df_clean = clean_dataset(df_raw)

    print("\n[3/3] Statistiques après nettoyage :")
    print(f"  Clients : {len(df_clean):,}")
    print(f"  Variables : {df_clean.shape[1] - 2} (hors customerID et Churn)")
    print(f"  Taux de churn : {df_clean['Churn'].mean():.2%}")
    print(f"  Valeurs manquantes :")
    missing = df_clean.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, n in missing.items():
            print(f"    {col}: {n}")
    else:
        print("    (aucune)")

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\n✓ Dataset propre sauvegardé : {CLEAN_PATH}")
    return df_clean


if __name__ == "__main__":
    main()
