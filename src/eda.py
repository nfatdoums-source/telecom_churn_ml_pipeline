"""
Exploratory Data Analysis (EDA) — IBM Telco Customer Churn
===========================================================
Analyse exploratoire du dataset réel :
- Statistiques descriptives
- Distribution du churn par segment
- Corrélations
- Visualisations sauvegardées dans outputs/figures/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "telco_churn.csv")


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("RÉSUMÉ — IBM TELCO CUSTOMER CHURN")
    print("=" * 70)
    print(f"Nombre de clients : {len(df):,}")
    print(f"Nombre de features : {df.shape[1] - 2}")
    print(f"\nTaux de churn global : {df['Churn'].mean():.2%}")
    print(f"Clients ayant churné : {df['Churn'].sum():,}")
    print(f"\nValeurs manquantes :\n{df.isna().sum()[df.isna().sum() > 0]}")


def plot_churn_overview(df: pd.DataFrame) -> None:
    """Vue d'ensemble du churn par segments principaux."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Vue d'ensemble du Churn par segments clés", fontsize=15, fontweight="bold"
    )

    # 1. Churn par type de contrat
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values()
    axes[0, 0].bar(contract_churn.index, contract_churn.values, color="#3498db")
    axes[0, 0].set_title("Taux de churn par type de contrat")
    axes[0, 0].set_ylabel("Taux de churn")
    axes[0, 0].set_ylim(0, max(contract_churn.values) * 1.2)
    for i, v in enumerate(contract_churn.values):
        axes[0, 0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

    # 2. Distribution de l'ancienneté par statut churn
    for status, label, color in [(0, "Non-churn", "#2ecc71"), (1, "Churn", "#e74c3c")]:
        axes[0, 1].hist(
            df[df["Churn"] == status]["tenure"],
            bins=30,
            alpha=0.6,
            label=label,
            color=color,
        )
    axes[0, 1].set_title("Distribution de l'ancienneté par statut de churn")
    axes[0, 1].set_xlabel("Ancienneté (mois)")
    axes[0, 1].set_ylabel("Nombre de clients")
    axes[0, 1].legend()

    # 3. Churn par méthode de paiement
    pm_churn = df.groupby("PaymentMethod")["Churn"].mean().sort_values()
    axes[1, 0].barh(pm_churn.index, pm_churn.values, color="#e74c3c")
    axes[1, 0].set_title("Taux de churn par méthode de paiement")
    axes[1, 0].set_xlabel("Taux de churn")
    for i, v in enumerate(pm_churn.values):
        axes[1, 0].text(v + 0.005, i, f"{v:.1%}", va="center", fontweight="bold")

    # 4. Charges mensuelles par statut
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Charges mensuelles par statut de churn")
    axes[1, 1].set_xlabel("Churn (0 = non, 1 = oui)")
    axes[1, 1].set_ylabel("Charges mensuelles ($)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_churn_overview.png")
    plt.close(fig)
    print("✓ Sauvegardé : 01_churn_overview.png")


def plot_service_analysis(df: pd.DataFrame) -> None:
    """Analyse de l'impact des services souscrits sur le churn."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Impact des services souscrits sur le churn",
        fontsize=15,
        fontweight="bold",
    )

    # 1. Churn par type d'internet
    internet_churn = df.groupby("InternetService")["Churn"].mean().sort_values()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    axes[0, 0].bar(internet_churn.index, internet_churn.values, color=colors)
    axes[0, 0].set_title("Taux de churn par type d'internet")
    axes[0, 0].set_ylabel("Taux de churn")
    for i, v in enumerate(internet_churn.values):
        axes[0, 0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

    # 2. Nombre de services additionnels vs Churn
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df_temp = df.copy()
    df_temp["num_addon_services"] = df_temp[service_cols].sum(axis=1)
    addon_churn = df_temp.groupby("num_addon_services")["Churn"].agg(["mean", "count"])
    axes[0, 1].bar(addon_churn.index, addon_churn["mean"], color="#9b59b6")
    axes[0, 1].set_title("Taux de churn par nb de services additionnels")
    axes[0, 1].set_xlabel("Nombre de services additionnels")
    axes[0, 1].set_ylabel("Taux de churn")
    for x, v in zip(addon_churn.index, addon_churn["mean"]):
        axes[0, 1].text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    # 3. Tech Support et Online Security (services protecteurs)
    pivot = pd.pivot_table(
        df,
        values="Churn",
        index="OnlineSecurity",
        columns="TechSupport",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot, annot=True, fmt=".1%", cmap="YlOrRd", ax=axes[1, 0],
        cbar_kws={"label": "Taux de churn"},
        xticklabels=["Pas de Tech Support", "Tech Support"],
        yticklabels=["Pas de OnlineSec", "Online Security"],
    )
    axes[1, 0].set_title("Taux de churn : Online Security × Tech Support")

    # 4. PaperlessBilling vs Churn
    pb_churn = df.groupby("PaperlessBilling")["Churn"].mean()
    axes[1, 1].bar(
        ["Facture papier", "Sans papier"], pb_churn.values, color=["#16a085", "#c0392b"]
    )
    axes[1, 1].set_title("Taux de churn : Paperless Billing")
    axes[1, 1].set_ylabel("Taux de churn")
    for i, v in enumerate(pb_churn.values):
        axes[1, 1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_service_analysis.png")
    plt.close(fig)
    print("✓ Sauvegardé : 02_service_analysis.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Matrice de corrélation des variables numériques."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Corrélation de Pearson"},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Matrice de corrélation des variables numériques",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_correlation_matrix.png")
    plt.close(fig)
    print("✓ Sauvegardé : 03_correlation_matrix.png")


def plot_demographics_analysis(df: pd.DataFrame) -> None:
    """Analyse du churn par variables démographiques."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse démographique du churn", fontsize=15, fontweight="bold")

    # 1. Gender
    g_churn = df.groupby("gender")["Churn"].mean()
    axes[0, 0].bar(g_churn.index, g_churn.values, color=["#3498db", "#e91e63"])
    axes[0, 0].set_title("Churn par genre")
    axes[0, 0].set_ylabel("Taux de churn")
    for i, v in enumerate(g_churn.values):
        axes[0, 0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold")

    # 2. Senior Citizen
    sc_churn = df.groupby("SeniorCitizen")["Churn"].mean()
    axes[0, 1].bar(
        ["Non senior", "Senior"], sc_churn.values, color=["#16a085", "#e67e22"]
    )
    axes[0, 1].set_title("Churn : Senior Citizen vs reste")
    axes[0, 1].set_ylabel("Taux de churn")
    for i, v in enumerate(sc_churn.values):
        axes[0, 1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

    # 3. Partner & Dependents
    df_temp = df.copy()
    df_temp["family_situation"] = (
        df_temp["Partner"].astype(str) + "_P_"
        + df_temp["Dependents"].astype(str) + "_D"
    )
    fs_churn = df_temp.groupby("family_situation")["Churn"].mean().sort_values()
    labels = {
        "0_P_0_D": "Seul",
        "1_P_0_D": "Avec partenaire",
        "0_P_1_D": "Avec dépendants",
        "1_P_1_D": "Famille complète",
    }
    fs_churn.index = [labels.get(i, i) for i in fs_churn.index]
    axes[1, 0].barh(fs_churn.index, fs_churn.values, color="#8e44ad")
    axes[1, 0].set_title("Churn par situation familiale")
    axes[1, 0].set_xlabel("Taux de churn")
    for i, v in enumerate(fs_churn.values):
        axes[1, 0].text(v + 0.005, i, f"{v:.1%}", va="center", fontweight="bold")

    # 4. Tenure × MonthlyCharges scatter
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    scatter = axes[1, 1].scatter(
        sample["tenure"],
        sample["MonthlyCharges"],
        c=sample["Churn"],
        cmap="coolwarm",
        alpha=0.5,
        s=20,
    )
    axes[1, 1].set_title("Ancienneté × Charges mensuelles (couleur = churn)")
    axes[1, 1].set_xlabel("Ancienneté (mois)")
    axes[1, 1].set_ylabel("Charges mensuelles ($)")
    plt.colorbar(scatter, ax=axes[1, 1], label="Churn")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_demographics.png")
    plt.close(fig)
    print("✓ Sauvegardé : 04_demographics.png")


def run_eda():
    df = load_data()
    print_summary(df)
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    plot_churn_overview(df)
    plot_service_analysis(df)
    plot_correlation_heatmap(df)
    plot_demographics_analysis(df)
    print(f"\n✓ Toutes les figures sont dans : {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
