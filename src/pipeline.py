"""
Pipeline ML production-grade — IBM Telco Customer Churn
=======================================================
Pipeline complet pour la prédiction du churn sur le dataset public IBM :
1. Feature engineering (ratios, agrégations, flags business)
2. Preprocessing avec ColumnTransformer (imputation + encoding + scaling)
3. Entraînement de 4 modèles : Logistic Regression, Random Forest,
   Gradient Boosting, XGBoost
4. Hyperparameter tuning avec RandomizedSearchCV (CV stratifié)
5. Threshold optimization sur validation set
6. Évaluation complète + visualisations + sauvegarde du meilleur modèle
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
for d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================
class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering métier sur le dataset IBM Telco.
    Crée des features dérivées :
    - num_addon_services : nb total de services additionnels
    - charge_per_tenure  : ARPU normalisé par ancienneté
    - is_new_customer    : flag <= 6 mois (période critique)
    - has_fiber          : flag fibre (segment à risque connu)
    - charge_per_service : valeur perçue
    - tenure_group       : segments d'ancienneté
    - has_protection     : a au moins une protection (sec/backup/device/tech)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Nombre de services additionnels souscrits
        addon_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        X["num_addon_services"] = X[addon_cols].sum(axis=1)

        # Charge par mois d'ancienneté (proxy de la stabilité de facturation)
        X["charge_per_tenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)

        # Flag client nouveau (les 6 premiers mois sont critiques)
        X["is_new_customer"] = (X["tenure"] <= 6).astype(int)

        # Flag fibre (segment à churn élevé bien documenté)
        X["has_fiber"] = (X["InternetService"] == "Fiber optic").astype(int)

        # Charge par service souscrit
        X["total_services"] = (
            X["num_addon_services"]
            + X["PhoneService"]
            + (X["InternetService"] != "No").astype(int)
        )
        X["charge_per_service"] = X["MonthlyCharges"] / (X["total_services"] + 1)

        # Groupes d'ancienneté
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[-1, 6, 12, 24, 48, 100],
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48m+"],
        ).astype(str)

        # A au moins une protection
        protection_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"
        ]
        X["has_protection"] = (X[protection_cols].sum(axis=1) > 0).astype(int)

        return X


# ============================================================
# 2. PREPROCESSING
# ============================================================
def build_preprocessor() -> ColumnTransformer:
    """Construit le préprocesseur (imputation + encoding + scaling)."""

    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        # Features engineered
        "num_addon_services", "charge_per_tenure", "total_services",
        "charge_per_service",
    ]

    categorical_features = [
        "gender", "Contract", "PaymentMethod", "InternetService", "tenure_group",
    ]

    binary_features = [
        "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling",
        "is_new_customer", "has_fiber", "has_protection",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ============================================================
# 3. MODÈLES & GRILLES D'HYPERPARAMÈTRES
# ============================================================
def get_models_and_grids():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [8, 12],
                "classifier__min_samples_leaf": [2, 5],
                "classifier__max_features": ["sqrt"],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [150, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
                tree_method="hist",
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [4, 6],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
        ),
    }


# ============================================================
# 4. ENTRAÎNEMENT + TUNING
# ============================================================
def train_and_tune(X_train, y_train, n_iter: int = 5):
    preprocessor = build_preprocessor()
    feature_engineer = TelecomFeatureEngineer()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, (estimator, param_grid) in get_models_and_grids().items():
        print(f"\n{'='*70}\n  Entraînement & tuning : {name}\n{'='*70}")

        pipeline = Pipeline(
            steps=[
                ("feature_eng", feature_engineer),
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            refit=True,
        )

        search.fit(X_train, y_train)

        print(f"  ✓ Meilleur ROC-AUC (CV) : {search.best_score_:.4f}")
        print(f"  ✓ Meilleurs hyperparamètres :")
        for k, v in search.best_params_.items():
            print(f"      {k.replace('classifier__', '')} = {v}")

        results[name] = {
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
        }

    return results


# ============================================================
# 5. ÉVALUATION & THRESHOLD TUNING
# ============================================================
def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def optimize_threshold(model, X_val, y_val, metric: str = "accuracy"):
    """Optimise le seuil de décision sur le validation set."""
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "accuracy":
            scores.append(accuracy_score(y_val, y_pred))
        elif metric == "f1":
            scores.append(f1_score(y_val, y_pred))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


# ============================================================
# 6. VISUALISATIONS
# ============================================================
def plot_evaluation_figures(models: dict, X_test, y_test):
    # ROC curves
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, info in models.items():
        y_proba = info["best_estimator"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC — comparaison des modèles", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_roc_curves.png")
    plt.close(fig)
    print("✓ Sauvegardé : 05_roc_curves.png")

    # Matrices de confusion
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, models.items()):
        y_proba = info["best_estimator"].predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= info["best_threshold"]).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Non-churn", "Churn"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
    fig.suptitle("Matrices de confusion (seuil optimisé)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_confusion_matrices.png")
    plt.close(fig)
    print("✓ Sauvegardé : 06_confusion_matrices.png")


def plot_feature_importance(best_model_info: dict, feature_names: list, model_name: str):
    pipeline = best_model_info["best_estimator"]
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        importance_label = "Importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        importance_label = "|Coefficient|"
    else:
        print(f"  (Pas de feature_importances_ ni coef_ pour {model_name})")
        return

    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices][::-1], color="#2980b9")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel(importance_label)
    ax.set_title(f"Top 20 features — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_feature_importance.png")
    plt.close(fig)
    print("✓ Sauvegardé : 07_feature_importance.png")


def plot_model_comparison(all_metrics: list):
    df_metrics = pd.DataFrame(all_metrics).set_index("model")
    df_plot = df_metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="black")
    ax.set_title(
        "Comparaison des modèles — métriques sur le test set",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_model_comparison.png")
    plt.close(fig)
    print("✓ Sauvegardé : 08_model_comparison.png")


def get_feature_names(pipeline) -> list:
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


# ============================================================
# 7. MAIN
# ============================================================
def run_pipeline():
    print("=" * 70)
    print("PIPELINE ML — IBM TELCO CUSTOMER CHURN")
    print("=" * 70)

    print("\n[1/6] Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    print(f"  ✓ {len(df):,} clients chargés (taux de churn : {df['Churn'].mean():.2%})")

    print("\n[2/6] Split train/val/test (70/15/15)...")
    feature_cols = [c for c in df.columns if c not in ("customerID", "Churn")]
    X = df[feature_cols]
    y = df["Churn"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.176, random_state=42, stratify=y_train_full,
    )
    print(f"  Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")

    print("\n[3/6] Entraînement & hyperparameter tuning...")
    models = train_and_tune(X_train, y_train, n_iter=5)

    print("\n[4/6] Optimisation du seuil de décision (validation set)...")
    for name, info in models.items():
        best_threshold, best_val_acc = optimize_threshold(
            info["best_estimator"], X_val, y_val, metric="accuracy"
        )
        info["best_threshold"] = best_threshold
        info["val_accuracy"] = best_val_acc
        print(
            f"  {name:20s} seuil optimal = {best_threshold:.3f}  "
            f"(val accuracy = {best_val_acc:.4f})"
        )

    print("\n[5/6] Évaluation sur le test set (avec seuil optimisé)...")
    all_metrics = []
    for name, info in models.items():
        metrics = evaluate_model(
            info["best_estimator"], X_test, y_test, name,
            threshold=info["best_threshold"],
        )
        info["test_metrics"] = metrics
        all_metrics.append(metrics)
        print(f"\n  {name} (seuil={metrics['threshold']:.3f}):")
        print(f"    Accuracy  : {metrics['accuracy']:.4f}")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1        : {metrics['f1']:.4f}")
        print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")

    best_model_name = max(all_metrics, key=lambda m: m["roc_auc"])["model"]
    best_model_info = models[best_model_name]
    print(f"\n  🏆 Meilleur modèle : {best_model_name}")
    print(f"      Accuracy : {best_model_info['test_metrics']['accuracy']:.4f}")
    print(f"      ROC-AUC  : {best_model_info['test_metrics']['roc_auc']:.4f}")

    print(f"\n  Classification report ({best_model_name}):")
    y_proba_best = best_model_info["best_estimator"].predict_proba(X_test)[:, 1]
    y_pred_best = (y_proba_best >= best_model_info["best_threshold"]).astype(int)
    print(classification_report(y_test, y_pred_best, target_names=["Non-churn", "Churn"]))

    print("\n[6/6] Génération des visualisations...")
    plot_evaluation_figures(models, X_test, y_test)
    plot_model_comparison(all_metrics)
    feature_names = get_feature_names(best_model_info["best_estimator"])
    plot_feature_importance(best_model_info, feature_names, best_model_name)

    model_path = MODELS_DIR / f"best_model_{best_model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": best_model_info["best_estimator"],
                "threshold": best_model_info["best_threshold"],
                "model_name": best_model_name,
            },
            f,
        )
    print(f"\n✓ Meilleur modèle sauvegardé : {model_path}")

    metrics_path = METRICS_DIR / "model_metrics.json"
    summary = {
        "dataset": "IBM Telco Customer Churn (7,043 clients)",
        "best_model": best_model_name,
        "best_threshold": best_model_info["best_threshold"],
        "all_models": {
            name: {
                "best_params": {
                    k.replace("classifier__", ""): v
                    for k, v in info["best_params"].items()
                },
                "cv_roc_auc": info["best_cv_score"],
                "optimal_threshold": info["best_threshold"],
                "val_accuracy": info["val_accuracy"],
                "test_metrics": info["test_metrics"],
            }
            for name, info in models.items()
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Métriques sauvegardées : {metrics_path}")

    return models, best_model_name, summary


if __name__ == "__main__":
    run_pipeline()
