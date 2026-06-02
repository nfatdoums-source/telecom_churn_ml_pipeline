# 📊 End-to-End ML Pipeline — IBM Telco Customer Churn Prediction

Pipeline de Machine Learning production-grade pour la prédiction du churn dans un contexte télécom, utilisant le **dataset public IBM Telco Customer Churn**.

## 📦 Dataset

**IBM Telco Customer Churn** — un benchmark public reconnu de l'industrie, publié par IBM.

- **Source** : [github.com/IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **7,043 clients** réels d'un opérateur télécom
- **21 colonnes** : démographie, services, contrat, facturation, churn
- **Taux de churn : 26.5%** (1,869 clients)
- Téléchargement automatique via `src/data_loading.py`

## 🎯 Résultats

**Meilleur modèle : XGBoost** (par ROC-AUC), avec **LogisticRegression** quasi équivalent et plus interprétable.

| Modèle | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **LogisticRegression** | **0.8136** | 0.8538 | 0.6781 | 0.5643 | 0.6160 |
| RandomForest | 0.8051 | 0.8558 | 0.6667 | 0.5286 | 0.5896 |
| GradientBoosting | 0.8004 | 0.8554 | 0.6751 | 0.4750 | 0.5577 |
| **XGBoost** | 0.8061 | **0.8573** | 0.6556 | 0.5643 | 0.6065 |

📌 **Ces résultats sont conformes à l'état de l'art** sur ce dataset (les solutions Kaggle de référence tournent entre 80-82% accuracy et 0.84-0.86 ROC-AUC).

## 🏗️ Architecture du pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  IBM Telco CSV  │───▶│  Feature         │───▶│  Preprocessing     │
│  (7,043 × 21)   │    │  Engineering     │    │  (Imputation +     │
│                 │    │  (7 nouv. feat.) │    │   OHE + Scaling)   │
└─────────────────┘    └──────────────────┘    └─────────┬──────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  Best Model     │◀───│  Threshold       │◀───│  RandomizedSearchCV│
│  + Threshold    │    │  Optimization    │    │  (4 modèles ML)    │
│  (.pkl)         │    │  (val set)       │    │  Stratified CV     │
└─────────────────┘    └──────────────────┘    └────────────────────┘
```

## 📂 Structure du projet

```
churn_ml_pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── telco_churn_raw.csv             # Brut depuis GitHub IBM
│   └── telco_churn.csv                 # Nettoyé (TotalCharges en float, etc.)
├── src/
│   ├── data_loading.py                 # Télécharge + nettoie le dataset IBM
│   ├── eda.py                          # 4 figures d'analyse exploratoire
│   ├── pipeline.py                     # Pipeline ML complet
│   └── predict.py                      # Script d'inférence + segmentation risque
└── outputs/
    ├── figures/                        # 8 visualisations PNG
    ├── models/best_model_*.pkl         # Pipeline sérialisé + seuil
    └── metrics/model_metrics.json      # Toutes les métriques
```

## 🔧 Composants techniques

### 1. Data loading (`data_loading.py`)
- Téléchargement automatique depuis le repo GitHub d'IBM
- Conversion de `TotalCharges` (string → float, gère les 11 valeurs manquantes)
- Conversion `Churn` Yes/No → 0/1
- Normalisation des valeurs "No internet service" / "No phone service" → "No"
- Encodage des variables binaires Yes/No → 0/1

### 2. EDA (`eda.py`)
4 sections d'analyse business :
- Vue d'ensemble du churn (contrat, ancienneté, paiement, charges)
- Impact des services souscrits (fibre vs DSL, bundle, protections)
- Matrice de corrélation
- Analyse démographique (genre, senior, situation familiale)

### 3. Pipeline ML (`pipeline.py`)

**Feature Engineering** (transformer sklearn compatible) :
- `num_addon_services` — nb de services additionnels souscrits
- `charge_per_tenure` — facture mensuelle / ancienneté
- `is_new_customer` — flag <= 6 mois (période critique)
- `has_fiber` — flag fibre (segment à churn élevé connu)
- `total_services` — nb total de services
- `charge_per_service` — valeur perçue
- `tenure_group` — segments d'ancienneté (0-6m, 6-12m, etc.)
- `has_protection` — au moins un service de protection

**Preprocessing** (`ColumnTransformer`) :
- Numériques → `SimpleImputer(median)` + `StandardScaler`
- Catégorielles → `SimpleImputer(most_frequent)` + `OneHotEncoder`
- Binaires → passthrough

**Modèles** :
- LogisticRegression
- RandomForest
- GradientBoosting
- XGBoost

**Tuning** :
- `RandomizedSearchCV` avec `StratifiedKFold(5)` sur train set
- Scoring : `roc_auc`

**Threshold optimization** :
- Sur le **validation set** uniquement (jamais sur le test set)
- Maximise l'accuracy

**Splits** : Train 70% / Val 15% / Test 15% (stratifié)

### 4. Inférence (`predict.py`)
Charge le pipeline sérialisé et applique :
1. Feature engineering automatique
2. Preprocessing
3. Prédiction de probabilité
4. Seuil optimisé
5. **Segmentation du risque** : Low / Medium / High / Critical

## 🚀 Utilisation

```bash
# Installer les dépendances
pip install -r requirements.txt

# 1. Télécharger + nettoyer le dataset IBM
python src/data_loading.py

# 2. Analyse exploratoire (8 figures dans outputs/figures/)
python src/eda.py

# 3. Entraîner les 4 modèles + tuning + sauvegarde du meilleur
python src/pipeline.py

# 4. Tester l'inférence sur un échantillon de 20 clients
python src/predict.py
```

## 📈 Insights business clés (issus du modèle et de l'EDA)

1. **Les contrats month-to-month ont un taux de churn 3-4× plus élevé que les contrats 2 ans** — c'est le driver #1. Pousser les renouvellements longue durée est le levier de rétention le plus efficace.
2. **La fibre optique a un churn nettement supérieur au DSL** — paradoxe classique : ces clients paient plus cher et ont des attentes plus élevées. Investir dans la qualité de service fibre est critique.
3. **Les 6 premiers mois sont critiques** — `is_new_customer` est parmi les top features. Programme d'onboarding renforcé recommandé.
4. **La méthode de paiement "Electronic check"** est corrélée à un churn élevé — signal de clients moins engagés. Pousser les paiements automatiques (bank transfer, credit card).
5. **Les services de protection (Online Security, Tech Support) réduisent fortement le churn** — leur upsell est aussi une stratégie de rétention.
6. **Les seniors churnent plus que la moyenne** — adapter le support et la communication à cette clientèle.

## 🛠️ Stack technique

- **Python 3.10+**
- **scikit-learn 1.4+** — Pipeline, ColumnTransformer, RandomizedSearchCV
- **pandas 2.0+** — Manipulation des données
- **NumPy** — Calcul vectoriel
- **XGBoost 2.0+** — Gradient boosting
- **Matplotlib + Seaborn** — Visualisations

## 📝 Notes méthodologiques

- **Pas de data leakage** : feature engineering ET imputation sont intégrés dans le pipeline sklearn, donc fittés uniquement sur le train fold à chaque itération de CV.
- **Threshold tuning isolé** : effectué sur la validation set, pas le test set → métriques de test honnêtes.
- **Classe minoritaire (26.5%)** : pas de SMOTE ici car la baseline est correcte ; pourrait être ajouté via `imblearn` si on voulait privilégier le recall.
- **Reproductibilité** : tous les `random_state=42`.
- **Benchmark de référence** : nos résultats (81% accuracy, 0.86 ROC-AUC) sont au niveau des meilleures solutions Kaggle publiées sur ce dataset.
