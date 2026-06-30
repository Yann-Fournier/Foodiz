# Guide d'execution — Foodiz

Ce document decrit comment installer et executer le projet de bout en bout : telechargement des donnees, pipeline ETL, entrainement des modeles, et lancement de l'interface de demonstration.

## 1. Prerequis

- **Python 3.11** (ou version compatible avec les paquets ci-dessous)
- **MongoDB** installe et lance en local sur le port par defaut (`mongodb://localhost:27017`)
- Environ **10 Go d'espace disque** libre (dataset CSV + base MongoDB)

## 2. Installation des dependances

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

Cela installe : `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `plotly`, `pymongo`, `scikit-learn`, `joblib`, `streamlit`.

## 3. Telechargement des donnees

Telecharger le fichier [en.openfoodfacts.org.products.csv](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz) depuis [Open Food Facts](https://world.openfoodfacts.org/data), le decompresser, et placer le CSV resultant dans :

```
Data/openfoodfacts-products.csv
```

## 4. Demarrer MongoDB

Verifier que le service MongoDB tourne en local avant de lancer le pipeline ou l'application. La base utilisee s'appelle `foodiz` et la collection `products` — elles sont creees automatiquement par le notebook 2.

## 5. Executer les notebooks (dans l'ordre)

Les notebooks se trouvent dans `Notebooks/` et doivent etre executes **dans l'ordre**, chacun dependant des resultats du precedent (CSV -> MongoDB -> modeles).

| # | Notebook | Role | Sortie |
|---|----------|------|--------|
| 1 | `1-Analyse_Exploratoire_des_Donnees.ipynb` | Exploration du dataset : structure, valeurs manquantes, distributions, correlations, anomalies | Aucune (analyse uniquement) |
| 2 | `2-Pipeline_ETL.ipynb` | Extraction du CSV, nettoyage, feature engineering (flags de regime), chargement dans MongoDB | Collection `foodiz.products` peuplee |
| 3 | `3-ML_Supervise_Classification_Regime.ipynb` | Entrainement d'un classifieur multi-label (Random Forest) predisant la compatibilite regime a partir du profil nutritionnel | `Models/regime_classifier_*.joblib` |
| 4 | `4-ML_Non_Supervise_Clustering.ipynb` | Clustering K-Means des produits par profil nutritionnel, ecriture du cluster dans MongoDB, systeme de recommandation d'alternatives | `Models/cluster_*.joblib` + champ `cluster` ajoute dans MongoDB |

**Temps d'execution indicatifs** : le notebook 1 prend quelques minutes (lecture rapide du CSV via PyArrow + colonnes utiles uniquement). Les notebooks 2 a 4 prennent de quelques minutes a une dizaine de minutes selon la machine, principalement a cause du chargement/ecriture MongoDB.

## 6. Lancer l'interface de demonstration

Une fois les notebooks 2, 3 et 4 executes (MongoDB peuple + modeles sauvegardes dans `Models/`), lancer l'application depuis la racine du projet :

```bash
streamlit run main.py
```

L'application s'ouvre dans le navigateur (par defaut `http://localhost:8501`) avec 4 pages accessibles depuis la barre laterale :

- **Vue d'ensemble** — statistiques globales et visualisations du dataset
- **Explorer les produits** — recherche de produits et consultation des fiches detaillees
- **Predire un regime** — prediction de compatibilite avec 9 regimes alimentaires (vegan, keto, sans gluten...) a partir d'un produit existant ou d'une saisie manuelle
- **Trouver des alternatives** — recommandation de produits similaires avec un meilleur Nutri-Score

## 7. Structure du projet

```
Foodiz/
├── Data/                          # Dataset Open Food Facts (non versionne, a telecharger)
├── Notebooks/
│   ├── 1-Analyse_Exploratoire_des_Donnees.ipynb
│   ├── 2-Pipeline_ETL.ipynb
│   ├── 3-ML_Supervise_Classification_Regime.ipynb
│   └── 4-ML_Non_Supervise_Clustering.ipynb
├── Models/                        # Modeles entraines (generes par les notebooks 3 et 4)
├── main.py                        # Application Streamlit
├── requirements.txt
├── README.md                      # Rapport technique (problematique, choix d'architecture)
└── EXECUTION.md                   # Ce document
```

## 8. Depannage

- **`ServerSelectionTimeoutError` au lancement de l'app** : MongoDB n'est pas demarre ou n'ecoute pas sur le port 27017.
- **Page "Predire un regime" / "Trouver des alternatives" indisponible** : les notebooks 3 et/ou 4 n'ont pas encore ete executes (fichiers manquants dans `Models/`).
- **`ModuleNotFoundError: No module named 'numpy._core'` (ou erreur similaire au chargement des modeles)** : incoherence entre l'environnement Python qui a entraine les modeles (celui utilise pour les notebooks) et celui qui execute `streamlit run main.py`. Les deux doivent utiliser le meme environnement, celui defini par `requirements.txt`.
