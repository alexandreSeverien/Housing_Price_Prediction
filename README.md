# Prédiction des Prix de Maisons - Participation au Concours Kaggle

Ce dépôt contient le code Python utilisé pour participer au concours **"House Prices for Kaggle Learn Users"** disponible sur Kaggle. Ce projet a été réalisé en deux phases distinctes : une participation initiale au début de mon apprentissage, suivie d'une reprise du projet avec plus d'expérience pour mieux comprendre et résoudre le problème.

## Objectifs du Projet

Le but principal est de prédire les prix des maisons en fonction de nombreuses variables descriptives. Plus spécifiquement, pour cette reprise du projet :
- Construire un modèle de prédiction **stable** et performant.
- Explorer différentes techniques de recherche d’hyperparamètres et de réduction d'overfitting.
- Comprendre les limites et les spécificités des données fournies, notamment la gestion des **outliers**.

## Méthodologie

### 1. Prétraitement des Données
- Nettoyage des données et gestion des valeurs manquantes.
- Suppression des maisons ayant un prix supérieur à **350 000 $** pour mieux concentrer l'analyse sur un sous-ensemble cohérent.
- Encodage des variables non numériques avec un **Ordinal Encoder**.

### 2. Recherche d'Hyperparamètres
- **Randomized Search CV** : Une recherche efficace pour explorer de nombreux hyperparamètres sans tester exhaustivement toutes les combinaisons.
- **Grid Search CV** : Affinement des meilleurs hyperparamètres issus de la recherche précédente pour obtenir un modèle optimisé.

### 3. Modèle Final
- Utilisation d’un modèle **Random Forest Regressor** ajusté avec les meilleurs hyperparamètres.
- Évaluation des performances avec des **courbes d’apprentissage** basées sur le R² pour détecter l’overfitting et analyser la généralisation.

## Résultats et Apprentissage

### Participation Initiale
Lors de ma première participation, mon modèle de **Random Forest** a atteint une **Mean Absolute Error (MAE)** de **17775.56448**, avec une position au leaderboard de **46164**. Cette première tentative a été l’occasion de me familiariser avec le concours et les concepts de base en machine learning.

### Reprise du Projet
En reprenant ce projet avec plus d'expérience, j’ai réalisé plusieurs choses :
- **Nature du concours** : Les données incluent des **outliers**, impossibles à exclure pour une soumission complète, ce qui rend crucial le choix de modèles robustes (comme les modèles d’ensembles).
- **Modèles stables** : J’ai appris à construire des modèles moins sensibles aux données et mieux adaptés aux variations des jeux d’entraînement et de test.
- **Recherche d’hyperparamètres** : Une recherche plus approfondie et itérative aurait permis de réduire davantage l’overfitting.

### Limites et pistes d’amélioration
- La suppression des maisons au-delà de 350 000 $ a permis de simplifier le problème mais limite la portée du modèle.
- Une recherche d’hyperparamètres plus longue et itérative (plusieurs passes de Randomized Search CV) aurait permis d’améliorer davantage la généralisation.
- Le projet pourrait être étendu à d'autres modèles (e.g., Gradient Boosting) tout en continuant à travailler sur l’optimisation.

## Structure du Projet

- **`exploratory_analysis/`** : Notebooks pour l'exploration des données et la sélection de features.
  - `exploration_dataset.ipynb`
  - `feature_selection.ipynb`

- **`model_optimization/`** : Notebook pour l’optimisation et l’évaluation du modèle Random Forest.
  - `random_forest_model.ipynb`

- **`participation_initiale/`** : Contient les fichiers et soumissions liés à ma première participation.
  - `participation_initiale.py`
  - `submissions/`
  - `train.csv`

- **`utils/`** : Scripts utilitaires pour le prétraitement des données.
  - `data_utils.py`
  - `feature_selection.py`
  - `__init__.py`

- **`home-data-for-ml-course/`** : Données et ressources fournies pour le concours.
  - `train.csv`, `test.csv`
  - `data_description.txt`
  - Fichiers de soumission exemple.


## Pour aller plus loin
Vous pouvez consulter le concours ici : [Kaggle Competition - House Prices](https://www.kaggle.com/competitions/home-data-for-ml-course).

Les résultats finaux et les insights principaux sont résumés dans ce dépôt. Ce projet illustre l'importance d'une **approche itérative** en machine learning pour construire des modèles robustes et adaptés aux données.