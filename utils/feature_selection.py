#Imports
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_classif



##### SCORING #####



def feature_importance_scoring(data, model='RF'):
    """
    Calculates the feature importances for a given dataset using either a Random Forest or XGBoost model.

    This function trains the selected model (either Random Forest or XGBoost) on the provided dataset, splits the data into 
    training and validation sets, and computes the importance of each feature based on the trained model.

    Parameters:
        data (pd.DataFrame): 
            A DataFrame containing the feature variables and the target variable.
            The target variable must be named 'SalePrice'.
        model (str, optional): 
            The model to use for training. Can be:
            - 'RF': Uses the Random Forest Regressor (default).
            - 'XGB': Uses the XGBoost Regressor.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the importance of each feature, sorted in descending order.
    
    Notes:
        - The dataset is split into training and validation sets with an 80-20 ratio.
        - The function uses a random state of 1 for reproducibility.

    Example:
        >>> feature_importance_scoring(data, model='RF')
        Feature                Importance
        ------------------     -------------
        OverallQual            0.585006
        GrLivArea             0.158602
        TotalBsmtSF           0.085868
        YearBuilt             0.031505
        GarageCars            0.025386
        ...
    """
    # Séparer les features et la cible
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    
    # Diviser les données en training et validation (80-20)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

    # Choisir et entraîner le modèle
    if model == 'RF':
        model = RandomForestRegressor(random_state=1)
        model.fit(X_train, y_train)
    elif model == 'XGB':
        model = XGBRegressor(random_state=1)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Model must be either 'RF' for Random Forest or 'XGB' for XGBoost.")
    
    # Obtenir l'importance des features
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["Importance"]).sort_values("Importance", ascending=False)

    return feature_importances




def coeff_elastic_net(df):
    """
    Effectue une régression ElasticNet avec recherche d'hyperparamètres via GridSearchCV et un pipeline.
    Retourne les coefficients associés à chaque caractéristique, le meilleur modèle, et les scores de performance.

    Paramètres :
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données d'entrée. Doit inclure une colonne 'SalePrice' comme variable cible.

    Retour :
    -------
    pandas.DataFrame
        Un DataFrame contenant les coefficients de la régression ElasticNet, triés par importance absolue.
    dict
        Le meilleur ensemble d'hyperparamètres trouvé par GridSearchCV.
    float
        Le score R² sur l'ensemble de test.
    float
        Le score MAE (Mean Absolute Error) sur l'ensemble de test.
    """
    # Séparation des variables explicatives (X) et de la cible (y)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Définition du pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('elastic_net', ElasticNet(random_state=42))
    ])

    # Définition de la grille de paramètres
    param_grid = {
        'elastic_net__alpha': [0.1, 1.0, 10.0],
        'elastic_net__l1_ratio': [0.1, 0.5, 0.9]
    }

    # Recherche avec validation croisée
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        verbose=1,
        n_jobs=-1
    )

    # Entraînement du modèle avec GridSearchCV
    grid_search.fit(X_train, y_train)

    # Meilleur modèle
    best_model = grid_search.best_estimator_

    # Performance sur l'ensemble de test
    test_score_r2 = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)
    test_score_mae = mean_absolute_error(y_test, y_pred)

    print(f"Meilleur score R² sur l'ensemble de test : {test_score_r2:.4f}")
    print(f"Meilleur score MAE sur l'ensemble de test : {test_score_mae:.4f}")
    print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")

    # Extraction des coefficients
    elastic_net = best_model.named_steps['elastic_net']
    feature_coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': elastic_net.coef_
    })

    # Trier les coefficients par importance absolue décroissante
    feature_coefficients = feature_coefficients.reindex(
        feature_coefficients['Coefficient'].abs().sort_values(ascending=False).index
    )

    return feature_coefficients, grid_search.best_params_, test_score_r2, test_score_mae



def make_mi_scores(X, y, discrete_features):
    """
    Calcule les scores d'information mutuelle (MI) entre les caractéristiques et la variable cible pour évaluer leur dépendance.

    Paramètres :
    ----------
    X : pandas.DataFrame
        DataFrame contenant les caractéristiques explicatives.
    y : pandas.Series ou numpy.array
        Variable cible continue.
    discrete_features : array-like, shape (n_features,)
        Indique si chaque caractéristique est discrète (True) ou continue (False).

    Retour :
    -------
    pandas.Series
        Une série Pandas contenant les scores d'information mutuelle pour chaque caractéristique, triés par ordre décroissant.
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores




def select_k_best_anova(df, target_column='SalePrice', k=10):
    """
    Effectue une sélection des k meilleures variables catégorielles (int) en fonction de leur relation
    avec la variable cible continue (float) via ANOVA.

    Paramètres :
    - df : pandas DataFrame contenant les variables catégorielles (int) et continues (float)
    - target_column : string, nom de la variable cible
    - k : int ou 'all', nombre de variables à retourner (par défaut 10, 'all' pour toutes les variables)

    Retourne :
    - Un DataFrame contenant les variables sélectionnées, leurs scores ANOVA et p-values.
    """
    # Vérifier si la colonne cible est dans le DataFrame
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame.")
    
    # Sélectionner les variables catégorielles (int)
    categorical_features = df.select_dtypes(include='int').columns.drop(target_column)
    
    if len(categorical_features) == 0:
        raise ValueError("Aucune variable catégorielle (int) trouvée dans le DataFrame.")
    
    # Cible continue (float)
    target = df[target_column]
    
    # Application de SelectKBest avec f_classif (test ANOVA)
    selector = SelectKBest(score_func=f_classif, k='all' if k == 'all' else k)
    selector.fit(df[categorical_features], target)
    
    # Récupération des résultats
    scores = selector.scores_
    p_values = selector.pvalues_
    selected_features = categorical_features[selector.get_support()]
    
    # Création du DataFrame des résultats
    results = pd.DataFrame({
        'Feature': selected_features,
        'Score': scores[selector.get_support()],
        'p-Value': p_values[selector.get_support()]
    })
    
    # Tri par score décroissant
    results = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    return results




def select_kendall_correlation(df, target_column='SalePrice'):
    """
    Calcule la corrélation de Kendall entre les variables continues (float) et une variable cible continue.
    
    Paramètres :
    - df : pandas DataFrame contenant les données.
    - target_column : string, nom de la variable cible (par défaut 'SalePrice').
    
    Retourne :
    - Un DataFrame contenant les variables continues et leurs scores Kendall.
    """
    # Vérifier si la colonne cible est présente dans le DataFrame
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame.")
    
    # Sélectionner uniquement les variables continues (float)
    continuous_features = df.select_dtypes(include='float').columns
    if target_column in continuous_features:
        continuous_features = continuous_features.drop(target_column)
    
    if len(continuous_features) == 0:
        raise ValueError("Aucune variable continue (float) trouvée dans le DataFrame.")
    
    # Calculer les corrélations Kendall entre chaque variable continue et la cible
    correlations = df[continuous_features].corrwith(df[target_column], method='kendall')
    
    # Créer un DataFrame pour les résultats
    results_df = correlations.reset_index()
    results_df.columns = ['Feature', 'Score']
    
    # Trier les résultats par score décroissant
    results_df = results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    return results_df
