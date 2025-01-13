import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix



 ###Dictionnaire de remplacements        
replacements = {
    'Bldg' : {
                "TwnhsE" : "other",
                "Duplex" : "other",
                "Twnhs" : "other" ,
                "2fmCon" : "other"
                },
    'LotShape' : {
                    "IR1" : "IR",
                    "IR2" : "IR",
                    "IR3" : "IR"
                },
    'Condition1' : {
                    "PosA" : "Positive",
                    "PosN" : "Positive",
                    "Feedr" : "Artery", 
                    "RRAn" : "Railroad",
                    "RRAe" : "Railroad",
                    "RRNn" : "Railroad",
                    "RRNe" : "Railroad",
                },
    'HouseStyle' : {
                    "2Story" : "more",
                    "1.5Fin" : "more",
                    "1.5Unf" : "more", 
                    "2.5Unf" : "more",
                    "SLvl" : "more",
                    "SFoyer" : "more",
                    "2.5Fin" : "more"
                },
    'Exterior1st' : {
                    'AsbShng': 'LowCost',
                    'AsphShn': 'LowCost',
                    'CBlock': 'LowCost',
                    'Other': 'LowCost',
                    'Plywood': 'LowCost',
                    'HdBoard': 'MidRange',
                    'MetalSd': 'MidRange',
                    'Wd Sdng': 'MidRange',
                    'WdShing': 'MidRange',
                    'BrkFace': 'HighEnd',
                    'BrkComm': 'HighEnd',
                    'Stone': 'HighEnd',
                    'CemntBd': 'HighEnd',
                    'Stucco': 'HighEnd',
                    'ImStucc': 'HighEnd',
                    'PreCast': 'HighEnd',
                    'VinylSd': 'Standard'
                },
    'Exterior2nd' : {
                    'AsbShng': 'LowCost',
                    'AsphShn': 'LowCost',
                    'CBlock': 'LowCost',
                    'Other': 'LowCost',
                    'Plywood': 'LowCost',
                    'HdBoard': 'MidRange',
                    'MetalSd': 'MidRange',
                    'Wd Shng': 'MidRange',
                    'WdShing': 'MidRange',
                    'BrkFace': 'HighEnd',
                    'Brk Cmn': 'HighEnd',
                    'Stone': 'HighEnd',
                    'CmentBd': 'HighEnd',
                    'Stucco': 'HighEnd',
                    'ImStucc': 'HighEnd',
                    'PreCast': 'HighEnd',
                    'VinylSd': 'Standard'
                },
    'ExterQual' : {
                    "Fa" : 1,
                    "Po" : 0,
                    "TA" : 1,
                    "Gd" : 3,
                    "Ex" : 4
                },
    'Foundation' : {
                    'Wood': 'Other',
                    'Stone': 'Other',
                    'Slab': 'Other',
                },
    'BsmtQual' : {
                    'Ex': 100,       
                    'Gd': 94.5,      
                    'TA': 84.5,      
                    'Fa': 74.5,      
                    'Po': 69,        
                    'no_bsmnt': 0    
                },
    'BsmtFinType1' : {
                    'GLQ': 5,
                    'ALQ': 4,
                    'BLQ': 3,
                    'Rec': 2,
                    'LwQ': 1,
                    'Unf': 0,
                    'no_bsmnt': 0  
                },
    'HeatingQC' : {
                    'Ex': 5,
                    'Gd': 4,
                    'TA': 3,
                    'Fa': 1,  
                    'Po': 1,  
                },
    'KitchenQual' : {
                    'Ex': 5,
                    'Gd': 4,
                    'TA': 2,
                    'Fa': 2  
                },
    'GarageType' : {
                    '2Types': 'Other',   
                    'Basment': 'Other',  
                    'CarPort': 'Other',  
                },
    'Neighborhood': {
    # High-tier neighborhoods (mean SalePrice above ~250,000)
    'NoRidge': 'High',
    'NridgHt': 'High',
    'StoneBr': 'High',

    # Mid-high-tier neighborhoods (mean SalePrice ~200,000 - 250,000)
    'Timber': 'MidHigh',
    'ClearCr': 'MidHigh',
    'Somerst': 'MidHigh',
    'Veenker': 'MidHigh',

    # Mid-tier neighborhoods (mean SalePrice ~150,000 - 200,000)
    'Crawfor': 'Mid',
    'Gilbert': 'Mid',
    'CollgCr': 'Mid',
    'SawyerW': 'Mid',
    'Blmngtn': 'Mid',

    # Mid-low-tier neighborhoods (mean SalePrice ~100,000 - 150,000)
    'NWAmes': 'MidLow',
    'NAmes': 'MidLow',
    'OldTown': 'MidLow',
    'BrkSide': 'MidLow',
    'Edwards': 'MidLow',
    'Sawyer': 'MidLow',

    # Low-tier neighborhoods (mean SalePrice below ~100,000)
    'IDOTRR': 'Low',
    'MeadowV': 'Low',
    'BrDale': 'Low',
    'NPkVill': 'Low',
    'SWISU': 'Low',
    'Blueste': 'Low'
}

}
    
    


def group_and_replace(dataframe, num=True, cat=True):
    """
    Applies groupings and replacements to specified columns in a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to modify.
    num : bool, optional, default=True
        If True, applies grouping logic to numerical columns such as 'BsmtFullBath', 
        'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', and 'Fireplaces', capping values 
        above specified thresholds.
    cat : bool, optional, default=True
        If True, applies replacements to categorical columns based on the global 
        `replacements` dictionary.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with grouped and replaced values.
    """
    global replacements
    
    # Apply categorical replacements
    if cat:
        for col, replacement in replacements.items():
            if col in dataframe.columns:
                dataframe[col].replace(replacement, inplace=True)
                # Ensure the column is numeric if all replacements are numeric
                dataframe[col] = pd.to_numeric(dataframe[col], errors='ignore')
    
    # Apply numerical capping
    if num:
        if 'BsmtFullBath' in dataframe.columns:
            dataframe['BsmtFullBath'] = dataframe['BsmtFullBath'].apply(lambda x: 1 if x > 1 else x)
        if 'HalfBath' in dataframe.columns:
            dataframe['HalfBath'] = dataframe['HalfBath'].apply(lambda x: 1 if x > 1 else x)
        if 'BedroomAbvGr' in dataframe.columns:
            dataframe['BedroomAbvGr'] = dataframe['BedroomAbvGr'].apply(lambda x: 4 if x > 4 else x)
        if 'TotRmsAbvGrd' in dataframe.columns:
            dataframe['TotRmsAbvGrd'] = dataframe['TotRmsAbvGrd'].apply(lambda x: 10 if x > 10 else x)
        if 'Fireplaces' in dataframe.columns:
            dataframe['Fireplaces'] = dataframe['Fireplaces'].apply(lambda x: 2 if x > 2 else x)
    
    return dataframe



def prep_data(source, delete_list=None, delete_mode=True, encoding=None, sparse=False, replace=None):
    """
    Prepares the dataset by cleaning, transforming, and optionally encoding categorical variables.

    Parameters
    ----------
    source : str or pd.DataFrame
        The source of the dataset, either as a file path to a CSV file or a DataFrame.
    delete_list : list or str, optional
        List of columns to drop (if delete_mode=True) or to keep (if delete_mode=False). 
        If 'default', a predefined list of columns will be used. Defaults to None.
    delete_mode : bool, optional
        - If True, drops columns specified in `delete_list`.
        - If False, keeps only the columns specified in `delete_list`.
        Defaults to True.
    encoding : str, optional
        Type of encoding to apply to categorical variables. Options include:
            - 'ordinal': Performs ordinal encoding and returns a DataFrame with ordinal encoded values.
            - 'onehot': Performs one-hot encoding and returns a sparse matrix (or dense DataFrame if sparse=False).
            - None: Returns the cleaned and filtered DataFrame without encoding.
    sparse : bool, optional
        If True, returns a sparse matrix when encoding='onehot'. Defaults to False.
    replace : str, optional
        Determines whether to group and replace variables using the `group_and_replace` function:
            - 'full': Applies grouping to both numerical and categorical columns.
            - 'num': Applies grouping only to numerical columns.
            - 'cat': Applies grouping only to categorical columns.
            - None: Skips the grouping and replacement step.

    Returns
    -------
    pd.DataFrame or scipy.sparse.csr_matrix
        The processed dataset:
            - A DataFrame if encoding='ordinal' or if sparse=False.
            - A sparse matrix if encoding='onehot' and sparse=True.
            - A cleaned and filtered DataFrame if no encoding is specified.
    
    Notes
    -----
    - The `group_and_replace` function must be defined and handles grouping logic for specific columns.
    - Handles missing values for basement- and garage-related columns, as well as numerical columns with median imputation.
    - Automatically retains and reattaches the 'SalePrice' column if present during one-hot encoding.
    """

    # Lire le fichier CSV
    if type(source) == str:
        data = pd.read_csv(source)
    else:
        data = source

    unusable_cols = ["Id", 'Utilities', 'Alley', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Condition2', 'OverallCond', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'Heating', 'CentralAir', 'Electrical', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'Functional', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PavedDrive', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'SaleType', 'SaleCondition', 'BsmtFinType2']

    # Sauvegarder la colonne 'SalePrice' avant de faire l'encodage
    sale_price = data['SalePrice'] if 'SalePrice' in data else None

    if delete_list is None:
        to_drop = unusable_cols
    elif delete_list == 'default':
        to_drop = ["Id", 'Utilities', 'Alley', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Condition2', 
               'HouseStyle', 'OverallCond', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior2nd', 'MasVnrType', 'ExterQual', 
               'ExterCond', 'BsmtCond', 'BsmtExposure', 'BsmtUnfSF', 'Heating', 'CentralAir', 'Electrical', 'LowQualFinSF', 
               'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'Functional', 'FireplaceQu', 'GarageArea', 'GarageQual', 
               'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
               'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', "LotArea", '1stFlrSF', 
               '2ndFlrSF', 'MasVnrArea', 'LotFrontage', 'BsmtFinSF1', 'GarageYrBlt', 'BedroomAbvGr', 'LotShape', 'BsmtFinType1']
    else:
        to_drop = delete_list if delete_mode else []

    # Gérer les colonnes à garder ou à supprimer
    if delete_mode:
        columns_to_drop = [col for col in to_drop if col in data.columns]
        filtered_data = data.drop(columns=columns_to_drop, axis=1)
    else:
        columns_to_keep = delete_list if delete_list else []
        if 'SalePrice' not in columns_to_keep and 'SalePrice' in data.columns:
            columns_to_keep.append('SalePrice')
        columns_to_keep = [col for col in columns_to_keep if col in data.columns]
        filtered_data = data[columns_to_keep]

    # Remplir les valeurs manquantes pour les colonnes liées au sous-sol et au garage
    bsmnt = ['BsmtQual', 'BsmtFinType1']
    garage = ['GarageType', 'GarageFinish']
    nums = ['LotFrontage', 'MasVnrArea']
    for col in bsmnt:
        if col in filtered_data.columns:
            filtered_data[col] = filtered_data[col].fillna('no_bsmnt')

    for col in garage:
        if col in filtered_data.columns:
            filtered_data[col] = filtered_data[col].fillna('no_garage')
    for col in nums:
        if col in filtered_data.columns:
            filtered_data[col] = filtered_data[col].fillna(filtered_data[col].median())
    if 'GarageYrBlt' in filtered_data.columns:
        filtered_data['GarageYrBlt'] = filtered_data['GarageYrBlt'].fillna(1900)

    # Groupement et remplacement des variables
    if replace == 'full':
        filtered_data = group_and_replace(filtered_data)
    elif replace == 'num':
        filtered_data = group_and_replace(filtered_data, num=True, cat=False)
    elif replace == 'cat':
        filtered_data = group_and_replace(filtered_data, num=False, cat=True)

    # Sélectionner les colonnes catégorielles
    categorical_cols = filtered_data.select_dtypes(include=['object']).columns

    # Si aucun encodage n'est demandé, retourner les données brutes
    if encoding is None:
        return filtered_data

    # Si l'encodage est 'ordinal'
    elif encoding == 'ordinal':
        ordinal_encoder = OrdinalEncoder()
        filtered_data[categorical_cols] = ordinal_encoder.fit_transform(filtered_data[categorical_cols])
        return filtered_data  # Retourne le DataFrame avec l'encodage ordinal

    # Si l'encodage est 'onehot'
    elif encoding == 'onehot':
        # OneHot encoding
        one_hot_encoder = OneHotEncoder(sparse_output=sparse)  # Gérer sparse ici
        encoded_data = one_hot_encoder.fit_transform(filtered_data[categorical_cols])
        
        if sparse:
            # Retourner la matrice sparse si sparse=True
            return encoded_data
        else:
            # Retourner le DataFrame dense si sparse=False
            encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

            # Réajouter la colonne 'SalePrice' à l'encoded_df
            if sale_price is not None:
                encoded_df['SalePrice'] = sale_price

            return encoded_df

    # Si encoding n'est pas spécifié ou reconnu, retourner les données brutes
    return filtered_data



def bin_years_into_decades(df, year_columns):
    """
    Bins years into decades for the specified columns in the DataFrame and ensures
    that houses built before 1900 are assigned to the 1900s.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing year columns to be binned.
    year_columns : list of str
        List of column names representing years to be binned.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the specified year columns replaced by their decade bins.
    """
    for col in year_columns:
        if col in df.columns:
            # Replace years earlier than 1900 with 1900
            df[col] = df[col].apply(lambda x: max(x, 1900))
            # Bin years into decades
            df[col] = (df[col] // 10) * 10  # Divide the year by 10 and multiply back to get the decade
    return df


def convert_and_display(df, 
                        discrete_columns=None, 
                        numerical_columns=None):
    """
    Convertit les colonnes discrètes en entiers et les colonnes continues en flottants,
    puis affiche un résumé clair des colonnes converties.
    
    Parameters:
        df (pd.DataFrame): Le DataFrame à convertir.
        discrete_columns (list, optional): Liste des colonnes discrètes. Si None, utilise une liste par défaut.
        numerical_columns (list, optional): Liste des colonnes continues. Si None, utilise une liste par défaut.
        
    Returns:
        pd.DataFrame: Le DataFrame converti.
    """
    # Définir les listes par défaut
    if discrete_columns is None:
        discrete_columns = [
            'MSSubClass', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'Foundation', 
            'BsmtQual', 'BsmtFinType1', 'HeatingQC', 'BsmtFullBath', 'FullBath', 
            'HalfBath', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 
            'GarageType', 'GarageFinish', 'GarageCars', 'Neighborhood', 'LotShape', 
            'Condition1', 'BldgType', 'HouseStyle', 'OverallQual'
        ]

    if numerical_columns is None:
        numerical_columns = [
            'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
            '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 
            'OpenPorchSF', 'LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd'
        ]

    # Création d'une copie pour éviter de modifier le DataFrame original
    df_converted = df.copy()

    # Conversion des colonnes discrètes en entiers
    for col in discrete_columns:
        if col in df_converted.columns and not pd.api.types.is_integer_dtype(df_converted[col]):
            df_converted[col] = df_converted[col].astype(int)

    # Conversion des colonnes continues en flottants
    for col in numerical_columns:
        if col in df_converted.columns and not pd.api.types.is_float_dtype(df_converted[col]):
            df_converted[col] = df_converted[col].astype(float)

    # Préparer l'affichage des colonnes
    discrete_converted = sorted([col for col in discrete_columns if col in df_converted.columns])
    numerical_converted = sorted([col for col in numerical_columns if col in df_converted.columns])

    print("\n--- Colonnes discrètes converties ---")
    for col in discrete_converted:
        print(f"{col} - Type: {df_converted[col].dtype} - Exemples: {df_converted[col].unique()[:5]}")

    print("\n--- Colonnes continues converties ---")
    for col in numerical_converted:
        print(f"{col} - Type: {df_converted[col].dtype} - Exemples: {df_converted[col].unique()[:5]}")

    return df_converted


def log_transformer(X):
    """
    Apply np.log1p to all numeric columns and maintain DataFrame structure.
    Parameters:
        X (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Transformed DataFrame with log1p applied to numeric columns.
    """
    X_transformed = np.log1p(X.select_dtypes(include=['int64', 'float64']))
    X_rest = X.select_dtypes(exclude=['int64', 'float64'])
    X_combined = pd.concat([X_transformed, X_rest], axis=1)
    return X_combined


