import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



##### Import data

file_path = './train.csv'

df = pd.read_csv(file_path)


##### X and y selection for first model (I am using a list of features obtained from Kaggle that are error-free. )


list_disorganized = "'MSSubClass' 'LotArea' 'OverallQual' 'OverallCond' 'YearBuilt' 'YearRemodAdd' '1stFlrSF' '2ndFlrSF' 'LowQualFinSF' 'GrLivArea' 'FullBath' 'HalfBath' 'BedroomAbvGr' 'KitchenAbvGr' 'TotRmsAbvGrd' 'Fireplaces' 'WoodDeckSF' 'OpenPorchSF' 'EnclosedPorch' '3SsnPorch' 'ScreenPorch' 'PoolArea' 'MiscVal' 'MoSold' 'YrSold'"

features = list_disorganized.split()

features = [word.strip("'") for word in features]

X_all_features = df[features]



y = df['SalePrice']

##### Get Mean Absolute Error function

def get_mae(X, y):
    '''Calculate the Mean Absolute Error (MAE) for a random forest regression model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.

    y : array-like, shape (n_samples,)
        Target values.

    Returns
    -------
    float
        The Mean Absolute Error (MAE) between the validation target values and the model predictions.
    '''
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state= 1)
    test_model = RandomForestRegressor(random_state= 1)
    test_model.fit(train_X, train_y)
    test_model_prediction = test_model.predict(val_X)
    return mean_absolute_error(val_y, test_model_prediction)



###### create and test a model with every features

all_features_model = RandomForestRegressor(random_state = 1)

mae_all_features = get_mae(X_all_features, y)


all_features_model.fit(X_all_features, y)

print(mae_all_features)

##### feature importance method

# Get an array with all the features importances with the features_importances_ method
importances_num = all_features_model.feature_importances_

# Get a list of tuples with the column names and the importances
importances = list(zip(features, importances_num))

# Get my list of feature importances in a descending order
def get_second_element(x):
    '''Return the second element of the given list or tuple.

    Parameters
    ----------
    x : list or tuple
        The input list or tuple.

    Returns
    -------
    object
        The second element of the input list or tuple.

    '''
    return x[1]

importances.sort(key= get_second_element, reverse= True)

n_features = len(features)

# Select the top `n_features` features based on their importances
selected_features = [feature for feature, _ in importances[:n_features]]

# Get a list of the mae with n most important features with different numbers of features
mae_list = []

for i in range(1, n_features + 1):
    mae_list.append(get_mae(df[selected_features[:i]], y))

######### Build a model using the number of features with the lowest mean absolute error
    
n_features = mae_list.index(min(mae_list))+ 1

selected_features = [feature for feature, _ in importances[:n_features]]

X_feature_importance = df[selected_features]

model_feature_importance = RandomForestRegressor(random_state= 1)

model_feature_importance.fit(X_feature_importance, y)

mae_feature_importance = get_mae(X_feature_importance, y)

print(mae_feature_importance)

############ Submission

# Import test dataframe
file_path_test = './test.csv'
df_test = pd.read_csv(file_path_test)
ids = df_test["Id"].values
X_test = df_test[selected_features]
predictions = model_feature_importance.predict(X_test)

# Create a dataframe to submit
submission_df = pd.DataFrame({'Id': ids, 'SalePrice': predictions})

# Naming the file
base_filename = "submission"
file_extension = ".csv"
folder_path = "./"

if os.path.isfile(os.path.join(folder_path, "submission.csv")):
    counter = 2
    while os.path.isfile(os.path.join(folder_path, f"{base_filename}_{counter}{file_extension}")):
        counter += 1
    submission_filename = os.path.join(folder_path, f"{base_filename}_{counter}{file_extension}")
else:
    submission_filename = os.path.join(folder_path, "submission.csv")

# Create the file
submission_df.to_csv(submission_filename, index=False)
