import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from dataset_analysis import prepare_train_df, prepare_test_df
from inference import generate_predictions


#  train random forest model with hyperparameter tuning
def train_random_forest_model(X, y):
    
    param_grid = {  # hyperparameter tuning gridsearch
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # innit model
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"best parameters: {grid_search.best_params_}\n")
    
    # init model with best params and train all all data (train + val)
    best_rf = RandomForestRegressor(
        **grid_search.best_params_,
        random_state=42
    )
    best_rf.fit(X, y)
    
    full_train_pred = best_rf.predict(X)
    full_train_rmse = np.sqrt(mean_squared_error(y, full_train_pred))
    full_train_r2 = r2_score(y, full_train_pred)
    
    print(f"RMSE: {full_train_rmse:.4f}, R^2: {full_train_r2:.4f}")
    
    return best_rf


if __name__ == "__main__":
    # load datasets
    train_df = pd.read_csv('/Users/banika/code/side_projects/kaggle/Predicting_house_prices_with_regression/house-prices-advanced-regression-techniques/train.csv')
    test_df = pd.read_csv('/Users/banika/code/side_projects/kaggle/Predicting_house_prices_with_regression/house-prices-advanced-regression-techniques/test.csv')

    # get cleaned data and relevant features
    X, y, encoders, train_features = prepare_train_df(train_df)
    
    # train model
    model = train_random_forest_model(X, y)

    # generate predictions for test set
    generate_predictions(model, test_df, train_features, encoders)

    