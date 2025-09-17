import pandas as pd
import numpy as np
from dataset_analysis import prepare_test_df


# prepare test data similar to train data
def prepare_test_data(test_df, train_features, label_encoders):
    cleaned_test_df = prepare_test_df(test_df)
    X_test = cleaned_test_df[train_features].copy()
    
    categorical_features = X_test.select_dtypes(include=['object']).columns.tolist()
    
    for feature in categorical_features:
        le = label_encoders[feature]
        X_test[feature] = X_test[feature].astype(str) 
        X_test[feature] = le.transform(X_test[feature])
        
    return X_test


# generate predictions and save to CSV
def generate_predictions(model, test_df, train_features, label_encoders, output_file='submission.csv'):
    X_test = prepare_test_data(test_df, train_features, label_encoders)
    
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)  # reverse log1p transformation
    
    # save to competition submission format
    pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': predictions
    }).to_csv(output_file, index=False)
