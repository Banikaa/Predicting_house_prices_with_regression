#  library imports
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy import stats
from sklearn.preprocessing import LabelEncoder


# internal imports
from missing_value_handler import analyze_missing_data, classify_missing_mechanisms, handle_missing_by_mechanism

# analyse the dataframe and do all transformations for the training data
def prepare_train_df(df):
    # check the distribution of the target variable
    analyze_target_variable(df)
    
    # add comprehensive handling for missing data
    cleaned_df = analyze_missing_data(df)

    #  outlier removal was tested based on z-score and IQR methods, but did not improve the model performance

    # create new features based on domain knowledge
    f_eng_df = engineer_house_features(cleaned_df)

    # keep most important features based on correlation analysis
    features = analyze_correlations(f_eng_df)

    X, y, encoders = prepare_features_for_modeling(f_eng_df, features, target='SalePrice')
    
    return X, y, encoders, features


# analyze the dataframe and do all transformations for the test data
def prepare_test_df(df):
    missing_analysis = classify_missing_mechanisms(df)
    cleaned_df = handle_missing_by_mechanism(df, missing_analysis)
    cleaned_df = engineer_house_features(cleaned_df)
    print(f"test data after cleaning: {cleaned_df.shape}")
    return cleaned_df


# feature engineering based on domain knowledge (train and test data)
def engineer_house_features(df):
    df_eng = df.copy()
    current_year = 2025
    
    # 1. total area features
    df_eng['TotalSF'] = df_eng['1stFlrSF'] + df_eng['2ndFlrSF'] + df_eng['TotalBsmtSF']
    df_eng['TotalPorchSF'] = (df_eng.get('OpenPorchSF', 0) + 
                             df_eng.get('EnclosedPorch', 0) + 
                             df_eng.get('3SsnPorch', 0) + 
                             df_eng.get('ScreenPorch', 0))
    
    # 2. age-related features
    df_eng['HouseAge'] = current_year - df_eng['YearBuilt']
    df_eng['YearsSinceRemodel'] = current_year - df_eng.get('YearRemodAdd', df_eng['YearBuilt'])
    df_eng['IsRemodeled'] = (df_eng.get('YearRemodAdd', df_eng['YearBuilt']) != df_eng['YearBuilt']).astype(int)
    
    # 3. total number of bathrooms
    df_eng['TotalBaths'] = (df_eng.get('FullBath', 0) + 
                           0.5 * df_eng.get('HalfBath', 0) + 
                           df_eng.get('BsmtFullBath', 0) + 
                           0.5 * df_eng.get('BsmtHalfBath', 0))
    
    # 4. binary features
    df_eng['HasPool'] = (df_eng.get('PoolArea', 0) > 0).astype(int)
    df_eng['HasGarage'] = (df_eng.get('GarageArea', 0) > 0).astype(int)
    df_eng['HasBasement'] = (df_eng.get('TotalBsmtSF', 0) > 0).astype(int)
    df_eng['HasFireplace'] = (df_eng.get('Fireplaces', 0) > 0).astype(int)
    df_eng['Has2ndFloor'] = (df_eng.get('2ndFlrSF', 0) > 0).astype(int)
    
    # 5. size ratios
    df_eng['LivingAreaRatio'] = df_eng['GrLivArea'] / df_eng['LotArea']
    df_eng['GarageRatio'] = df_eng.get('GarageArea', 0) / df_eng['GrLivArea']
    
    # 6. quality 
    df_eng['QualityCondition'] = df_eng['OverallQual'] * df_eng['OverallCond']
    
    print(f"added {len(df_eng.columns) - len(df.columns)} features")
    
    return df_eng


# take the selected features and prepare the feature matrix and target vector for training
def prepare_features_for_modeling(df, selected_features, target='SalePrice'):
    
    X = df[selected_features].copy()
    y = df[target].copy()

    label_encoders = {}
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le
    
    y_transformed = np.log1p(y)  
    
    print(f"\nFeatures prepared for modeling:")
    print(f"- Feature matrix shape: {X.shape}")
    print(f"- Target shape: {y_transformed.shape}")
    print(f"- Categorical features encoded: {len(categorical_features)}")
    print(f"- Selected features: {list(X.columns)}")
    
    return X, y_transformed, label_encoders

    

#  "salePrice"  in depth analysis (train data only)
def analyze_target_variable(df, target='SalePrice'): 
    target_data = df[target]
    
    # statistics
    print(f"Target: {target}")
    print(f"Mean: ${target_data.mean():,.2f}")
    print(f"Median: ${target_data.median():,.2f}")
    print(f"Min: ${target_data.min():,.2f}")
    print(f"Max: ${target_data.max():,.2f}")
    print(f"Standard Deviation: ${target_data.std():,.2f}\n")
    
    skewness = skew(target_data)
    kurt = kurtosis(target_data)
    log_target = np.log1p(target_data)
    log_skewness = skew(log_target)
    
    print(f'distribution characteristics:\n skewness = {skewness:.4f}, kurtosis = {kurt:.4f} \n log-skewness = {log_skewness:.4f}, log-kurtosis = {kurtosis(log_target):.4f}\n\n')
    
    # PLOTTING - show the effectivness of log-transformation for normalizing the target variable
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # original distribution
    axes[0, 0].hist(target_data, bins=50, alpha=0.7, color='skyblue', density=True)
    axes[0, 0].set_title(f'{target} Distribution (Original)')
    axes[0, 0].set_xlabel(target)
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f'Skewness: {skewness:.3f}', 
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # original Q-Q plot
    stats.probplot(target_data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f'{target} Q-Q Plot (Original)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # log-transformed distribution
    axes[1, 0].hist(log_target, bins=50, alpha=0.7, color='lightgreen', density=True)
    axes[1, 0].set_title(f'Log-transformed {target} Distribution')
    axes[1, 0].set_xlabel(f'log({target})')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, f'Log Skewness: {log_skewness:.3f}', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # log-transformed Q-Q plot
    stats.probplot(log_target, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Log-transformed {target} Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#  analyze correlations and select most important features 
def analyze_correlations(df, target='SalePrice'):
    num_treshold = 0.4
    cat_treshold = 0.5

    # get columns by type
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # calculate similarity for numerical features
    corr_matrix = df[numerical_cols].corr()
    target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
    strong_num_corr = target_corr[abs(target_corr) > num_treshold]

    # calculate similarity for categorical features
    categorical_correlations = []
    for col in categorical_cols:
        if df[col].notna().sum() > 0:
            eta_corr = calculate_correlation_ratio(df, col, target)
            categorical_correlations.append((col, eta_corr))
    strong_cat_corr = [item for item in categorical_correlations if abs(item[1]) > cat_treshold]

    print(f"there are: {len(strong_num_corr)} numerical features strongly correlated with {target} (|corr| > {num_treshold})")
    print(f"there are: {len(strong_cat_corr)} categorical features strongly correlated with {target} (|corr| > {cat_treshold})\n")

    #  consolidate strong features
    strong_features = list(strong_num_corr.index) + [item[0] for item in strong_cat_corr]
    return strong_features


# calculate correlation ratio (eta) for categorical features
def calculate_correlation_ratio(df, cat_col, target_col):
    clean_df = df[[cat_col, target_col]].dropna()
    
    if len(clean_df) == 0:
        return 0.0
    
    # calculate group means
    group_means = clean_df.groupby(cat_col)[target_col].mean()
    overall_mean = clean_df[target_col].mean()
    
    # sum of squares
    ss_between = sum(clean_df.groupby(cat_col).size() * (group_means - overall_mean) ** 2)
    ss_total = sum((clean_df[target_col] - overall_mean) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    eta_squared = ss_between / ss_total
    return eta_squared ** 0.5

