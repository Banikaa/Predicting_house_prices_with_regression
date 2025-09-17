import numpy as np


def analyze_missing_data(df):
    missing_analysis = classify_missing_mechanisms(df)
    
    
    mcar_features = [f for f, info in missing_analysis.items() if info['mechanism'] == 'MCAR']
    mar_features = [f for f, info in missing_analysis.items() if info['mechanism'] == 'MAR']
    mnar_features = [f for f, info in missing_analysis.items() if info['mechanism'] == 'MNAR']
    
    print(f"MCAR (Random): {len(mcar_features)} features")
    print(f"MAR (Predictable): {len(mar_features)} features")  
    print(f"MNAR (Meaningful): {len(mnar_features)} features")
    
    # Step 3: Apply appropriate handling
    df_cleaned = handle_missing_by_mechanism(df, missing_analysis)
    
    # Step 4: Verify completion
    remaining_missing = df_cleaned.isnull().sum().sum()
    print(f"\nMissing values after handling: {remaining_missing}")
    
    return df_cleaned


# get missing data and classify by mechanism 
def classify_missing_mechanisms(df):
    missing_analysis = {}
    
    # get features with missing data
    missing_features = df.columns[df.isnull().any()].tolist()
    
    for feature in missing_features:
        missing_count = df[feature].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        # heuristic rules for classification
        if feature in ['PoolQC', 'PoolArea', 'Fence', 'MiscFeature', 'Alley']:
            mechanism = 'MNAR'  # = none
            strategy = 'domain_knowledge'
            
        elif feature in ['GarageType', 'GarageQual', 'GarageCond', 'GarageFinish', 
                        'GarageYrBlt', 'GarageArea', 'GarageCars']:
            mechanism = 'MNAR'  # = no garage
            strategy = 'domain_knowledge'
            
        elif feature in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                        'BsmtFinType2', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']:
            mechanism = 'MNAR'  # = no basement
            strategy = 'domain_knowledge'
            
        elif feature in ['FireplaceQu']:
            mechanism = 'MNAR'  # = no fireplace
            strategy = 'domain_knowledge'
            
        elif missing_percent > 50:
            mechanism = 'MNAR'  # large missing -> likely important
            strategy = 'domain_knowledge'
            
        elif missing_percent < 5:
            mechanism = 'MCAR'  # small missing -> likely random
            strategy = 'simple_imputation'
            
        else:
            mechanism = 'MAR'  # dependent on other features
            strategy = 'predictive_imputation'
        
        missing_analysis[feature] = {
            'count': missing_count,
            'percentage': missing_percent,
            'mechanism': mechanism,
            'strategy': strategy
        }
    
    return missing_analysis


#  based on the classified mechanisms, handle missing data
def handle_missing_by_mechanism(df, missing_analysis):    
    df_clean = df.copy()
    
    for feature, info in missing_analysis.items():
        mechanism = info['mechanism']
        strategy = info['strategy']
        
        print(f"Handling {feature}: {mechanism} -> {strategy}")
        
        if mechanism == 'MNAR':
            df_clean = handle_mnar(df_clean, feature)
        elif mechanism == 'MAR':
            df_clean = handle_mar(df_clean, feature)
        elif mechanism == 'MCAR':
            df_clean = handle_mcar(df_clean, feature)
    
    return df_clean


# missing = NOT random
def handle_mnar(df, feature):
    if 'Pool' in feature:   # Pool
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna('None')
        else:
            df[feature] = df[feature].fillna(0)
    
    elif 'Garage' in feature:   # Garage
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna('None')
        else:
            df[feature] = df[feature].fillna(0)
    
    elif 'Bsmt' in feature:     # Basement
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna('None')
        else:
            df[feature] = df[feature].fillna(0)
    
    elif 'Fireplace' in feature:    # Fireplace      
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna('None')
        else:
            df[feature] = df[feature].fillna(0)
    
    # other features set to None or 0
    else:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna('None')
        else:
            df[feature] = df[feature].fillna(0)
    
    return df


# missing depends on other data -> sample from observed distributions depending on type
def handle_mar(df, feature):
    if df[feature].dtype == 'object':
       df[feature] = sample_cat(df, feature)  
    else:
       df[feature] = sample_num(df, feature)
    return df


#  missing values = random -> sample from observed distributions depending on type
def handle_mcar(df, feature):
    if df[feature].dtype == 'object':
        df[feature] = sample_cat(df, feature) 
    else:
       df[feature] = sample_num(df, feature)
    return df


# sample missing from observed categorical distribution
def sample_cat(df, feature):
    missing_mask = df[feature].isnull()
    if not missing_mask.any():
        return df[feature]
    
    # value counts and probabilities
    value_counts = df[feature].value_counts()
    categories = value_counts.index.tolist()
    probabilities = (value_counts / value_counts.sum()).tolist()
    
    n_missing = missing_mask.sum()
    
    # Ssample based on observed distribution
    sampled_values = np.random.choice(categories, size=n_missing, p=probabilities)
    
    # fill
    filled_feature = df[feature].copy()
    filled_feature.loc[missing_mask] = sampled_values
    
    return filled_feature


# sample missing numeric from observed distribution with noise
def sample_num(df, feature): 
    observed_values = df[feature].dropna()
    missing_mask = df[feature].isnull()
    n_missing = missing_mask.sum()
    
    if n_missing == 0 or len(observed_values) == 0:
        return df[feature]
    
    # calc Normal distribution standard deviation
    std_val = observed_values.std()
    
    # bootstrap samples from observed values
    sampled_values = np.random.choice(observed_values, size=n_missing, replace=True)
    
    # add random noise to introduce variability
    noise_factor = 0.1  # 10% of std dev
    noise = np.random.normal(0, std_val * noise_factor, size=n_missing)
    imputed_values = sampled_values + noise
    
    # fill
    filled_feature = df[feature].copy()
    filled_feature.loc[missing_mask] = imputed_values
    
    return filled_feature


