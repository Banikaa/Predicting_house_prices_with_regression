# House Prices - Advanced Regression Techniques
<img width="1000" height="177" alt="kaggle_5407_media_housesbanner" src="https://github.com/user-attachments/assets/f8e11f3e-c9c8-453a-9497-e838140697a0" />

Kaggle competition submission code for house price prediction using regression.
 
Competition Description: Predict the final price of each home (SalePrice) based on house characteristics, location, and quality metrics.

 <h2>Dataset:</h2>
  - Training Data: 1,460 houses with known sale prices
  - Test Data: 1,459 houses requiring price predictions
  - Features: 79 variables including:
  - Categorical Features: Neighborhood, house style, exterior materials, basement conditions
  - Numerical Features: Lot size, living area, number of rooms, garage capacity
  - Ordinal Features: Overall quality, kitchen quality, basement rating
  - Time Features: Year built, year sold, remodel dates
  

<h2>Evaluation metric:</h2>
 - the submission csv results are calculated based on RMSE
 - this work achieved a RMSE of 0.13875 (1785/4146)

<h2>File Structure:</h2>

 - rand_forest.py -> main file, managing the project pipeline, model training and hyperparameter tuning
 - dataset_analysis.py  -> data exploration and feature engineering
 - missing_value_handler.py -> advanced missing data imputation with MCAR, MNAR and MAR method
 - inference.py -> prediction generation and submission

<h2>Pipeline:</h2>
 1. Analyse and load the data  
   - CSV read and transformed into a pandas dataframe  
   - Check the target variable's distribution and normalise it:  
     * Plotting the distribution and the QQ plot, as well as calculating the skewness and kurtosis  
     * High skewness and tail-heavy distribution → applying log normalisation improved skewness (1.89 → 0.12)  
   - Analyse and handle missing data:  
     - Missing values that are not randomly missing are set to `'0'` or `'None'`  
     - For values assumed MCAR/MAR/MNAR, impute based on observed distribution:  
       * Numbers → sampled from the normal distribution with random noise for variance  
       * Categorical → sampled from the observed distribution of populated rows  
   - Feature engineering: add more relevant features such as `TotalBathrooms`, `OverallQuality`, and binary values (e.g. Pool/No Pool)  
   - Keep most important features based on correlation analysis (threshold chosen to keep <25 features as dataset has 1460 samples)  
   - Resulting features transformed for model processing with `LabelEncoder` for Random Forest training  

2. Same process for the test data (without target variable analysis)  

3. Train Random Forest and tune hyperparameters with GridSearch  
   - Split training set 80/20 with validation for more accurate tuning results  
   - Train the best model on the whole training dataset  

4. Model evaluation and saving results in competition format (`csv` with columns: `Id`, `SalePrice`)  

      
         
    
  
 

 
