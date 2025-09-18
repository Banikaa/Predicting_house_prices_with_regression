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
 1. analyse and load the data:
      - csv read and tranformed into a pandas dataframe
      - check the target variable's distribution and Normalise it:
         * plotting the distribution and the QQ plot, as well as calculating the skewness and Kurtosis -> high skewness and tail-heavy distribution -> applying log Normalised the data grealy (skew 1.89 -> 0.12)
      - analysisng and adding missing data: adding the values based on the column type and domain knowledge, using MCAR, MAR and MNAR technique
              - missing values that are not randomly missing are set to '0' or 'None'
              - and the ones that are not random, to keep the model as statistically sound as possible, the missing value is sampled from the observed distribution:
                  * numbers are sampled from the Normal distribution observed with random noise for variance
                  * missing categorical features are sampled from the observed distribution of the populated rows
     - feature engineering: adding more relevant features such as: TotalBathrooms, OverallQuality and binary values (eg. Pool/No Pool).
     - keep most important features based on correlation analysis, where the tresholds for correlation were chosen to keep the total number of features undder 25 as the training dataset has 1460 samples.
     - resulting features are tranformed for model processing with LabelEncoder for training Random Forest algorithm.
2. Same for the Test data but without target variable analysis
3. train Random Forest and do hyperparameter tuning with gridsearch
    - split training set in 80/20, adding validation for more accurate hyperparam tuning results
    - trained best model on the whole training data
4. Model evaluation and saving results in competition format: csv with col: 'Id', 'SalePrice'
      
         
    
  
 

 
