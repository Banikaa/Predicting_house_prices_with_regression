# House Prices - Advanced Regression Techniques
<img width="1000" height="177" alt="kaggle_5407_media_housesbanner" src="https://github.com/user-attachments/assets/f8e11f3e-c9c8-453a-9497-e838140697a0" />

Kaggle competition submission code for house price prediction using regression.
 
Competition Description: Predict the final price of each home (SalePrice) based on house characteristics, location, and quality metrics.

 <h2>Dataset:</h2>
  <ul>
  <li>Training Data: 1,460 houses with known sale prices</li>
  <li>Test Data: 1,459 houses requiring price predictions</li>
  <li>
    Features: 79 variables including:
    <ul>
      <li>Categorical Features: Neighborhood, house style, exterior materials, basement conditions</li>
      <li>Numerical Features: Lot size, living area, number of rooms, garage capacity</li>
      <li>Ordinal Features: Overall quality, kitchen quality, basement rating</li>
      <li>Time Features: Year built, year sold, remodel dates</li>
    </ul>
  </li>
</ul>

  

<h2>Evaluation metric:</h2>
 - the submission csv results are calculated based on RMSE
 - this work achieved a RMSE of 0.13875 (1785/4146)

<h2>File Structure:</h2>

 - rand_forest.py -> main file, managing the project pipeline, model training and hyperparameter tuning
 - dataset_analysis.py  -> data exploration and feature engineering
 - missing_value_handler.py -> advanced missing data imputation with MCAR, MNAR and MAR method
 - inference.py -> prediction generation and submission

<h2>Pipeline:</h2>
 <ol>
  <li>
    Analyse and load the data
    <ul>
      <li>CSV read and transformed into a pandas dataframe</li>
      <li>
        Check the target variable's distribution and normalise it:
        <ul>
          <li>Plot the distribution and QQ plot; compute skewness and kurtosis</li>
          <li>High skewness and tail-heavy distribution → log normalisation improved skew (1.89 → 0.12)</li>
        </ul>
      </li>
      <li>
        Analyse and handle missing data:
        <ul>
          <li>For non-random missingness, set values to <code>0</code> or <code>None</code></li>
          <li>For MCAR/MAR/MNAR, impute from observed distributions:
            <ul>
              <li>Numeric → sample from observed normal distribution with noise</li>
              <li>Categorical → sample from observed category frequencies</li>
            </ul>
          </li>
        </ul>
      </li>
      <li>Feature engineering: add <code>TotalBathrooms</code>, <code>OverallQuality</code>, and binary flags (e.g., Pool / No Pool)</li>
      <li>Select most important features via correlation (thresholds chosen to keep &lt; 25 features given 1,460 samples)</li>
      <li>Transform features for model processing with <code>LabelEncoder</code> for Random Forest training</li>
    </ul>
  </li>

  <li>Same process for the test data (without target variable analysis)</li>

  <li>
    Train Random Forest and tune hyperparameters with GridSearch
    <ul>
      <li>Split training set 80/20; use validation for more accurate tuning</li>
      <li>Train the best model on the full training data</li>
    </ul>
  </li>

  <li>Model evaluation and save results in competition format: CSV with columns <code>Id</code>, <code>SalePrice</code></li>
</ol>


      
         
    
  
 

 
