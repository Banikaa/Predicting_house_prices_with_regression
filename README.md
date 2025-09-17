# House Prices - Advanced Regression Techniques
<img width="1000" height="177" alt="kaggle_5407_media_housesbanner" src="https://github.com/user-attachments/assets/f8e11f3e-c9c8-453a-9497-e838140697a0" />

Kaggle competition submission code for house price prediction using regression.
 
Competition Description: Predict the final price of each home (SalePrice) based on house characteristics, location, and quality metrics.

Dataset:
  - Training Data: 1,460 houses with known sale prices
  - Test Data: 1,459 houses requiring price predictions
  - Features: 79 variables including:
  - Categorical Features: Neighborhood, house style, exterior materials, basement conditions
  - Numerical Features: Lot size, living area, number of rooms, garage capacity
  - Ordinal Features: Overall quality, kitchen quality, basement rating
  - Time Features: Year built, year sold, remodel dates

Evaluation metric:
 - the submission csv results are calculated based on RMSE
 - this work achieved a RMSE of 0.13875 (1785/4146)

File Structure:
 - rand_forest.py -> main file, managing the project pipeline, model training and hyperparameter tuning
 - dataset_analysis.py  -> data exploration and feature engineering
 - missing_value_handler.py -> advanced missing data imputation with MCAR, MNAR and MAR method
 - inference.py -> prediction generation and submission
 

 
