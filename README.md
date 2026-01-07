# House Price Prediction Model Card

This repository contains a machine learning pipeline for predicting house prices using scikit-learn. Features include preprocessing, feature engineering, and models like Gradient Boosting and Random Forest. Includes a .ipynb, submission file, and model card. Ideal for educational purposes, real estate and predictive modeling projects.

# Basic Information
* **Group Members:**
   * **Prajwal Kusha - p.kusha@gwu.edu**
   * **Manasi Jha - m.jha@gwu.edu**
     
* **Model Date: December, 2024**
* **Model Version: 1.0**
* **License: MIT** 
* **Model Implementation Code: [House_Price_Prediction_Model](https://github.com/PrajwalKusha/House_Price_Prediction/blob/main/house-price-prediction-model.ipynb)**

## Intended Use:
* **Intended Uses:** Predict house sale prices based on property features for real estate analysis and market trend predictions.
* **Intended Users:** Real estate professionals, data analysts, and researchers.
* **Out-of-Scope Uses:** This model is not suitable for applications requiring legal or financial guarantees or for use outside the distribution of the training dataset.

# Training Data: 
* **Source:** The dataset is sourced from **Kaggle's "House Prices: Advanced Regression Techniques"** competition.

* **How Data was divided:**
  * **80% training data**
  * **20% validation data**

* **Data Dictionary:**
 
| Name           | Modeling Role | Measurement Level | Description                             |
|----------------|---------------|-------------------|-----------------------------------------|
| MSSubClass     | Predictor     | Categorical       | Building class                          |
| LotFrontage    | Predictor     | Continuous        | Lot size in feet                        |
| PropertyAge    | Predictor     | Continuous        | Age of the property at sale             |
| OverallQual    | Predictor     | Ordinal           | Overall material and finish quality     |
| SalePrice      | Target        | Continuous        | Sale price of the house                 |

# Test Data:
* **Source:** Kaggle's **"House Prices: Advanced Regression Techniques" test dataset.**
* **Number of Rows:** 1459 rows.

* **Differences:**
  * The test dataset lacks the `SalePrice` column, which is the target variable in the training dataset.
 
# Model Details:
1. **Columns Used as Inputs in the Final Model**
 * **Numerical Features:**
   * These features were imputed (for missing values) using the median and scaled appropriately (if required).
   * Examples: `OverallQual`, `GrLivArea`, `GarageCars`, `PropertyAge`, `LotArea`, etc.
   * Features engineered: `PropertyAge`: Derived as `YrSold - YearBuilt`.

 2. **Categorical Features:**
   * These features were one-hot encoded after imputing missing values with the most frequent category.
   * Examples: `MSZoning`, `Neighborhood`, `BldgType`, `HouseStyle`, `ExterQual`, `KitchenQual`, etc.

3. **Columns Used as Targets in the Final Model**
 * **Target Variable:**
   * `SalePrice` (continuous numerical variable): Represents the sale price of the house.

4. **Type of Model**
  * Gradient Boosting Regressor: A tree-based ensemble model that iteratively trains weak learners (decision trees) and aggregates their predictions to minimize the residual errors.

5. **Software used to implement the model:**
  * **Python Programming Language**
  * **Libraries:**
    * `scikit-learn`: For preprocessing, pipeline creation, hyperparameter tuning, and modeling.
    * `matplotlib` and `seaborn`: For visualization.
    * `pandas` and `numpy`: For data manipulation and analysis.

 * **Software Version**
   * **Python Version: 3.9**
   * **scikit-learn Version: 1.2.2**

6. **Hyperparameters**
 * **Hyperparameter Tuning for Gradient Boosting Regressor:**
   * Used `GridSearchCV` to find the best combination of hyperparameters.
   * Final Hyperparameters:
     * `n_estimators`: 200 (Number of boosting stages).
     * `max_depth`: 3 (Maximum depth of individual trees to prevent overfitting).
     * `learning_rate`: 0.1 (Controls the contribution of each tree to the final prediction).

# Quantitative Analysis
* **Metrics Used:**
  * **Root Mean Squared Error (RMSE):** Measures the average error magnitude between actual and predicted values.
* **Final Values:**

| Dataset      | RMSE       |
|--------------|------------|
| Training     | 26348.71   |
| Validation   | 27897.11   |
| Test (estimate) | TBD        |

 **Note**: The test RMSE is not included because the true labels (`SalePrice`) for the test dataset are not available in the provided data. Validation RMSE is used as an estimate of the test performance.

* **Plots:**

**1. Feature Importance Plot:**
 * **Description:** The plot ranks the features by their importance in predicting house prices. The top features contributing to the model's accuracy are shown.
 * **Top Features:** OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF
 * **Insights:** The model relies heavily on quality and size-related features to predict the target variable.

![Screenshot 2024-12-08 at 3 41 49 PM](https://github.com/user-attachments/assets/02e46cd3-e7da-4044-b3fa-b5af8e2544cc)

**2. Correlation Heatmap:**
 * **Description:** The heatmap displays the correlation matrix of numerical features, helping identify multicollinearity and key predictors of SalePrice.
 * **Insights:** OverallQual, GrLivArea, and TotalBsmtSF are strongly positively correlated with SalePrice, while BsmtUnfSF shows a weaker correlation.

![Screenshot 2024-12-08 at 3 42 23 PM](https://github.com/user-attachments/assets/69c66588-3c13-4869-b749-20653095f462)
    
**3. Model Performance Comparison**
 * **Description:** A bar chart comparing the RMSE scores of the Random Forest and Gradient Boosting models on validation data.
 * **Insights:** Gradient Boosting performs slightly better than Random Forest, making it the preferred model for predictions.
  
![Screenshot 2024-12-08 at 3 42 29 PM](https://github.com/user-attachments/assets/d8069ec6-632f-41fc-a4dd-164b79f5dcb8)

# Potential Negative Impacts

**1. Math or Software Problems** 
 * **Data Preprocessing Errors:**
  * Incorrect handling of missing data could introduce bias, leading to predictions that are systematically inaccurate for certain types of properties (e.g., properties with missing values in LotFrontage or MasVnrArea).
  * Inconsistent encoding of categorical variables may lead to mismatches between training and test datasets, reducing model performance.
    
 * **Model Overfitting or Underfitting:**
  * Overfitting to patterns in the training data could cause the model to perform poorly on unseen properties.
  * Simplistic assumptions in feature engineering (e.g., linear treatment of property age) may overlook complex relationships between features and sale prices.

**2. Real-World Risks**
 * **Discrimination:**
   * The model may undervalue properties in underrepresented neighborhoods due to insufficient training data, perpetuating existing inequalities in housing markets.
   * Features like Neighborhood may act as proxies for socioeconomic status, potentially introducing biases if not handled carefully.

 * **Financial Risks:**
   * Overreliance on the model could lead to poor investment or pricing decisions by real estate professionals and buyers, especially in volatile markets.
   * High-stakes decisions, such as mortgage approvals, may be unjustly influenced by biased or inaccurate predictions.
  
# Uncertainities 

**1. Math or Software Problems** 
 * **Rare Categories:**
  * Properties with rare categories in features (e.g., unique RoofMatl or Exterior1st values) may have undefined or poorly calibrated predictions if these categories were underrepresented or missing in the training data.
    
 * **Impact of Outliers:**
  * Outliers in the training data (e.g., extremely high SalePrice values) may skew the model, resulting in overpredictions or underpredictions for typical properties.

 * **Feature Drift:**
  * The test data may exhibit characteristics not present in the training data, leading to decreased model performance. For instance, changes in construction trends (e.g., new roofing materials) could introduce unseen categories.

**2. Real-World Risks**
 * **Economic Shifts:**
   * The model does not account for broader macroeconomic factors such as inflation, interest rates, or housing market cycles, which can significantly impact property prices.

 * **Geographic  Generalizability:**
   * Predictions may not generalize well to regions with fundamentally different housing markets if the training data is geographically localized.
  
# Unexpected Results

**1. Rare Features** 
  * Properties with unusual architectural styles, exceptionally large or small lots, or unique amenities may be inaccurately predicted because such features were underrepresented in the training dataraining data.
  * Conversely, properties missing critical features (e.g., no GarageArea) may be undervalued if the model overemphasizes these features.
    
**2. Interpreting Feature Importance**
  * Misinterpretation of feature importance by users could lead to undue emphasis on features with spurious correlations (e.g., KitchenQual may seem important but could be influenced by other underlying factors).

**3. Bias in Validation**
  * If the validation data does not fully represent the diversity of the housing market, the reported RMSE could give an overly optimistic view of the model's performance on real-world data.
