# Used Car Price Prediction and Visualization

## Project Overview

This project focuses on predicting the prices of used cars based on various attributes such as the car's brand, model, mileage, fuel type, engine, transmission, and other features. The objective is to provide valuable insights into how different car attributes affect their price and to build a predictive model that can estimate the price of used cars accurately.

## Key Features

- **Data Preprocessing**: Handling missing values, cleaning, and converting data to suitable formats for machine learning.
- **Exploratory Data Analysis (EDA)**: Visualizations and correlation analysis to understand relationships between variables.
- **Machine Learning Models**: Using **Random Forest Regression** and **Linear Regression** to predict car prices.
- **Hyperparameter Tuning**: Fine-tuning the Random Forest model using **RandomizedSearchCV** for optimal performance.
- **Evaluation Metrics**: Evaluating the model performance using metrics like **MAE**, **RMSE**, and **R² Score**.

## Dataset

The dataset used in this project contains information about used cars including:

- **brand**: Brand of the car (e.g., Ford, Hyundai, Audi)
- **model**: Car model name
- **model_year**: The year the car was manufactured
- **milage**: Mileage of the car (in miles)
- **fuel_type**: The type of fuel the car uses (e.g., Gasoline, Hybrid, Electric)
- **engine**: Engine specifications
- **transmission**: Type of transmission (e.g., Automatic, Manual)
- **ext_col**: Exterior color of the car
- **int_col**: Interior color of the car
- **accident**: Whether the car has been in an accident (Yes/No)
- **clean_title**: Whether the car has a clean title (Yes/No)
- **price**: The price of the car (target variable)

## Steps Involved

### 1. **Data Preprocessing**
   - Handling missing values.
   - Cleaning columns such as **milage** and **price**.
   - Converting columns to appropriate data types.

### 2. **Exploratory Data Analysis (EDA)**
   - Visualizing relationships between variables using **seaborn** and **matplotlib**.
   - Analyzing the correlation between different features and the target variable (price).
   
### 3. **Feature Engineering**
   - Extracting features from the **engine** column such as **horse power** and **liter capacity**.
   - Extracting the **number of cylinders** from the engine specifications.
   - Encoding categorical features using **OneHotEncoder**.

### 4. **Modeling**
   - Building and training **Random Forest Regression** and **Linear Regression** models.
   - Evaluating model performance using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.
   - Hyperparameter tuning of the **Random Forest model** using **RandomizedSearchCV** to improve performance.

### 5. **Model Evaluation**
   - Comparing the performance of the initial and tuned models.
   - Analyzing the feature importance to understand which attributes are most significant in predicting car prices.

## Model Performance

The final model achieved the following results after hyperparameter tuning:

- **Mean Absolute Error (MAE)**: 5795.2394
- **Root Mean Squared Error (RMSE)**: 8705.9198
- **R² Score**: 0.8395

## Future Improvements

- Implementing other machine learning algorithms such as **Gradient Boosting** or **XGBoost** for comparison.
- Exploring advanced feature engineering methods to further improve model accuracy.
- Incorporating additional external data sources for better prediction performance.

## Dependencies

The following Python libraries are required to run this project:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
