# CAP4770FinalProject

# Project Overview

This project aims to build a machine learning model to predict house prices based on a comprehensive dataset of property characteristics. By leveraging real-world housing data and advanced machine learning techniques, the project provides a scalable, data-driven solution to accurately estimate house prices while uncovering key factors driving property valuations.

# Introduction

Accurate house price prediction is crucial for buyers, sellers, real estate agents, and financial institutions. House prices are influenced by numerous factors such as property size, location, amenities, and market conditions. This project uses a dataset sourced from Kaggle to train and evaluate machine learning models for predicting house prices.

Key objectives of this project include:

* Developing a machine learning model that minimizes prediction errors.
* Understanding the most influential factors affecting house prices.
* Providing a framework for future property valuation models.

# Dataset
* Source  [Kaggle] (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
* Size: 545 records of housing data.
Key Features:
* Numerical: price (target variable), area, bedrooms, bathrooms, stories, parking.

* Categorical: furnishingstatus, mainroad, guestroom, basement, airconditioning, hotwaterheating, prefarea.

# Workflow

Data Collection:
* The dataset was downloaded from Kaggle and loaded into a SQL Server database for efficient querying and manipulation.

Data Preprocessing:
* Handled missing values (imputation).
* Encoded categorical variables (e.g., one-hot encoding for furnishingstatus).
* Normalized numerical features for compatibility with machine learning models.

Feature Engineering:

Created interaction features:
* area_per_bedroom = area / bedrooms
* bathroom_to_bedroom_ratio = bathrooms / bedrooms
* Introduced a binary luxury_indicator feature based on the presence of air conditioning, parking spaces, and furnished status.

Modeling:
* Trained multiple models (Random Forest, XGBoost) to predict house prices.
* Performed hyperparameter tuning using GridSearchCV for optimal Random Forest parameters.

Evaluated models using metrics:
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Evaluation and Insights:

* Visualized actual vs. predicted prices, residuals, and feature importance.
* Identified area, bathrooms, and airconditioning as the most influential features.

# Results

Hyperparameters:
* max_depth = 10
* min_samples_leaf = 2
* min_samples_split = 10
* n_estimators = 100

Performance:
* MAE: 0.563
* RMSE: 0.728
The model outperformed a baseline predictor based on the mean price, which had an MAE of 0.944.

# Visualizations 
Key visualizations included:

* Actual vs. Predicted Prices: Demonstrates alignment between model predictions and actual values.
* Residual Analysis: Highlights the absence of systematic errors in predictions.
* Feature Importance: Shows that area, bathrooms, and airconditioning are the top drivers of house prices.

# How to Run the Project
Prerequisites:

* Python 3.8+
* Install the required Python libraries:
```pip install pandas numpy scikit-learn matplotlib seaborn sqlalchemy joblib```

Ensure access to SQL Server for querying the dataset (or use the provided CSV file).

# Steps

* Clone this repository:
```git clone <repository-url>```
```cd <repository-folder>```

* Open the Jupyter Notebook:

```jupyter notebook house_price_prediction.ipynb```

Follow the steps in the notebook to:
* Preprocess the dataset.
* Train and evaluate machine learning models.
* Visualize results.

Save the final model for deployment:

```import joblib```
```joblib.dump(final_model, "final_house_price_model.pkl")```



