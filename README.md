House Price Prediction Model
============================

This Python script presents a straightforward house price prediction model utilizing linear regression with polynomial features. The dataset employed for training the model is expected to be in CSV format and is loaded from the file 'kc_house_data.csv'. This script is part of the House Price Prediction Decision Support System Project, which aims to develop a robust system for predicting house prices.

Project Overview
-------------

The House Price Prediction Decision Support System Project endeavors to create a house price prediction system tailored for use by real estate agents and individuals seeking an estimate of a house's price. The system leverages a machine learning model trained on a dataset of house prices in the United States, utilizing the Linear Regression algorithm.

- **Frontend and Backend Structures**: Explore the project's [frontend](https://github.com/senaoz/dss-frontend) and [backend](https://github.com/senaoz/dss-backend) structures here.

Prerequisites
-------------

Make sure you have the following libraries installed:

`pip install numpy pandas matplotlib seaborn scikit-learn`

Usage
-----

1.  Clone the repository:

`git clone https://github.com/your_username/your_repository.git cd your_repository`

1.  Open the Python script 'house_price_prediction.py' in your preferred Python environment (e.g., Jupyter Notebook, PyCharm, VSCode).

2.  Run the script to train the model and make predictions.

Description
-----------

### Data Preprocessing

The script loads the dataset, performs some basic preprocessing, and splits the data into features (X) and target variable (y).

### Model Training

The `train_model` function uses polynomial features of degree 2 and linear regression to train the prediction model.

### Prediction

The `predict` function takes an input data array and returns the predicted house price using the trained model.

### Model Evaluation

The script evaluates the model's performance using mean squared error (MSE) and R-squared (R2) on a test set.

### Exploratory Data Analysis (EDA)

The script includes various EDA sections, such as:

-   **Correlation Matrix**: Visualizes the correlation between numerical features using a heatmap.

-   **Outlier Analysis**: Displays boxplots to identify outliers in numerical features.

-   **Feature Engineering**: Utilizes distribution plots to analyze the distribution of numerical features.

### Model Analysis

The script provides insights into the model's performance, including:

-   **Quality Metrics**: Displays MSE for both the training and test sets.

-   **Feature Importance**: Ranks features based on their absolute coefficients in the trained model.

-   **Prediction Analysis**: Demonstrates a single data point's prediction, actual value, and relevant metrics.
