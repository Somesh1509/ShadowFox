# ShadowFox
# Boston House Price Prediction

This project implements a regression model to predict house prices in Boston using the **Boston Housing Dataset**. The dataset contains various features such as crime rates, number of rooms, property tax rates, and other factors that influence housing prices. The model is implemented in **Python** using **Google Colab**.

## Features of the Dataset

The dataset includes the following attributes:

1. **CRIM** - Per capita crime rate by town
2. **ZN** - Proportion of residential land zoned for large lots
3. **INDUS** - Proportion of non-retail business acres per town
4. **CHAS** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. **NOX** - Nitrogen oxide concentration
6. **RM** - Average number of rooms per dwelling
7. **AGE** - Proportion of owner-occupied units built before 1940
8. **DIS** - Weighted distance to five Boston employment centers
9. **RAD** - Index of accessibility to radial highways
10. **TAX** - Property tax rate per $10,000
11. **PTRATIO** - Pupil-teacher ratio by town
12. **B** - 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents in town
13. **LSTAT** - Percentage of lower-status population
14. **MEDV** - Median value of owner-occupied homes (Target Variable)

## Project Workflow

1. **Load Data**: Read the dataset into a Pandas DataFrame.
2. **Preprocessing**: Handle missing values, standardize numerical features.
3. **Train-Test Split**: Divide data into training and testing sets.
4. **Model Training**: Train Linear Regression and Random Forest models.
5. **Evaluation**: Compute metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score.
6. **Visualization**: Plot actual vs predicted house prices for analysis.

## Installation & Dependencies

To run this project in Google Colab, install the necessary libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

## Usage

1. Upload the dataset (HousingData.csv) to Google Colab.
2. Run the provided Python script to preprocess and train models.
3. Evaluate model performance using the printed metrics.
4. Analyze the actual vs. predicted price visualization.

## Results

The model's performance is evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared Score (R²)**


# Car Selling Price Prediction & Analysis 🚗💰

Welcome to the Car Selling Price Prediction & Analysis project! This machine learning model is designed to predict the selling price of used cars based on various features like age, mileage, fuel type, transmission, and more. This project builds upon my previous experience with the Boston House Price Prediction, applying similar regression techniques but adapting them for automobile pricing.

**📌 Project Overview**

This project aims to develop a robust ML model that helps users estimate the selling price of used cars with high accuracy. The model is trained using real-world car sales data and employs advanced data preprocessing, feature engineering, and hyperparameter tuning techniques.
