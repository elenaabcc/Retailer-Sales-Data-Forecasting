# Sales Forecasting Project

## Overview

Forecast sales data for different states. 
The workflow includes:
-  data **preprocessing** like handling missing values, 
- **training** using models like SARIMA,  XGBOOST and ..., 
-  **evaluating model performance** using MSE, RMSE, MAE and ...

Additionally, the project generates and saves visualizations of the sales predictions compared to actual sales data.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Handling Missing Values](#handling-missing-values)
3. [Modeling](#modeling)
4. [Evaluation](#evaluation)
5. [Visualization](#visualization)
6. [Future Work](#future-work)

## Data Preparation

1. **Data Loading**:

    - Use `Retailer Sales Data.csv`.

2. **Order Date nan Handling**: 

    - Handles missing values in `Order Date` by estimating based on the average shipping days.

3. **Sales Handling**:
   - Fills missing sales values using the average sales for the `Sub-Category` or `Category` is sub-cat is also missing
   
4. **State Handling**:
   - Fills missing values in the `State` column based on the `Customer ID` and `Country`.




#  Top-Down Approach

1. **Data Aggregation**:
   - Aggregates sales data by `Year-Month` and `State`.

2. **Train-Test Split**:
   - Splits the data into training and testing datasets based on the date.
3. **Proportion of Forecasting by States Based on Previous Sales Year**
    - Generate state-specific forecasts that are aligned with historical sales trends


## Modeling w SARIMA Model

Defines and trains a SARIMA model on the training data and makes forecasts for the test period.


## Evaluation

1. **Metrics Calculation**:
   - Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) for the overall sales forecast.

2. **State-wise Evaluation**:
   - Calculates evaluation metrics for each state by comparing predicted and actual sales.

3. **Results Saving**:
   - Saves the evaluation results (MSE, RMSE, MAE) in sarima_state_metrics.csv

## Visualization

1. **Generate Plots**:
   - Creates plots comparing predicted and actual sales for each state.
   - Saves plots in a directory named `state_sales_figures`.


![Missisipi Predictions vs Actual Graph](sarima_state_sales_figures/Mississippi_sales.png)

## Modeling w XGBOOSTING Model

