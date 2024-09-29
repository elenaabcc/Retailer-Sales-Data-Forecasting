[19:10, 9/29/2024] Albi🍺 🐞🌻: # Top-Down Approach - Part 1: XGBoost Modeling

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# 1. Data Preparation

# Aggregate sales data by 'Year-Month'
monthly_sales = df.groupby('Year-Month')['Sales'].sum().reset_index()

# Sort by 'Year-Month'
monthly_sales = monthly_sales.sort_values('Year-Month')

# Create time-based features
monthly_sales['Month'] = monthly_sales['Year-Month'].dt.month
monthly_sales['Year'] = monthly_sales['Year-Month'].dt.year

# Generate lag features (e.g., sales from the previous 3 months)
for lag in range(1, 4):
    monthly_sales[f'lag_{lag}'] = monthly_sales['Sales'].shift(lag)

# Drop rows with NaN values due to lagging
monthly_sales.dropna(inplace=True)

# Encode cyclical features
monthly_sales['Month_sin'] = np.sin(2 * np.pi * monthly_sales['Month']/12)
monthly_sales['Month_cos'] = np.cos(2 * np.pi * monthly_sales['Month']/12)

# Define feature columns
feature_cols = ['Month_sin', 'Month_cos', 'Year'] + [f'lag_{lag}' for lag in range(1, 4)]

# Define target variable
X = monthly_sales[feature_cols]
y = monthly_sales['Sales']

# 2. Split the data into training and testing sets based on date
train_data = monthly_sales[monthly_sales['Year-Month'] < '2018-01']
test_data = monthly_sales[monthly_sales['Year-Month'] >= '2018-01']

train_X = train_data[feature_cols]
train_y = train_data['Sales']
test_X = test_data[feature_cols]
test_y = test_data['Sales']

# 3. Feature Scaling
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# 4. Model Training

# Initialize the XGBoost regressor
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train the model
xgb_model.fit(train_X_scaled, train_y)

# 5. Forecasting

# Make predictions on the test set
predictions = xgb_model.predict(test_X_scaled)

# Create a Series for forecasted values
forecast_series = pd.Series(predictions, index=test_data['Year-Month'])

# 6. Evaluation

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Year-Month'], monthly_sales['Sales'], label='Historical Sales', color='blue')
plt.plot(test_data['Year-Month'], predictions, label='XGBoost Forecast', color='red', linestyle='--')
plt.title('XGBoost Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_y, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 7. Disaggregate Predictions by State

# Calculate the proportion of sales by state from historical data
sales_by_state = df.groupby(['Year-Month', 'State'])['Sales'].sum().unstack().fillna(0)
state_proportions = sales_by_state.div(sales_by_state.sum(axis=1), axis=0)

# Allocate forecasted total sales proportionally across states
forecast_proportioned = pd.DataFrame(index=forecast_series.index, columns=state_proportions.columns)

for state in state_proportions.columns:
    forecast_proportioned[state] = forecast_series * state_proportions[state]

# Save the forecasted proportions if needed
forecast_proportioned.to_csv('xgboost_state_forecasts.csv')

# 8. Save Evaluation Metrics

# Create a DataFrame for evaluation metrics
evaluation_metrics = pd.DataFrame({
    'Model': ['XGBoost'],
    'MSE': [mse],
    'RMSE': [rmse],
    'MAE': [mae]
})

# Save metrics to a CSV file
evaluation_metrics.to_csv('xgboost_evaluation_metrics.csv', index=False)

print("XGBoost modeling completed and evaluation metrics saved.")
[21:25, 9/29/2024] Albi🍺 🐞🌻: # Retailer Sales Data Forecasting

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Methodology](#methodology)
  - [Part 1: Top-Down Approach](#part-1-top-down-approach)
    - [1. Data Aggregation](#1-data-aggregation)
    - [2. SARIMA Model Training](#2-sarima-model-training)
    - [3. Sales Disaggregation](#3-sales-disaggregation)
    - [4. Model Evaluation](#4-model-evaluation)
  - [Part 2: Bottom-Up Approach](#part-2-bottom-up-approach)
    - [1. Data Encoding and Scaling](#1-data-encoding-and-scaling)
    - [2. LSTM Model Development](#2-lstm-model-development)
    - [3. Model Evaluation](#3-model-evaluation-1)
  - [Ensemble Modeling](#ensemble-modeling)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Author](#author)

## Overview

This project focuses on forecasting the daily sales data of a US retailer at the state level over the next 12 months using historical sales data. The analysis employs both statistical and machine learning approaches to ensure robust and accurate predictions.

## Dataset

The dataset contains daily sales records of a US retailer with the following key features:
- *Order Date*: The date when the order was placed.
- *Ship Date*: The date when the order was shipped.
- *Sales*: The amount of sales for each order.
- *Sub-Category*: The sub-category of the product sold.
- *Category*: The main category of the product.
- *Customer ID*: Unique identifier for each customer.
- *State*: The US state where the sale was made.
- *Product ID*: Unique identifier for each product.
- *Product Name*: Name of the product.
- *Country*: Country of the sale.

Note: The dataset may contain missing values which are handled during the preprocessing phase.

## Objective

To forecast the total sales for each state over the next 12 months using historical sales data. This involves:
- Predicting sales data to a monthly granularity.
- Developing forecasting models using both statistical (SARIMA) and machine learning (LSTM) approaches.
- Evaluating and comparing model performances.
- Creating an ensemble model to leverage the strengths of individual models.

## Methodology

### Shared part: ETL
 1. Data Cleaning:
    - Handle missing values in 'Order Date' by estimating based on average shipping days.
    - Fill missing 'Sub-Category' values using the most frequent sub-category within the same 'Category'.
    - Impute missing 'State' information based on 'Customer ID' and 'Country'.
 2. Data Transformation:
    - Convert 'Order Date' and 'Ship Date' to datetime format.
    - Create a new 'Year-Month' column for monthly aggregation.

### Part 1: Top-Down Approach

#### 1. Data Aggregation
- *Objective*: Aggregate daily sales data to a year-month level.
- *Steps*:
  - Convert Order Date and Ship Date to datetime format.
  - Create a new column Year-Month for monthly aggregation.
  - Aggregate sales data by Year-Month and State.

#### 2. SARIMA Model Training
- *Objective*: Train a Seasonal ARIMA (SARIMA) model to forecast monthly sales.
- *Steps*:
  - Define SARIMA parameters considering seasonality (monthly data).
  - Fit the SARIMA model on the training data.
  - Handle warnings related to parameter estimation.

#### 3. Sales Disaggregation
- *Objective*: Disaggregate the aggregated sales forecasts back to the state level.
- *Steps*:
  - Calculate the proportion of sales by state for each month.
  - Allocate total forecasted sales to each state based on these proportions.

#### 4. Model Evaluation
- *Objective*: Assess the accuracy of the SARIMA model.
- *Steps*:
  - Generate forecasts for the next 12 months.
  - Plot forecasted vs actual sales.

### Part 2: Bottom-Up Approach

#### 1. Data Encoding and Scaling
- *Objective*: Prepare data for LSTM modeling.
- *Steps*:
  - Aggregate sales data by Year-Month, State, and Sub-Category.
  - Apply one-hot encoding to categorical variables.
  - Scale sales data using MinMaxScaler to normalize the values.

#### 2. LSTM Model Development
- *Objective*: Develop and train an LSTM neural network for sales forecasting.
- *Steps*:
  - Split data into training, validation, and test sets.
  - Define the LSTM architecture with multiple layers, dropout, batch normalization, and regularization.
  - Compile and train the model with early stopping to prevent overfitting.

#### 3. Model Evaluation
- *Objective*: Evaluate the performance of the LSTM model.
- *Steps*:
  - Plot training and validation MAE and loss.
  - Generate predictions on the test set.
  - Inverse transform the scaled predictions to original scale.
  - Plot predicted vs actual sales for each state.

### Ensemble Modeling

- *Objective*: Combine SARIMA and LSTM predictions to enhance forecasting accuracy.
- *Steps*:
  - Merge predictions from both models.
  - Calculate a weighted ensemble of the predictions.
  - Evaluate the ensemble model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
  - Save evaluation metrics and plots for analysis.

## Results

The project provides comprehensive forecasts of monthly sales at the state level using both SARIMA and LSTM models. Additionally, an ensemble model combining both approaches offers improved accuracy. Detailed plots and evaluation metrics are generated for each state to facilitate performance comparison.

## Evaluation Metrics

For each forecasting model and the ensemble, the following metrics are calculated:
- *Mean Squared Error (MSE)*: Measures the average of the squares of the errors.
- *Root Mean Squared Error (RMSE)*: The square root of MSE, providing error magnitude in the same units as the target variable.
- *Mean Absolute Error (MAE)*: The average of absolute errors, offering a straightforward measure of prediction accuracy.

These metrics are computed for both individual models and the ensemble to assess performance comprehensively.

## Project Structure

Retailer-Sales-Forecasting/
├── data/
│ └── Retailer Sales Data.csv
├── figures/
│ ├── sarima_state_sales_figures/
│ ├── lstm_state_sales_figures/
│ └── ensemble_state_sales_figures/
├── models/
│ └── lstm_model.h5
├── notebooks/
│ └── Retailer Sales Data Forecasting.ipynb
├── outputs/
│ ├── models_results_outputs.csv
│ └── models_results_outputs_state.csv
├── README.md
└── requirements.txt



- *data/*: Contains the raw dataset.
- *figures/*: Stores all generated plots for SARIMA, LSTM, and Ensemble models.
- *models/*: Saves trained machine learning models.
- *notebooks/*: Jupyter notebook with the complete analysis.
- *outputs/*: CSV files with evaluation metrics.
- *README.md*: This documentation.
- *requirements.txt*: List of Python dependencies.

## Installation

1. *Clone the Repository*
    bash
    git clone https://github.com/yourusername/Retailer-Sales-Forecasting.git
    cd Retailer-Sales-Forecasting
    

2. *Create a Virtual Environment*
    bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    

3. *Install Dependencies*
    bash
    pip install -r requirements.txt
    

## Usage

1. *Launch Jupyter Notebook*
    bash
    jupyter notebook
    

2. *Open the Notebook*
    - Navigate to notebooks/Retailer Sales Data Forecasting.ipynb.

3. *Run the Notebook*
    - Execute each cell sequentially to perform data preprocessing, modeling, evaluation, and visualization.

4. *View Results*
    - Generated plots will be saved in the respective directories within figures/.
    - Evaluation metrics are available in CSV files within the outputs/ directory.

## Dependencies

The project relies on the following Python libraries:

- *Data Handling and Manipulation*
  - pandas
  - numpy
  - os

- *Visualization*
  - matplotlib
  - visualkeras

- *Machine Learning and Modeling*
  - scikit-learn
  - statsmodels
  - keras (with TensorFlow backend)

- *Others*
  - keras.callbacks.EarlyStopping

Ensure all dependencies are installed by running:
bash
pip install -r requirements.txt


## Author

- *Elena Abcc*  
  Email: elenaabcc@example.com  
  GitHub: [github.com/elenaabcc](https://github.com/elenaabcc)