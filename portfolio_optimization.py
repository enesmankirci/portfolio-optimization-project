# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:32:40 2023

@author: enesm
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'SPY']

aapl_data = pd.read_csv("AAPL.csv")
aapl_data = aapl_data.rename(columns={'adjclose': 'AAPL'})
aapl_data = aapl_data[['date', 'AAPL']]

googl_data = pd.read_csv("GOOGL.csv")
googl_data = googl_data.rename(columns={'adjclose': 'GOOGL'})
googl_data = googl_data[['date', 'GOOGL']]

msft_data = pd.read_csv("MSFT.csv")
msft_data = msft_data.rename(columns={'adjclose': 'MSFT'})
msft_data = msft_data[['date', 'MSFT']]

amzn_data = pd.read_csv("AMZN.csv")
amzn_data = amzn_data.rename(columns={'adjclose': 'AMZN'})
amzn_data = amzn_data[['date', 'AMZN']]

fb_data = pd.read_csv("FB.csv")
fb_data = fb_data.rename(columns={'adjclose': 'FB'})
fb_data = fb_data[['date', 'FB']]

spy_data = pd.read_csv("SPY.csv")
spy_data = spy_data.rename(columns={'adjclose': 'SPY'})
spy_data = spy_data[['date', 'SPY']]

# Merge the dataframes
merged_data = aapl_data.merge(googl_data, on='date')
merged_data = merged_data.merge(amzn_data, on='date')
merged_data = merged_data.merge(msft_data, on='date')
merged_data = merged_data.merge(fb_data, on='date')
merged_data = merged_data.merge(spy_data, on='date')

# Convert 'date' column to datetime format
merged_data['date'] = pd.to_datetime(merged_data['date'])

# Set the 'date' column as the index
merged_data = merged_data.set_index('date')

# Display the merged data
pd.set_option('display.max_columns', None)
print(merged_data.head())


# Resample data to monthly frequency
monthly_data = merged_data.resample('M').last()

# Calculate the monthly returns for each stock
monthly_returns = monthly_data.pct_change()

# Calculate the expected returns of each stock
expected_returns = monthly_returns.mean()

# Calculate the covariance matrix of the portfolio
cov_matrix = monthly_returns.cov()

# Generate random weights for the portfolio
rng = np.random.default_rng()
weights = rng.random(len(stocks))
weights /= np.sum(weights)

# Calculate the expected return and volatility of the portfolio
trading_months = len(monthly_data.index)  # Calculate the actual number of trading months
expected_return = np.sum(expected_returns * weights) * trading_months
volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_months)

# Calculate the Sharpe ratio of the portfolio
sharpe_ratio = expected_return / volatility

# Simulate random portfolios and calculate Sharpe ratio for each portfolio
num_portfolios = 1000
sharpe_ratios = np.zeros(num_portfolios)
all_weights = np.zeros((num_portfolios, len(stocks)))

for i in range(num_portfolios):
    # Generate random weights for the portfolio
    weights = rng.random(len(stocks))
    weights /= np.sum(weights)  # Ensure weights sum to 1.0

    # Calculate expected returns and volatility for the portfolio
    expected_returns = np.sum(monthly_returns.mean() * weights) * trading_months
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_months)
    sharpe_ratios[i] = expected_returns / volatility

    # Store the weights for the portfolio
    all_weights[i, :] = weights

# Find the optimal portfolio
max_sharpe_idx = sharpe_ratios.argmax()
optimal_weights = all_weights[max_sharpe_idx, :]

# Calculate the expected return and volatility of the optimal portfolio
expected_return = np.sum(expected_returns * optimal_weights) * trading_months
volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))) * np.sqrt(trading_months)
sharpe_ratio = expected_return / volatility

# Display the details of the optimal portfolio
print("Optimal Weights:", optimal_weights)
print("Expected Return:", expected_return)
print("Volatility:", volatility)
print("Sharpe Ratio:", sharpe_ratio)




























