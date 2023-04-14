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

# Set the 'Date' column as the index
merged_data = merged_data.set_index('date')

# Display the merged data
print(merged_data.head())

# Calculate the daily returns for each stock
daily_returns = merged_data.pct_change()

# Display the daily returns
print(daily_returns.head())

# Calculate the expected returns of each stock
expected_returns = daily_returns.mean()

# Calculate the covariance matrix of the portfolio
cov_matrix = daily_returns.cov()

# Generate random weights for the portfolio
weights = np.random.random(len(stocks))
weights /= np.sum(weights)

# Calculate the expected return and volatility of the portfolio
expected_return = np.sum(expected_returns * weights) * 252
volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))* np.sqrt(252)

# Calculate the Sharpe ratio of the portfolio
sharpe_ratio = expected_return / volatility

# Create arrays to store the values of Sharpe ratios and all possible weights
num_portfolios = 1000
sharpe_ratios = np.zeros(num_portfolios)
all_weights = np.zeros((num_portfolios, len(stocks)))

log_returns = np.log(merged_data / merged_data.shift(1))
trading_days = len(merged_data.index)


# Simulate random portfolios and calculate Sharpe ratio for each portfolio
for i in range(num_portfolios):
    # Generate random weights for the portfolio
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)  # Ensure weights sum to 1.0
    
    # Calculate expected returns, volatility, and Sharpe ratio for the portfolio
    expected_return = np.sum(log_returns.mean() * weights) * trading_days
    volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * trading_days, weights)))
    sharpe_ratios[i] = expected_return / volatility
    
    # Store the weights for the portfolio
    all_weights[i, :] = weights

# Find the optimal portfolio
max_sharpe_idx = sharpe_ratios.argmax()
optimal_weights = all_weights[max_sharpe_idx, :]

# Display the details of the optimal portfolio
print("Optimal Weights:", optimal_weights)
print("Expected Return:", expected_return)
print("Volatility:", volatility)
print("Sharpe Ratio:", sharpe_ratios[max_sharpe_idx])




























