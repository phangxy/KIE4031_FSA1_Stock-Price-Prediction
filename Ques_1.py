import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Data Collection
df = yf.download("MSFT", start="2015-01-01", end="2025-12-31")
df.to_csv("msft_stock.csv")
print("Raw Data Head:\n", df.head())

# 2. Data Cleaning
df = pd.read_csv("msft_stock.csv", header=[0, 1], index_col=0, parse_dates=True)

# Keep only the Close price and flatten the column name if yfinance created a multi-index
if isinstance(df.columns, pd.MultiIndex):
    close_prices = df[[('Close', 'MSFT')]].copy()
    close_prices.columns = ['Close']
else:
    close_prices = df[['Close']].copy()

# Use forward fill instead of dropping to maintain the timeline
close_prices = close_prices.ffill()

# 3. Prevent Data Leakage: Find train split index first
train_end_index = int(len(close_prices) * 0.70)

# 4. Normalisation
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler ONLY on the training portion
scaler.fit(close_prices.iloc[:train_end_index])

# Transform ALL data using the rules learned only from the training set
scaled_data = scaler.transform(close_prices)

# 5. Format into sequences
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])  # past 60 days
        y.append(data[i, 0])                # next day's price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window_size=60)

# Reshape X for RNN input: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 6. Train, Validation, and Test Split
train_end = int(len(X) * 0.70)
val_end   = int(len(X) * 0.90)  # 70% + 20%

X_train = X[:train_end]
y_train = y[:train_end]

X_val   = X[train_end:val_end]
y_val   = y[train_end:val_end]

X_test  = X[val_end:]
y_test  = y[val_end:]

print(f"\nDataset Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

