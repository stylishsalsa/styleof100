import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your dataset
# Example: CSV file with a time series column (e.g., 'Date', 'Value')
df = pd.read_csv('your_timeseries_data.csv', parse_dates=['Date'], index_col='Date')

# Ensure the time series data is sorted
df = df.sort_index()

# Select only the value column
series = df['Value']

# Visualize the time series (optional)
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title('Time Series Data')
plt.show()

# Split into train and test
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit ARIMA model
# You might need to tune the (p, d, q) parameters
model = ARIMA(train, order=(5, 1, 0))  # (p,d,q)
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate forecast
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse:.4f}')

# Plot forecast vs actual
plt.figure(figsize=(10, 4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()
