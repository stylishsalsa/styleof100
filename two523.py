import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load your dataset (replace with your CSV or DataFrame)
# It should have a datetime index and a single column for values
# Example: df = pd.read_csv("your_data.csv", parse_dates=['Date'], index_col='Date')
# Here's a synthetic example:
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = pd.Series(50 + np.random.normal(0, 1, size=100).cumsum(), index=date_range)
df = pd.DataFrame(data, columns=['value'])

# Plot the time series
df['value'].plot(title='Time Series Data', figsize=(10, 4))
plt.show()

# Split into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit the ARIMA model (p, d, q) - these should ideally be determined using AIC/BIC or auto_arima
model = ARIMA(train, order=(5, 1, 0))  # Example: AR(5), I(1), MA(0)
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# Plot the results
plt.figure(figsize=(10, 4))
plt.plot(train.index, train, label='Training')
plt.plot(test.index, test, label='Actual')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'RMSE: {rmse:.2f}')
