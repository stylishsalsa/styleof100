import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your data
df = pd.read_csv('your_data.csv', parse_dates=['Date'], index_col='Date')
df = df.sort_index()
series = df['Value']

# Split into training data (all except last value)
train = series[:-1]
actual_today = series.iloc[-1]
today_date = series.index[-1]

# Fit ARIMA model (you may need to tune order=(p,d,q))
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Predict today's value with confidence interval
forecast_result = model_fit.get_forecast(steps=1)
predicted = forecast_result.predicted_mean.iloc[0]
conf_int = forecast_result.conf_int(alpha=0.05).iloc[0]  # 95% CI

# Report
print(f"Date: {today_date.date()}")
print(f"Predicted value: {predicted:.2f}")
print(f"Actual value: {actual_today:.2f}")
print(f"95% Confidence Interval: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]")

# Check if the value is out of expected range
if actual_today < conf_int[0]:
    print(" Actual value is LOWER than expected.")
elif actual_today > conf_int[1]:
    print(" Actual value is HIGHER than expected.")
else:
    print(" Actual value is within expected range.")

# Visualize
plt.figure(figsize=(8, 4))
plt.plot(series[-30:], label='Recent Values')
plt.axhline(predicted, color='red', linestyle='--', label='Predicted Today')
plt.fill_between([today_date], conf_int[0], conf_int[1], color='red', alpha=0.3, label='95% CI')
plt.scatter(today_date, actual_today, color='black', label='Actual Today')
plt.legend()
plt.title('Today\'s Prediction vs Actual')
plt.show()
