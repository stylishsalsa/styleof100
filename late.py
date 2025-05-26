import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Reproducible
np.random.seed(42)

# Generate synthetic data with weekly pattern and anomalies
days = 100
weekly_period = 7
anomaly_chance = 0.05

dates = pd.date_range(start='2023-01-01', periods=days)
t = np.arange(days)
weekly_pattern = 10 + 5 * np.sin(2 * np.pi * t / weekly_period)
noise = np.random.normal(0, 1, days)
data = weekly_pattern + noise

# Inject anomalies
true_anomalies = np.random.rand(days) < anomaly_chance
data[true_anomalies] += np.random.choice([-15, 15], size=true_anomalies.sum())

df = pd.DataFrame({'date': dates, 'value': data})

# Apply sliding window with Local Outlier Factor
window_size = 30
lof_labels = np.full(days, False)  # Default: not anomalous

for i in range(window_size, days):
    window_data = df['value'].iloc[i - window_size:i].values.reshape(-1, 1)
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    labels = lof.fit_predict(window_data)

    # Only check the last point in the window (the current day)
    if labels[-1] == -1:
        lof_labels[i] = True

# Add to DataFrame
df['lof_anomaly'] = lof_labels

# Plotting results
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['value'], label='Value')
plt.scatter(df['date'][df['lof_anomaly']], df['value'][df['lof_anomaly']], 
            color='orange', label='LOF Anomaly', marker='x')
plt.title('Anomaly Detection using Sliding Window + LOF')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
