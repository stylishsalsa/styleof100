import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

def run_forecast_loop(filename='data.csv'):
    # Load existing data
    try:
        df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
    except FileNotFoundError:
        print("No data file found. Creating new one.")
        df = pd.DataFrame(columns=['Date', 'Value'])
        df.set_index('Date', inplace=True)

    while True:
        # Ensure there's enough data to train
        if len(df) < 10:
            print("⚠️ Not enough data. Please input at least 10 values to begin predictions.")

        # Predict next value using ARIMA if possible
        if len(df) >= 10:
            train = df['Value']
            model = ARIMA(train, order=(5, 1, 0))
            model_fit = model.fit()

            # Forecast next time step
            forecast_result = model_fit.get_forecast(steps=1)
            predicted = forecast_result.predicted_mean.iloc[0]
            conf_int = forecast_result.conf_int(alpha=0.05).iloc[0]

            print(f"\n🔮 Predicted next value: {predicted:.2f}")
            print(f"🔒 95% Confidence Interval: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
        else:
            predicted = None
            conf_int = None

        # Get user input for next value
        try:
            value = float(input("Enter actual value for the next day (or type 'exit' to quit): "))
        except ValueError:
            print("Exiting.")
            break

        # Determine next date
        if df.empty:
            next_date = pd.Timestamp.today().normalize()
        else:
            next_date = df.index[-1] + timedelta(days=1)

        # Append new value to dataframe
        df.loc[next_date] = value
        df.to_csv(filename)

        # Analyze the input vs prediction
        if predicted is not None:
            if value < conf_int[0]:
                print("⚠️ Value is LOWER than expected.")
            elif value > conf_int[1]:
                print("⚠️ Value is HIGHER than expected.")
            else:
                print("✅ Value is within expected range.")

        # Optional: visualize
        plt.figure(figsize=(8, 4))
        plt.plot(df[-30:].index, df[-30:].Value, label='Recent Values')
        plt.axhline(predicted, color='red', linestyle='--', label='Predicted Next')
        if predicted is not None:
            plt.fill_between([next_date], conf_int[0], conf_int[1], color='red', alpha=0.3, label='95% CI')
        plt.scatter(next_date, value, color='black', label='New Input')
        plt.legend()
        plt.title('ARIMA Prediction vs Input')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Run the loop
run_forecast_loop()
