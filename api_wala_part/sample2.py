import requests
import pandas as pd
import matplotlib.pyplot as plt

# 1. API endpoint (you can replace this with another if needed)
url = "https://open.er-api.com/v6/latest/USD"

# 2. Make the request
response = requests.get(url)

# 3. Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # 4. Extract exchange rates
    rates_dict = data.get("rates", {})
    if not rates_dict:
        print("No rates found in the response.")
    else:
        # 5. Convert to DataFrame
        df = pd.DataFrame(list(rates_dict.items()), columns=["Currency", "Rate"])

        # 6. Add metadata columns
        df["Base"] = data.get("base", "USD")
        df["Date"] = data.get("date")

        # 7. Display the DataFrame
        print("Exchange Rate Data:")
        print(df.head())

        # 8. Plot a bar chart
        plt.figure(figsize=(14, 6))
        df_sorted = df.sort_values("Rate", ascending=False)  # Top 20 for readability
        plt.bar(df_sorted["Currency"], df_sorted["Rate"], color="skyblue")
        plt.title(f"Top 20 Exchange Rates vs {df['Base'][0]} on {df['Date'][0]}")
        plt.xlabel("Currency")
        plt.ylabel("Exchange Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
