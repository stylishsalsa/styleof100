import requests
import json
import pandas as pd

# Replace with your actual API endpoint
url = "https://api.exchangerate-api.com/v4/latest/USD"

# Make the request
response = requests.get(url)

# Check for successful response
if response.status_code == 200:
    data = response.json()  # Convert JSON to Python dict
    print("Top-level keys in the response:")
    print(data.keys())  # See what the structure looks like

    # Now inspect to find where your actual records are
    # For example, if data['shops'] contains the list of records:
    if "shops" in data and isinstance(data["shops"], list):
        df = pd.DataFrame(data["shops"])
        print("Data loaded into DataFrame:")
        print(df.head())
    else:
        print("You need to find the correct key where the data records are stored.")
else:
    print(f"Failed to fetch data: {response.status_code}")
