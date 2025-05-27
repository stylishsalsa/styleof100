import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_json("complete_shops_january_2025.json")

# Sort by date for clean plotting
df = df.sort_values("date")

# List of (countrycode, state, category)
group_list = [
    ("AU", "Queensland", "cars"),
    ("IN", "Maharashtra", "clothes"),
    ("US", "Texas", "malls")
]

# Loop through each group
for country, state, category in group_list:
    # Filter data
    filtered = df[
        (df["countrycode"] == country) &
        (df["state"] == state) &
        (df["category"] == category)
    ]
    
    # Show the table
    print(f"\n=== {country} | {state} | {category} ===")
    print(filtered[["date", "number_of_specified_shops_and_outlets_of_the_category"]])

    # Plot the graph
    plt.figure(figsize=(8, 4))
    plt.plot(
        filtered["date"],
        filtered["number_of_specified_shops_and_outlets_of_the_category"],
        marker='o',
        linestyle='-'
    )
    plt.title(f"{category.title()} Shops in {state}, {country} (Jan 2025)")
    plt.xlabel("Date")
    plt.ylabel("Number of Shops")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
