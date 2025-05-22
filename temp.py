import pandas as pd
import numpy as np

def generate_custom_dataframe():
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    values = []

    for i in range(100):
        if i % 14 == 0:
            high_values = np.random.normal(loc=120, scale=10, size=5)
            low_values = np.random.normal(loc=90, scale=10, size=9)
            mixed = np.concatenate([high_values, low_values])
            np.random.shuffle(mixed)
            window = mixed.tolist()
        values.append(window[i % 14])

    df = pd.DataFrame({'date': dates, 'value': values})
    return df

def conditional_moving_average(df, window=7, threshold=105):
    df = df.copy()
    moving_averages = []

    for i in range(len(df)):
        current_value = df.loc[i, 'value']
        category = 'high' if current_value >= threshold else 'low'

        # Get past values (not including current)
        past_values = df.loc[:i-1, 'value']
        
        # Filter past values by category
        if category == 'high':
            category_values = past_values[past_values >= threshold]
        else:
            category_values = past_values[past_values < threshold]

        # Take the last `window` number of matching values
        recent_values = category_values.tail(window)

        if len(recent_values) > 0:
            moving_avg = recent_values.mean()
        else:
            moving_avg = np.nan

        moving_averages.append(moving_avg)

    df['conditional_moving_avg'] = moving_averages
    return df

# Generate and apply conditional moving average
df = generate_custom_dataframe()
df_with_cond_avg = conditional_moving_average(df)

print(df_with_cond_avg.head(20))
