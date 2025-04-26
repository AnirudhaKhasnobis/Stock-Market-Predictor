import requests
import pandas as pd
import os

API_KEY = '5KYZ6YDIWW25BZBG'  # Your API key
symbol = 'AAPL'  # Stock symbol (you can change this to any other stock)
interval = 'Daily'  # Interval for the stock data ('Daily', 'Weekly', 'Monthly')
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

def get_stock_data():
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        dates = list(time_series.keys())
        prices = [time_series[date] for date in dates]
        
        df = pd.DataFrame(prices, index=dates)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    else:
        print("Error fetching data.")
        return None

# Define the folder where you want to save the data
folder = r'C:\Users\tituo\OneDrive\Documents\Programming\Stock Market Predictions'

# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

# Save the data to the specified folder
df = get_stock_data()
if df is not None:
    file_path = os.path.join(folder, 'stock_data.csv')
    df.to_csv(file_path)
    print(f"Stock data saved to {file_path}")
else:
    print("Failed to retrieve stock data.")
