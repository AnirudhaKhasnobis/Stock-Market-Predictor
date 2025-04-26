import pandas as pd

# Load the stock data (make sure stock_data.csv is in the same directory as this script)
df = pd.read_csv('stock_data.csv')

# Print the first few rows to check column names and structure
print(df.head())

# Check if 'Date' or another column name is used for the date
# Assuming the date column is in the first column or is named 'timestamp'
if 'Date' not in df.columns:
    # If 'Date' is not found, use the first column (if it holds the date)
    df['Date'] = pd.to_datetime(df.iloc[:, 0])  # Adjust if the date column has a different name
else:
    df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)  # Set the Date column as the index

# Calculate 50-day moving average
df['50_MA'] = df['Close'].rolling(window=50).mean()

# Calculate Relative Strength Index (RSI)
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_rsi(df)

# Save the processed data
df.to_csv('processed_stock_data.csv')

print("Data Preprocessed and Saved to processed_stock_data.csv")
